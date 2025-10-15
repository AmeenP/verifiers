import argparse
import asyncio
import contextlib
import importlib
import importlib.util
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, cast

import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers import setup_logging
from verifiers.types import Endpoints
from verifiers.utils.checkpoint import SimpleCheckpoint, RunConfig, _json_sha256
from verifiers.utils.client_utils import setup_client
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls

# Setup logger for eval script using verifiers logging format
logger = logging.getLogger("verifiers.scripts.eval")


def dataset_fingerprint(dataset: Dataset) -> str:
    """Generate a fingerprint for a dataset."""
    # Use dataset info if available (HuggingFace datasets)
    if hasattr(dataset, "info") and hasattr(dataset.info, "config_name"):
        return f"{dataset.info.builder_name}@{dataset.info.config_name}"
    # Fall back to hash of first few rows
    sample = dataset[:min(5, len(dataset))]
    return _json_sha256(sample)


def resolve_indices(dataset: Dataset, num_examples: int, seed: int | None) -> list[int]:
    """Resolve dataset indices deterministically."""
    n = len(dataset)
    if num_examples > 0:
        n = min(num_examples, n)
    indices = list(range(n))
    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices


def aggregate_from_jsonl(path: Path) -> dict:
    """Aggregate results from a JSONL file."""
    rewards = []
    metrics_lists: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    rewards.append(row.get("metrics", {}).get("reward", 0.0))
                    for k, v in row.get("metrics", {}).items():
                        if k != "reward":
                            if k not in metrics_lists:
                                metrics_lists[k] = []
                            metrics_lists[k].append(v)
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
    result = {"rewards": rewards, "metrics": metrics_lists}
    if rewards:
        result["avg_reward"] = sum(rewards) / len(rewards)
        result["std_reward"] = np.std(rewards)
    return result


def default_run_dir(env_id: str, model_name: str) -> Path:
    """Generate default run directory path."""
    module_name = env_id.replace("-", "_")
    env_model_str = f"{env_id}--{model_name.replace('/', '--')}"
    uuid_str = str(uuid.uuid4())[:8]
    return Path("./outputs") / "evals" / env_model_str / uuid_str


def infer_stage(e: Exception) -> str:
    """Infer which stage an error occurred in."""
    error_str = str(e).lower()
    if "timeout" in error_str or "connection" in error_str:
        return "inference"
    if "parse" in error_str or "json" in error_str:
        return "parse"
    return "rubric"


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    verbose: bool,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
    extra_headers: Dict[str, str],
    # Simplified checkpoint parameters
    output_dir: str | None,
    checkpoint_every: int,
    seed: int,
):
    setup_logging("DEBUG" if verbose else "INFO")
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            logger.debug(f"Loading endpoint registry from {endpoints_file}")
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            # check that module exposes ENDPOINTS
            if not hasattr(endpoints_module, "ENDPOINTS"):
                raise AttributeError(
                    f"Module '{endpoints_file}' does not have a 'ENDPOINTS' attribute"
                )
            ENDPOINTS = cast(Endpoints, endpoints_module.ENDPOINTS)
            logger.debug(
                f"Successfully loaded {len(ENDPOINTS)} endpoints from registry"
            )
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"No local endpoint registry found at {endpoints_path}. "
            f"Please specify the model name (-m), API host base URL (-b), and API key variable name (-k). "
            f"Error details: {str(e)}"
        )
        logger.debug("Using default empty endpoints registry")
        ENDPOINTS: Endpoints = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]
        logger.debug(f"Using endpoint configuration for model '{model}' from registry")
    else:
        logger.debug(
            f"Model '{model}' not found in endpoint registry, using command-line arguments"
        )

    # Setup eval client with high limits to prevent API timeout errors
    client = setup_client(
        api_base_url,
        api_key_var,
        timeout=3600.0,  # 1h
        max_connections=28000,  # Number of available ports
        max_keepalive_connections=28000,  # Number of available ports
        max_retries=10,  # 10 retries (w/ exponential backoffs)
        extra_headers=extra_headers,
    )
    logger.debug(f"Initialized OpenAI client with base_url: {api_base_url}")
    async_client = AsyncOpenAI(api_key=client.api_key, base_url=str(client.base_url))

    vf_env = vf.load_environment(env_id=env, **env_args)
    # Merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if sampling_args is not None:
        merged_sampling_args.update(sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = temperature

    # Get dataset and resolve indices
    if vf_env.eval_dataset is None:
        logger.info("eval_dataset is not set, falling back to train dataset")
        dataset = vf_env.get_dataset(n=num_examples, seed=seed)
    else:
        dataset = vf_env.get_eval_dataset(n=num_examples, seed=seed)

    # Build deterministic work keys: "idx/roll"
    indices = resolve_indices(dataset, num_examples, seed)
    all_keys = [f"{i}/{r}" for i in indices for r in range(rollouts_per_example)]

    # Build run configuration for checkpointing
    ds_fp = dataset_fingerprint(dataset)
    idx_sha = _json_sha256(indices)
    cfg = RunConfig(
        env_id=env,
        split="eval" if vf_env.eval_dataset is not None else "train",
        env_args=env_args,
        dataset_fingerprint=ds_fp,
        indices_sha256=idx_sha,
        model=model,
        sampling_args=merged_sampling_args,
        num_examples=len(indices),
        rollouts_per_example=rollouts_per_example,
        seed=seed,
        max_concurrent=max_concurrent,
        verifiers_version=vf.__version__,
        env_version=getattr(vf_env, "__version__", None),
    )

    # Setup output directory and checkpoint writer
    run_dir = Path(output_dir) if output_dir else default_run_dir(env, model)

    # SimpleCheckpoint handles resume automatically based on manifest
    cp = SimpleCheckpoint(run_dir, cfg, checkpoint_every=checkpoint_every)

    worklist = cp.pending_keys(all_keys)
    if not worklist:
        logger.info("All items already completed. Nothing to do.")
        print("All items already completed. Nothing to do.")
        return

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(
        f"Configuration: num_examples={num_examples}, rollouts_per_example={rollouts_per_example}, max_concurrent={max_concurrent}"
    )
    logger.info(f"Pending items: {len(worklist)} / {len(all_keys)}")
    start_time = time.time()

    # Run async evaluation with checkpointing
    async def run_evaluation():
        sem = asyncio.Semaphore(max_concurrent)

        async def process(key: str):
            idx, roll = map(int, key.split("/"))
            example = dataset[idx]
            async with sem:
                try:
                    prompt = example["prompt"]
                    answer = example.get("answer", "")
                    task = example.get("task", "default")
                    info = example.get("info", {})

                    # Run single rollout
                    completion, state = await vf_env.rollout(
                        client=async_client,
                        model=model,
                        prompt=prompt,
                        answer=answer,
                        task=task,
                        info=info,
                        sampling_args=merged_sampling_args,
                    )

                    # Score the rollout
                    rollout_score = await vf_env.rubric.score_rollout(
                        prompt=prompt,
                        completion=completion,
                        answer=answer,
                        state=state,
                        task=task,
                        info=info,
                    )

                    # Record success
                    await cp.queue.put(
                        (
                            "ok",
                            {
                                "key": key,
                                "idx": idx,
                                "rollout": roll,
                                "status": "ok",
                                "request": {"prompt": prompt, "sampling_args": merged_sampling_args},
                                "completion": sanitize_tool_calls(completion),
                                "parsed": state.get("parsed", {}),
                                "metrics": {
                                    "reward": rollout_score.reward,
                                    **rollout_score.metrics,
                                },
                                "timing": state.get("timing", {}),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            },
                        )
                    )
                except Exception as e:
                    # Always skip-on-error (simplified design)
                    await cp.queue.put(("error", {
                        "key": key,
                        "idx": idx,
                        "rollout": roll,
                        "status": "error",
                        "stage": infer_stage(e),
                        "error": f"{type(e).__name__}: {e}",
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }))

        writer_task = asyncio.create_task(cp.run(total_items=len(all_keys)))
        try:
            from tqdm.asyncio import tqdm_asyncio

            await tqdm_asyncio.gather(
                *[process(k) for k in worklist],
                total=len(worklist),
                desc="Running rollouts",
            )
        finally:
            writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer_task

    # Execute async evaluation
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio  # type: ignore

        nest_asyncio.apply()
        loop.run_until_complete(run_evaluation())
    except RuntimeError:
        asyncio.run(run_evaluation())

    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    # Aggregate results from JSONL
    summary = aggregate_from_jsonl(run_dir / "results.jsonl")

    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print(f"Output directory: {run_dir}")
    print()
    print("--- Summary ---")
    if "avg_reward" in summary:
        print(
            f"reward: avg - {summary['avg_reward']:.3f}, std - {summary['std_reward']:.3f}"
        )
        print(f"Total completed: {len(summary['rewards'])}")
        print(f"Total failed: {cp.num_failed}")
    else:
        print("No results found.")

    # Print per-rollout breakdown if we have enough data
    if "rewards" in summary and len(summary["rewards"]) >= rollouts_per_example:
        rewards = summary["rewards"]
        r = rollouts_per_example
        n = len(rewards) // r
        print("\nRewards by rollout:")
        for i in range(r):
            trials = [round(rewards[(i * n) + j], 3) for j in range(n)]
            print(f"r{i + 1}: {trials[:10]}{'...' if len(trials) > 10 else ''}")

    # Print metrics breakdown
    for k, v in summary.get("metrics", {}).items():
        if v:
            print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")

    # Note: --save-dataset and --save-to-hf-hub are not yet implemented with checkpointing
    # The results are already saved to the output directory in JSONL format
    if save_dataset or save_to_hf_hub:
        logger.warning(
            "--save-dataset and --save-to-hf-hub are not yet implemented with checkpointing. "
            f"Results are saved in JSONL format at {run_dir}"
        )

    # Exit with appropriate code
    if cp.num_failed > 0:
        pending = len(cp.pending_keys(all_keys))
        if pending == 0:
            sys.exit(1)  # completed with failures
        else:
            sys.exit(2)  # interrupted/partial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env", type=str, default="gsm8k", help="Environment module name"
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment module arguments as JSON object (e.g., \'{"key": "value", "num": 42}\')',
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="Extra HTTP header to pass to inference API. 'Name: Value'. Repeatable.",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=5,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=3,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (unset to use model default)",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help=(
            "Sampling arguments as JSON object. Keys here override --max-tokens/--temperature. "
            'Example: \'{"enable_thinking": false, "max_tokens": 256}\''
        ),
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--save-dataset",
        "-s",
        default=False,
        action="store_true",
        help="Save dataset to disk",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )

    # Checkpointing arguments (simplified)
    checkpoint_group = parser.add_argument_group("checkpointing")
    checkpoint_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write artifacts; if it already exists, resume automatically (default: auto-generated)",
    )
    checkpoint_group.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Rewrite failures + manifest every N finished items (default: 50)",
    )
    checkpoint_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset ordering (default: 42)",
    )

    args = parser.parse_args()

    # Build headers from repeated --header flags
    merged_headers: Dict[str, str] = {}
    for h in args.header or []:
        if ":" not in h:
            raise ValueError(f"--header must be 'Name: Value', got: {h!r}")
        k, v = h.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError("--header name cannot be empty")
        merged_headers[k] = v

    eval_environment(
        env=args.env,
        env_args=args.env_args,
        env_dir_path=args.env_dir_path,
        endpoints_path=args.endpoints_path,
        model=args.model,
        api_key_var=args.api_key_var,
        api_base_url=args.api_base_url,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sampling_args=args.sampling_args,
        verbose=args.verbose,
        save_dataset=args.save_dataset,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
        extra_headers=merged_headers,
        # Simplified checkpoint parameters
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
