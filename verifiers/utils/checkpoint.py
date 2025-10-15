# verifiers/utils/checkpoint.py
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _json_sha256(obj: Any) -> str:
    """Compute SHA256 hash of a JSON-serializable object."""
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Atomically write a JSON object to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def _scan_success_keys(results_path: Path) -> Set[str]:
    """Scan results.jsonl for successful completion keys (ground truth for resume)."""
    done: Set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(Exception):
                row = json.loads(line)
                k = row.get("key")
                if isinstance(k, str):
                    done.add(k)
    return done


@dataclass
class RunConfig:
    """Configuration for a checkpoint run."""

    env_id: str
    split: str
    env_args: Dict[str, Any]
    dataset_fingerprint: str
    indices_sha256: str
    model: str
    sampling_args: Dict[str, Any]
    num_examples: int
    rollouts_per_example: int
    seed: int
    max_concurrent: int
    verifiers_version: str
    env_version: Optional[str] = None


class SimpleCheckpoint:
    """
    Single-writer checkpoint manager with simplified semantics:
    - Appends successes and failures immediately per-item (crash-safe)
    - Rewrites failures.jsonl at checkpoints to contain only *current* failures
    - Auto-resumes based on results.jsonl (ground truth)
    - Always skip-on-error (failures don't crash the run)
    """

    def __init__(
        self,
        out_dir: Path,
        run_cfg: RunConfig,
        checkpoint_every: int = 50,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results = out_dir / "results.jsonl"
        self.failures = out_dir / "failures.jsonl"
        self.manifest = out_dir / "manifest.json"
        self.cfg = asdict(run_cfg)
        self.signature = _json_sha256(self.cfg)

        # Validate signature if resuming
        if self.manifest.exists():
            prev = json.loads(self.manifest.read_text())
            if prev.get("signature") != self.signature:
                raise SystemExit(
                    "Output directory contains a different run (signature mismatch). "
                    "Choose a new --output-dir to start fresh."
                )

        # Resume: successes define ground truth of 'done'
        self.completed: Set[str] = _scan_success_keys(self.results)

        # Current failure records: key -> last error record
        self.fail_records: Dict[str, Dict[str, Any]] = {}

        # Single writer queue
        self.queue: "asyncio.Queue[Tuple[str, Dict[str, Any]]]" = asyncio.Queue()

        # Open files for append
        self._ok = self.results.open("a", encoding="utf-8", buffering=1)
        self._ko = self.failures.open("a", encoding="utf-8", buffering=1)

        self._checkpoint_every = max(1, int(checkpoint_every))
        self._since = 0

        # Initial manifest
        self._write_manifest(total_items=0)

    @property
    def num_done(self) -> int:
        """Total items processed (successes + failures)."""
        return len(self.completed) + len(self.fail_records)

    @property
    def num_failed(self) -> int:
        """Current failure count."""
        return len(self.fail_records)

    def pending_keys(self, all_keys: Iterable[str]) -> List[str]:
        """Return keys not yet completed (failures are retried automatically)."""
        return [k for k in all_keys if k not in self.completed]

    async def run(self, total_items: int) -> None:
        """Main writer loop: consume queue and write items."""
        try:
            while True:
                kind, rec = await self.queue.get()
                k = rec["key"]

                if kind == "ok":
                    # Append success immediately
                    self._ok.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._ok.flush()
                    os.fsync(self._ok.fileno())
                    self.completed.add(k)
                    # If it previously failed, forget the failure
                    self.fail_records.pop(k, None)
                else:
                    # Append failure immediately for crash-safety
                    self._ko.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._ko.flush()
                    os.fsync(self._ko.fileno())
                    self.fail_records[k] = rec

                self._since += 1
                if self._since >= self._checkpoint_every:
                    self._checkpoint(total_items)
        finally:
            # Final checkpoint on cancellation/exit
            self._checkpoint(total_items)
            self._ok.close()
            self._ko.close()

    def _checkpoint(self, total_items: int) -> None:
        """Rewrite failures.jsonl as snapshot of current failures + update manifest."""
        # 1) Rewrite failures.jsonl as a *snapshot* of current failures
        self._ko.close()  # close append handle before replacing the file

        snapshot = ""
        if self.fail_records:
            snapshot = "\n".join(
                json.dumps(v, ensure_ascii=False) for v in self.fail_records.values()
            ) + "\n"
        _atomic_write_text(self.failures, snapshot)

        # Reopen append handle for subsequent immediate failure writes
        self._ko = self.failures.open("a", encoding="utf-8", buffering=1)

        # 2) Update manifest atomically
        self._write_manifest(total_items)
        self._since = 0

    def _write_manifest(self, total_items: int) -> None:
        """Write manifest with current counters."""
        man = {
            "version": 1,
            "signature": self.signature,
            "config": self.cfg,
            "counters": {
                "total": total_items,
                "done": len(self.completed),
                "failed": len(self.fail_records),
            },
            "paths": {
                "results": str(self.results),
                "failures": str(self.failures),
            },
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _atomic_write_json(self.manifest, man)

    def immediate_flush(self, total_items: int):
        """Force an immediate checkpoint (for graceful shutdown)."""
        self._checkpoint(total_items)
