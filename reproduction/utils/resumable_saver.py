"""Resumable incremental saver for long-running single-threaded generation jobs.

Each sample is tracked in a SQLite manifest with states:
``pending -> done`` or ``pending -> failed``. Completed samples are skipped
on resume. Corrupted or missing output files are detected via SHA-256 checksum
and automatically reset to ``pending``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union

PayloadT = TypeVar("PayloadT")


class SaveStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class Serializer(Protocol[PayloadT]):
    def dumps(self, payload: PayloadT) -> bytes: ...

    def loads(self, data: bytes) -> PayloadT: ...


class PickleSerializer(Serializer[Any]):
    """Default serializer using pickle."""

    def dumps(self, payload: Any) -> bytes:
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)


@dataclass(frozen=True)
class SaveRecord:
    sample_id: str
    status: SaveStatus
    output_path: Optional[str]
    checksum: Optional[str]
    error: Optional[str]
    meta: Optional[Dict[str, Any]]
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SaveRecord:
        meta_raw = row["meta_json"]
        meta = json.loads(meta_raw) if meta_raw else None
        return cls(
            sample_id=row["sample_id"],
            status=SaveStatus(row["status"]),
            output_path=row["output_path"],
            checksum=row["checksum"],
            error=row["error"],
            meta=meta,
            updated_at=float(row["updated_at"]),
        )


@dataclass
class RecoveryReport:
    done_missing_file_to_pending: int = 0
    done_checksum_mismatch_to_pending: int = 0
    failed_to_pending: int = 0

def _utc_now() -> float:
    return time.time()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)


class ResumableSaver:
    """Track and persist generated samples with resume/skip support.

    Not thread-safe. Designed for single-threaded generation loops where each
    sample is expensive to compute. Completed samples are skipped automatically
    on re-runs; failed samples can optionally be retried.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        serializer: Optional[Serializer[Any]] = None,
        retry_failed: bool = False,
        auto_recover: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "outputs"
        self.manifest_path = self.root_dir / "manifest.db"
        self.serializer = serializer or PickleSerializer()
        self.retry_failed = retry_failed

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.manifest_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        if auto_recover:
            self.recover_stale_records()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResumableSaver:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                sample_id   TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                output_path TEXT,
                checksum    TEXT,
                error       TEXT,
                meta_json   TEXT,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
            """
        )
        self._conn.commit()

    def _output_path_for(self, sample_id: str) -> Path:
        shard = sample_id[:2]
        return self.output_dir / shard / f"{sample_id}.pkl"

    def _verify_output_file(self, output_path: Path, checksum: Optional[str]) -> bool:
        if not output_path.is_file():
            return False
        if checksum is None:
            return True
        data = output_path.read_bytes()
        return hashlib.sha256(data).hexdigest() == checksum

    def is_done(self, sample_id: str) -> bool:
        record = self.get_record(sample_id)
        if record is None or record.status != SaveStatus.DONE or record.output_path is None:
            return False
        return self._verify_output_file(Path(record.output_path), record.checksum)

    def get_record(self, sample_id: str) -> Optional[SaveRecord]:
        row = self._conn.execute(
            "SELECT * FROM records WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        return SaveRecord.from_row(row) if row is not None else None

    def register_pending(self, sample_id: str, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Insert a sample as pending if it does not already exist."""
        meta_json = json.dumps(dict(meta)) if meta is not None else None
        self._conn.execute(
            """
            INSERT OR IGNORE INTO records(
                sample_id, status, output_path, checksum,
                error, meta_json, updated_at
            ) VALUES(?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (sample_id, SaveStatus.PENDING.value, meta_json, _utc_now()),
        )
        self._conn.commit()

    def save_success(
        self,
        sample_id: str,
        payload: Any,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Atomically persist payload and mark the sample as done."""
        output_path = self._output_path_for(sample_id)
        data = self.serializer.dumps(payload)
        checksum = hashlib.sha256(data).hexdigest()
        _atomic_write_bytes(output_path, data)

        meta_json = json.dumps(dict(meta)) if meta is not None else None
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO records(
                        sample_id, status, output_path, checksum,
                        error, meta_json, updated_at
                    ) VALUES(?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        sample_id,
                        SaveStatus.DONE.value,
                        str(output_path),
                        checksum,
                        meta_json,
                        _utc_now(),
                    ),
                )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return str(output_path)

    def save_failure(self, sample_id: str, error: Union[str, BaseException]) -> None:
        """Record a failed attempt."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO records(
                    sample_id, status, output_path, checksum,
                    error, meta_json, updated_at
                ) VALUES(?, ?, NULL, NULL, ?, NULL, ?)
                """,
                (sample_id, SaveStatus.FAILED.value, str(error), _utc_now()),
            )

    def run_sample(
        self,
        sample_id: str,
        fn: Callable[[], PayloadT],
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> PayloadT:
        """Skip if done; otherwise compute, persist, and return the payload.

        If ``fn`` raises, the failure is recorded and the exception re-raised.
        """
        if self.is_done(sample_id):
            return self.load(sample_id)  # type: ignore[return-value]
        try:
            payload = fn()
            self.save_success(sample_id, payload, meta=meta)
            return payload  # type: ignore[return-value]
        except Exception as exc:
            self.save_failure(sample_id, exc)
            raise

    def load(self, sample_id: str) -> Any:
        """Load a previously saved payload for a done sample."""
        record = self.get_record(sample_id)
        if record is None or record.status != SaveStatus.DONE or record.output_path is None:
            raise FileNotFoundError(f"No completed output found for sample_id={sample_id!r}")
        output_path = Path(record.output_path)
        if not self._verify_output_file(output_path, record.checksum):
            raise FileNotFoundError(f"Output file missing or invalid for sample_id={sample_id!r}")
        return self.serializer.loads(output_path.read_bytes())

    def recover_stale_records(self) -> RecoveryReport:
        """Reset inconsistent records to pending so they are retried.

        - ``done`` records with a missing output file are reset to ``pending``.
        - ``done`` records whose checksum does not match are reset to ``pending``
          and the corrupted file is deleted.
        - ``failed`` records are reset to ``pending`` when ``retry_failed=True``.
        """
        report = RecoveryReport()
        now = _utc_now()

        with self._conn:
            done_rows = self._conn.execute(
                "SELECT sample_id, output_path, checksum FROM records WHERE status = ?",
                (SaveStatus.DONE.value,),
            ).fetchall()
            for row in done_rows:
                output_path = Path(row["output_path"]) if row["output_path"] else None
                if output_path is None or not output_path.is_file():
                    self._conn.execute(
                        "UPDATE records SET status=?, output_path=NULL, checksum=NULL, updated_at=? WHERE sample_id=?",
                        (SaveStatus.PENDING.value, now, row["sample_id"]),
                    )
                    report.done_missing_file_to_pending += 1
                    continue

                if row["checksum"] is not None:
                    actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
                    if actual != row["checksum"]:
                        output_path.unlink(missing_ok=True)
                        self._conn.execute(
                            "UPDATE records SET status=?, output_path=NULL, checksum=NULL, updated_at=? WHERE sample_id=?",
                            (SaveStatus.PENDING.value, now, row["sample_id"]),
                        )
                        report.done_checksum_mismatch_to_pending += 1

            if self.retry_failed:
                cursor = self._conn.execute(
                    "UPDATE records SET status=?, error=NULL, updated_at=? WHERE status=?",
                    (SaveStatus.PENDING.value, now, SaveStatus.FAILED.value),
                )
                report.failed_to_pending = cursor.rowcount

        return report

    def list_records(self, status: Optional[SaveStatus] = None) -> List[SaveRecord]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM records ORDER BY updated_at ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM records WHERE status = ? ORDER BY updated_at ASC",
                (status.value,),
            ).fetchall()
        return [SaveRecord.from_row(row) for row in rows]

    def iter_todo(self) -> Iterator[str]:
        """Yield sample IDs that still need processing (pending, and optionally failed)."""
        statuses = [SaveStatus.PENDING.value]
        if self.retry_failed:
            statuses.append(SaveStatus.FAILED.value)
        placeholders = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT sample_id FROM records WHERE status IN ({placeholders}) ORDER BY updated_at ASC",
            statuses,
        ).fetchall()
        for row in rows:
            yield row["sample_id"]

    def stats(self) -> Dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) AS count FROM records GROUP BY status").fetchall()
        counts = {status.value: 0 for status in SaveStatus}
        for row in rows:
            counts[row["status"]] = int(row["count"])
        counts["total"] = sum(counts.values())
        return counts


def build_sample_id(*parts: Union[str, int], config_hash: Optional[str] = None) -> str:
    """Build a stable filesystem-safe sample identifier from key parts."""
    safe_parts = [
        str(part).replace("/", "_").replace("\\", "_").replace("::", "__")
        for part in parts
    ]
    normalized = "::".join(safe_parts)
    if config_hash:
        normalized = f"{normalized}::{config_hash}"
    return normalized


if __name__ == "__main__":
    # Two usage patterns are demonstrated below:
    #
    # Example 1 (register_pending + iter_todo): preferred for production batch jobs.
    # Pre-registering all sample IDs gives upfront totals via stats()/SQLite,
    # pairs cleanly with auto_recover + iter_todo for stale/failed retries, and
    # matches a task-board pattern that scales to multi-worker setups later.
    #
    # Example 2 (direct run_sample loop): shorter code for demos and quick local
    # experiments when you already have the full ID list and do not need a
    # persisted job queue or progress visibility before processing starts.

    root = Path(".resumable_saver_demo")
    config_hash = "demo_v1"
    sample_ids = [build_sample_id("pocket_a", i, config_hash=config_hash) for i in range(5)]

    with ResumableSaver(root) as saver:
        for sid in sample_ids:
            saver.register_pending(sid, meta={"pocket": "pocket_a"})

        for sid in saver.iter_todo():
            result = saver.run_sample(
                sid,
                lambda s=sid: {"sample_id": s, "coords": [1.0, 2.0, 3.0]},
                meta={"tag": "demo"},
            )
            print(f"  saved {sid[:8]}... -> {result}")

        stats = saver.stats()
        print(f"\nStats after run: {stats}")

        first_done = next((sid for sid in sample_ids if saver.is_done(sid)), None)
        if first_done is not None:
            loaded = saver.load(first_done)
            print(f"Loaded result for {first_done[:8]}...: {loaded}")

    # Example 2: direct loop when you already have the full ID list and do not
    # need upfront manifest stats or iter_todo(). For a persisted job queue, use
    # register_pending + iter_todo() as in Example 1 above.
    print("\n=== Example 2: direct loop (no pre-registration) ===")
    root_direct = Path(".resumable_saver_demo_direct")
    sample_ids_direct = [build_sample_id("pocket_b", i, config_hash=config_hash) for i in range(5)]

    with ResumableSaver(root_direct) as saver:
        for sid in sample_ids_direct:
            result = saver.run_sample(
                sid,
                lambda s=sid: {"sample_id": s, "coords": [1.0, 2.0, 3.0]},
                meta={"tag": "demo_direct"},
            )
            print(f"  saved {sid[:8]}... -> {result}")

        stats = saver.stats()
        print(f"\nStats after run: {stats}")
