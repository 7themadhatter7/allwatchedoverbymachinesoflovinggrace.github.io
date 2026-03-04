"""
02_log_rotation.py  —  Rotate and trim log files in sparky/logs/
Rules:
  - Any log > 50MB: compress and truncate
  - Any log > 10MB and older than 7 days: compress
  - Any log > 100MB: truncate to last 10000 lines regardless of age
  - mother_dialog_*.jsonl files older than 30 days: archive
  - Empty log files: delete
Runs every 12 hours.
"""
import gzip
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from _lib import MaintenanceScript, log, SPARKY, disk_free_pct

LOGS_DIR   = SPARKY / "logs"
ARCHIVE    = SPARKY / "archive" / "logs"
ARCHIVE.mkdir(parents=True, exist_ok=True)

MB          = 1024 * 1024
MAX_LINES   = 10_000
AGE_7_DAYS  = 7  * 24 * 3600
AGE_30_DAYS = 30 * 24 * 3600


def compress(path: Path) -> bool:
    """Gzip a file, replace original with .gz. Returns True on success."""
    gz_path = path.with_suffix(path.suffix + ".gz")
    try:
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        path.unlink()
        return True
    except Exception:
        if gz_path.exists():
            gz_path.unlink()
        return False


def truncate_to_tail(path: Path, n_lines: int = MAX_LINES):
    """Keep only the last n_lines of a log file."""
    try:
        lines = path.read_bytes().splitlines()
        if len(lines) > n_lines:
            path.write_bytes(b"\n".join(lines[-n_lines:]) + b"\n")
            return len(lines) - n_lines
    except Exception:
        pass
    return 0


class LogRotation(MaintenanceScript):
    NAME        = "log_rotation"
    DESCRIPTION = "Rotate, compress, and trim log files"
    INTERVAL    = 43200  # 12 hours

    def run(self) -> dict:
        if not LOGS_DIR.exists():
            return {"skipped": "logs dir not found"}

        now        = time.time()
        compressed = []
        truncated  = []
        archived   = []
        deleted    = []

        for f in LOGS_DIR.iterdir():
            if not f.is_file():
                continue

            size = f.stat().st_size
            age  = now - f.stat().st_mtime

            # Empty files
            if size == 0:
                f.unlink()
                deleted.append(f.name)
                continue

            # mother_dialog files older than 30 days -> archive
            if "mother_dialog_" in f.name and age > AGE_30_DAYS:
                dest = ARCHIVE / f.name
                shutil.move(str(f), str(dest))
                archived.append(f.name)
                continue

            # Very large logs -> truncate first regardless of age
            if size > 100 * MB:
                removed = truncate_to_tail(f)
                if removed:
                    truncated.append(f"{f.name} (-{removed} lines)")
                    log(self.NAME, f"Truncated {f.name}: removed {removed} lines")

            # Large + old -> compress
            elif size > 10 * MB and age > AGE_7_DAYS:
                if compress(f):
                    compressed.append(f.name)
                    log(self.NAME, f"Compressed {f.name} ({size//MB}MB)")

            # Large -> compress regardless of age
            elif size > 50 * MB:
                if compress(f):
                    compressed.append(f.name)
                    log(self.NAME, f"Compressed large {f.name} ({size//MB}MB)")

        log(self.NAME,
            f"compressed={len(compressed)} truncated={len(truncated)} "
            f"archived={len(archived)} deleted={len(deleted)}")
        return {
            "compressed": compressed, "truncated": truncated,
            "archived":   archived,   "deleted":   deleted,
        }


if __name__ == "__main__":
    LogRotation().execute()
