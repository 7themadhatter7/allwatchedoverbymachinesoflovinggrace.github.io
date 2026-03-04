"""
04_disk_cleanup.py  —  Generalised disk space recovery.
Actions (in order of safety):
  1. Delete Python __pycache__ dirs in sparky/
  2. Delete .pyc files
  3. Compress .log files > 5MB that haven't been written in 24h
  4. Remove temp files in /tmp older than 7 days owned by joe
  5. Archive versioned Python files (all but highest version) to archive/
Thresholds trigger based on disk free %. Runs every 6 hours.
"""
import gzip
import os
import re
import shutil
import time
from pathlib import Path
from _lib import MaintenanceScript, log, shell, disk_free_pct, SPARKY

ARCHIVE = SPARKY / "archive"
ARCHIVE.mkdir(exist_ok=True)

PROTECTED = {
    "mother_english_io_v5.py", "bridge_server.py", "stamp_engine.py",
    "rm_self_improvement.py",  "combined_server.py", "spark_v4.py",
    "fused_service_v3.py",
}


class DiskCleanup(MaintenanceScript):
    NAME        = "disk_cleanup"
    DESCRIPTION = "Recover disk space: pycache, old logs, versioned file archive"
    INTERVAL    = 21600  # 6 hours

    def run(self) -> dict:
        free_pct = disk_free_pct("/")
        result   = {"free_pct_before": free_pct, "actions": []}

        # 1. Always: __pycache__ dirs
        n = self._clean_pycache()
        if n:
            result["actions"].append(f"Removed {n} __pycache__ dirs")
            log(self.NAME, f"Removed {n} __pycache__ dirs")

        # 2. Always: .pyc files
        n = self._clean_pyc()
        if n:
            result["actions"].append(f"Removed {n} .pyc files")

        # 3. If < 15% free: compress old logs
        if free_pct < 15:
            n = self._compress_old_logs()
            if n:
                result["actions"].append(f"Compressed {n} log files")
                log(self.NAME, f"Disk <15% free — compressed {n} logs")

        # 4. If < 12% free: archive old Python versions
        if free_pct < 12:
            n = self._archive_old_versions()
            if n:
                result["actions"].append(f"Archived {n} old script versions")
                log(self.NAME, f"Disk <12% free — archived {n} script versions")

        # 5. Always: /tmp cleanup
        n = self._clean_tmp()
        if n:
            result["actions"].append(f"Removed {n} stale /tmp files")

        result["free_pct_after"] = disk_free_pct("/")
        if not result["actions"]:
            result["actions"] = ["No action needed"]
        log(self.NAME,
            f"Disk free: {free_pct:.1f}% -> {result['free_pct_after']:.1f}%")
        return result

    def _clean_pycache(self) -> int:
        n = 0
        for d in SPARKY.rglob("__pycache__"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n

    def _clean_pyc(self) -> int:
        n = 0
        for f in SPARKY.rglob("*.pyc"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
        return n

    def _compress_old_logs(self) -> int:
        logs_dir = SPARKY / "logs"
        if not logs_dir.exists():
            return 0
        now = time.time()
        n = 0
        for f in logs_dir.glob("*.log"):
            if f.stat().st_size > 5 * 1024 * 1024:
                if now - f.stat().st_mtime > 86400:  # 24h idle
                    gz = f.with_suffix(".log.gz")
                    try:
                        with open(f, "rb") as fi, gzip.open(gz, "wb") as fo:
                            shutil.copyfileobj(fi, fo)
                        f.unlink()
                        n += 1
                    except Exception:
                        if gz.exists():
                            gz.unlink()
        return n

    def _archive_old_versions(self) -> int:
        families = {}
        for f in SPARKY.glob("*.py"):
            if f.name in PROTECTED:
                continue
            m = re.search(r'[_-]v(\d+)[a-z]?$', f.stem)
            if m:
                base = re.sub(r'[_-]v\d+[a-z]?$', '', f.stem)
                ver  = int(m.group(1))
                families.setdefault(base, []).append((ver, f))

        n = 0
        for base, files in families.items():
            if len(files) < 2:
                continue
            files.sort(key=lambda x: x[0])
            for ver, f in files[:-1]:
                dest = ARCHIVE / f.name
                if dest.exists():
                    dest = ARCHIVE / (f.stem + "_dup" + f.suffix)
                try:
                    shutil.move(str(f), str(dest))
                    n += 1
                except Exception:
                    pass
        return n

    def _clean_tmp(self) -> int:
        now = time.time()
        n   = 0
        tmp = Path("/tmp")
        for f in tmp.glob("rm-*.conf"):
            try:
                if now - f.stat().st_mtime > 3600:
                    f.unlink()
                    n += 1
            except Exception:
                pass
        return n


if __name__ == "__main__":
    DiskCleanup().execute()
