"""
maintenance/_lib.py  —  shared utilities for all maintenance scripts
"""
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

MAINT_DIR  = Path(__file__).parent
STATE_DIR  = MAINT_DIR / "state"
LOG_DIR    = MAINT_DIR / "logs"
SPARKY     = Path("/home/joe/sparky")
STATE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Protected files — never moved, deleted, or modified by any maintenance script
PROTECTED = {
    "mother_english_io_v5.py",
    "bridge_server.py",
    "stamp_engine.py",
    "rm_self_improvement.py",
    "combined_server.py",
    "spark_v4.py",
    "fused_service_v3.py",
    "master.py",
    "_lib.py",
}


def log(script_name: str, msg: str, level: str = "INFO"):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}][{level:5s}][{script_name}] {msg}"
    print(line, flush=True)
    log_file = LOG_DIR / f"{script_name}.log"
    with open(log_file, "a") as f:
        f.write(line + "\n")


def shell(cmd: str, timeout: int = 30) -> tuple:
    """Run a shell command. Returns (stdout+stderr, returncode)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (r.stdout + r.stderr).strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1
    except Exception as e:
        return str(e), 1


def load_state(name: str) -> dict:
    f = STATE_DIR / f"{name}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def save_state(name: str, state: dict):
    f = STATE_DIR / f"{name}.json"
    state["_last_saved"] = datetime.now().isoformat()
    f.write_text(json.dumps(state, indent=2))


def disk_free_pct(path: str = "/") -> float:
    """Return free disk percentage for path."""
    out, rc = shell(f"df {path} | tail -1")
    if rc != 0:
        return 100.0
    parts = out.split()
    try:
        used_pct = int(parts[4].replace("%", ""))
        return 100.0 - used_pct
    except Exception:
        return 100.0


def mem_free_gb() -> float:
    """Return available RAM in GB."""
    out, _ = shell("grep MemAvailable /proc/meminfo")
    try:
        kb = int(out.split()[1])
        return round(kb / 1024 / 1024, 1)
    except Exception:
        return 0.0


class MaintenanceScript:
    """Base class for all maintenance scripts."""
    NAME        = "base"
    DESCRIPTION = ""
    # How often this script should run (seconds). 0 = every master cycle.
    INTERVAL    = 3600

    def __init__(self):
        self.state = load_state(self.NAME)

    def should_run(self) -> bool:
        if self.INTERVAL == 0:
            return True
        last = self.state.get("last_run_ts", 0)
        return (time.time() - last) >= self.INTERVAL

    def run(self) -> dict:
        raise NotImplementedError

    def execute(self) -> dict:
        """Called by master. Wraps run() with timing, state, error handling.

        Scripts that define MAX_RUN_SECONDS are launched as hard-isolated
        subprocesses. This ensures child processes (e.g. ProcessPoolExecutor
        workers) are also killed on timeout, keeping the master non-blocking.
        """
        import subprocess as _sp, glob as _glob, sys
        max_s = getattr(self, "MAX_RUN_SECONDS", None)
        start = time.time()
        result = {"script": self.NAME, "ok": False, "ts": datetime.now().isoformat()}

        if max_s:
            # Find script file by NAME
            script_dir = Path(__file__).parent
            matches = sorted(_glob.glob(str(script_dir / f"*{self.NAME}*.py")))
            script_path = matches[0] if matches else None

            if script_path:
                try:
                    proc = _sp.Popen(
                        [sys.executable, script_path, "--run-once"],
                        stdout=_sp.PIPE, stderr=_sp.PIPE,
                        cwd=str(script_dir.parent)
                    )
                    try:
                        stdout, stderr = proc.communicate(timeout=max_s)
                        result["ok"] = proc.returncode == 0
                        if stderr:
                            result["stderr"] = stderr.decode(errors="replace")[-300:]
                    except _sp.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                        result["error"] = f"Timeout after {max_s}s (subprocess killed)"
                        log(self.NAME, f"Timed out after {max_s}s — killed", "WARN")
                    return result
                except Exception as e:
                    log(self.NAME, f"Subprocess launch failed: {e} — running inline", "WARN")

        # Inline execution (no MAX_RUN_SECONDS, or subprocess launch failed)
        try:
            result.update(self.run())
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()[-500:]
            log(self.NAME, f"ERROR: {e}", "ERROR")

        result["duration_s"] = round(time.time() - start, 2)
        self.state["last_run_ts"]     = time.time()
        self.state["last_run_result"] = result
        save_state(self.NAME, self.state)
        return result
