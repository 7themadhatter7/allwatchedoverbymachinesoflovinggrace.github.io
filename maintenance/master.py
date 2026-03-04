#!/usr/bin/env python3
"""
master.py  —  RM Maintenance Orchestrator
Ghost in the Machine Labs

Runs all maintenance scripts on their individual schedules.
Operates as a daemon: loops every 60 seconds, checks which scripts
are due, runs them in sequence.

Usage:
  python3 master.py            # run daemon
  python3 master.py --once     # single pass, then exit
  python3 master.py --status   # print last run results and exit
  python3 master.py --run 03   # force-run a specific script by prefix

Cron entry (keep daemon alive if killed):
  * * * * * pgrep -f "maintenance/master.py" > /dev/null || \
            nohup python3 /home/joe/sparky/maintenance/master.py \
            >> /home/joe/sparky/maintenance/logs/master.log 2>&1 &
"""
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

MAINT_DIR  = Path(__file__).parent
LOG_FILE   = MAINT_DIR / "logs" / "master.log"
STATE_FILE = MAINT_DIR / "state" / "master.json"

# Ordered list of scripts to load (prefix determines order)
SCRIPT_GLOB = "0*.py"

_state = {}


def log(msg: str, level: str = "INFO"):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}][{level:5s}][MASTER] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_script(path: Path):
    """Dynamically import a maintenance script module."""
    spec   = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Add maintenance dir to path so _lib imports work
    sys.path.insert(0, str(MAINT_DIR))
    spec.loader.exec_module(module)
    return module


def discover_scripts() -> list:
    """Find all 0*.py scripts, return list of (path, class_instance)."""
    scripts = []
    for path in sorted(MAINT_DIR.glob(SCRIPT_GLOB)):
        if path.name == "master.py":
            continue
        try:
            mod = load_script(path)
            # Find the MaintenanceScript subclass
            from _lib import MaintenanceScript
            cls = next(
                (v for v in vars(mod).values()
                 if isinstance(v, type)
                 and issubclass(v, MaintenanceScript)
                 and v is not MaintenanceScript),
                None
            )
            if cls:
                scripts.append((path.name, cls()))
            else:
                log(f"No MaintenanceScript subclass in {path.name}", "WARN")
        except Exception as e:
            log(f"Failed to load {path.name}: {e}", "ERROR")
            traceback.print_exc()
    return scripts


def run_pass(scripts: list, force_prefix: str = None) -> dict:
    results = {}
    for name, script in scripts:
        # Force-run if prefix matches
        if force_prefix and not name.startswith(force_prefix):
            continue

        if force_prefix or script.should_run():
            log(f"Running {name} ({script.DESCRIPTION})")
            result = script.execute()
            results[name] = result
            status = "OK" if result.get("ok") else "FAIL"
            log(f"  {status}  {name}  ({result.get('duration_s', 0):.1f}s)")
        # else: not due yet — silent

    _state["last_pass_ts"] = datetime.now().isoformat()
    _state["last_results"] = results
    STATE_FILE.write_text(json.dumps(_state, indent=2))
    return results


def print_status(scripts: list):
    print(f"\n{'='*60}")
    print(f"  RM MAINTENANCE STATUS  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'='*60}")
    for name, script in scripts:
        state    = script.state
        last_run = state.get("last_run_ts")
        last_res = state.get("last_run_result", {})
        if last_run:
            import time as _t
            ago = int(_t.time() - last_run)
            ago_str = f"{ago//3600}h{(ago%3600)//60}m ago"
        else:
            ago_str = "never"
        status = "OK" if last_res.get("ok") else ("FAIL" if last_res else "---")
        due_in = max(0, int(script.INTERVAL - (time.time() - (last_run or 0))))
        print(f"\n  [{status}]  {name}")
        print(f"    {script.DESCRIPTION}")
        print(f"    Last run: {ago_str}  |  Next due: {due_in//60}m  |  "
              f"Interval: {script.INTERVAL//60}m")
        if last_res.get("error"):
            print(f"    Error: {last_res['error'][:80]}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    args = sys.argv[1:]

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    scripts = discover_scripts()
    log(f"Loaded {len(scripts)} maintenance scripts")

    if "--status" in args:
        print_status(scripts)
        sys.exit(0)

    force_prefix = None
    if "--run" in args:
        idx = args.index("--run")
        force_prefix = args[idx + 1] if idx + 1 < len(args) else None
        log(f"Force-running scripts matching prefix: {force_prefix}")
        run_pass(scripts, force_prefix=force_prefix)
        sys.exit(0)

    once = "--once" in args

    log("Maintenance daemon starting")
    for name, script in scripts:
        log(f"  {name}  interval={script.INTERVAL//60}m  {script.DESCRIPTION}")

    while True:
        try:
            results = run_pass(scripts)
            ran = [k for k, v in results.items() if v.get("ok")]
            if ran:
                log(f"Cycle complete: ran {ran}")
        except Exception as e:
            log(f"Cycle error: {e}", "ERROR")
            traceback.print_exc()

        if once:
            break

        time.sleep(60)
