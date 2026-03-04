# RM Maintenance Scripts
# Ghost in the Machine Labs

## Structure
  master.py          — Orchestrator daemon (run this)
  _lib.py            — Shared utilities (imported by all scripts)
  01_kernel_params   — Sysctl tuning + persistence (6h)
  02_log_rotation    — Compress/trim logs (12h)
  03_service_watchdog — Verify services alive (5m)
  04_disk_cleanup    — pycache, old versions, tmp (6h)
  05_session_memory  — Dialog -> substrate memory (24h)
  06_process_audit   — Zombie/stale/duplicate process detection (30m)
  07_arcy_health     — ARCY node availability (10m)
  logs/              — Per-script log files
  state/             — Per-script JSON state (last run, counters)

## Usage
  # Daemon (recommended — kept alive by cron):
  nohup python3 /home/joe/sparky/maintenance/master.py >> maintenance/logs/master.log 2>&1 &

  # Status:
  python3 /home/joe/sparky/maintenance/master.py --status

  # Single pass:
  python3 /home/joe/sparky/maintenance/master.py --once

  # Force run one script:
  python3 /home/joe/sparky/maintenance/master.py --run 03

## Adding a new script
  1. Create 08_your_script.py in this folder
  2. Import MaintenanceScript from _lib
  3. Subclass it, set NAME/DESCRIPTION/INTERVAL, implement run() -> dict
  4. master.py discovers it automatically on next start
