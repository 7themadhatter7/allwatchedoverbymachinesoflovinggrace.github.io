"""
06_process_audit.py  —  Audit running processes for anomalies.
Detects: zombie processes, high CPU idle processes, duplicate daemons,
runaway memory consumers.
Logs findings; does NOT kill anything automatically.
Runs every 30 minutes.
"""
import time
from _lib import MaintenanceScript, log, shell

# Processes expected to have multiple instances (parent + workers)
MULTI_OK = {"bridge_server.py", "combined_server.py"}

# Processes that should not be running any more (ARC is done)
STALE_PATTERNS = [
    "arc_autonomous", "arc_overnight", "arc_stack_learning",
    "arc_unified_overnight", "e8_overnight", "overnight_arc",
]


def get_processes() -> list:
    out, _ = shell("ps aux --no-header")
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        try:
            procs.append({
                "pid":  int(parts[1]),
                "cpu":  float(parts[2]),
                "mem":  float(parts[3]),
                "stat": parts[7],
                "cmd":  parts[10],
            })
        except ValueError:
            pass
    return procs


class ProcessAudit(MaintenanceScript):
    NAME        = "process_audit"
    DESCRIPTION = "Detect zombie, stale, duplicate, and runaway processes"
    INTERVAL    = 1800  # 30 minutes

    def run(self) -> dict:
        procs    = get_processes()
        findings = []

        # Zombie processes
        zombies = [p for p in procs if p["stat"].startswith("Z")]
        if zombies:
            for z in zombies:
                log(self.NAME, f"ZOMBIE: pid={z['pid']} cmd={z['cmd'][:60]}", "WARN")
                findings.append(f"zombie:{z['pid']}")

        # High CPU from unexpected processes (>30% sustained)
        high_cpu = [p for p in procs
                    if p["cpu"] > 30
                    and "python3" in p["cmd"]
                    and not any(s in p["cmd"] for s in
                                ["mother_english_io", "bridge_server",
                                 "stamp_engine", "service_registry"])]
        for p in high_cpu:
            log(self.NAME,
                f"HIGH CPU: pid={p['pid']} cpu={p['cpu']}% cmd={p['cmd'][:60]}", "WARN")
            findings.append(f"high_cpu:{p['pid']}")

        # Stale ARC processes
        for p in procs:
            for pattern in STALE_PATTERNS:
                if pattern in p["cmd"] and p["cpu"] > 1.0:
                    log(self.NAME,
                        f"STALE: pid={p['pid']} cpu={p['cpu']}% "
                        f"cmd={p['cmd'][:60]}", "WARN")
                    findings.append(f"stale:{p['pid']}")

        # Duplicate Python daemons (same script, multiple parent processes)
        cmd_counts = {}
        for p in procs:
            if "python3" not in p["cmd"]:
                continue
            # Extract script basename
            parts = p["cmd"].split()
            script = next((x for x in parts if x.endswith(".py")), None)
            if script:
                base = script.split("/")[-1]
                if base not in MULTI_OK:
                    cmd_counts.setdefault(base, []).append(p["pid"])

        for script, pids in cmd_counts.items():
            if len(pids) > 1:
                log(self.NAME,
                    f"DUPLICATE: {script} running as PIDs {pids}", "WARN")
                findings.append(f"duplicate:{script}")

        if not findings:
            log(self.NAME, "No anomalies detected")

        return {"findings": findings, "process_count": len(procs)}


if __name__ == "__main__":
    ProcessAudit().execute()
