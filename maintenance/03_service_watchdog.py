"""
03_service_watchdog.py  —  Verify critical RM services are alive.
Checks systemd status and port reachability.
Logs findings; does NOT auto-restart (systemd handles that).
Runs every 5 minutes.
"""
import socket
import time
from _lib import MaintenanceScript, log, shell

# service_name: {port, systemd_unit, critical}
SERVICES = {
    "mother":   {"port": 8892, "unit": "rm-mother.service",    "critical": True},
    "bridge":   {"port": 8787, "unit": "rm-bridge.service",    "critical": True},
    "registry": {"port": 8880, "unit": None,                   "critical": False},
}


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def systemd_active(unit: str) -> str:
    out, _ = shell(f"systemctl is-active {unit} 2>/dev/null")
    return out.strip()


class ServiceWatchdog(MaintenanceScript):
    NAME        = "service_watchdog"
    DESCRIPTION = "Verify critical RM services are alive"
    INTERVAL    = 300  # 5 minutes

    def run(self) -> dict:
        results   = {}
        problems  = []

        for name, cfg in SERVICES.items():
            port      = cfg["port"]
            unit      = cfg.get("unit")
            critical  = cfg.get("critical", False)

            port_alive    = port_open(port)
            systemd_state = systemd_active(unit) if unit else "n/a"
            healthy       = port_alive

            results[name] = {
                "port":     port,
                "port_ok":  port_alive,
                "systemd":  systemd_state,
                "healthy":  healthy,
            }

            if not healthy and critical:
                problems.append(name)
                log(self.NAME,
                    f"CRITICAL DOWN: {name} port={port} systemd={systemd_state}",
                    "ERROR")
            elif not healthy:
                log(self.NAME,
                    f"DOWN (non-critical): {name} port={port}", "WARN")

        if not problems:
            log(self.NAME, "All critical services healthy")

        return {"services": results, "problems": problems}


if __name__ == "__main__":
    ServiceWatchdog().execute()
