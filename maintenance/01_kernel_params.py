"""
01_kernel_params.py  —  Keep OS kernel parameters at RM-optimal values.
Reads live sysctl, compares to targets, applies any drift via sudo sysctl.
Also writes /etc/sysctl.d/99-rm-tuning.conf for reboot persistence.
Runs every 6 hours. Safe to re-run any number of times.
"""
import subprocess
from pathlib import Path
from _lib import MaintenanceScript, log, shell, save_state, SPARKY

SYSCTL_CONF = Path("/etc/sysctl.d/99-rm-tuning.conf")

# Target values — reasoned for 121GB RAM, bridge_server workload, E8 substrate
TARGETS = {
    "vm.swappiness":                "1",
    "vm.dirty_ratio":               "10",
    "vm.dirty_background_ratio":    "3",
    "vm.nr_hugepages":              "512",
    "net.core.somaxconn":           "8192",
    "net.core.netdev_max_backlog":  "4096",
    "net.ipv4.tcp_tw_reuse":        "1",
    "net.ipv4.tcp_fin_timeout":     "15",
}


class KernelParams(MaintenanceScript):
    NAME        = "kernel_params"
    DESCRIPTION = "Keep kernel params at RM-optimal values; persist across reboots"
    INTERVAL    = 21600  # 6 hours

    def run(self) -> dict:
        applied = []
        skipped = []
        failed  = []

        for param, target in TARGETS.items():
            current, rc = shell(f"sysctl -n {param}")
            if rc != 0:
                log(self.NAME, f"Cannot read {param}", "WARN")
                continue

            if current.strip() == target:
                skipped.append(param)
                continue

            out, rc = shell(f"sudo sysctl -w {param}={target}")
            if rc == 0:
                applied.append(f"{param}: {current.strip()} -> {target}")
                log(self.NAME, f"Applied {param}={target}")
            else:
                failed.append(param)
                log(self.NAME, f"Failed {param}: {out}", "WARN")

        # Write persistence file
        conf_lines = [
            "# RM OS tuning — Ghost in the Machine Labs",
            "# Managed by maintenance/01_kernel_params.py",
        ]
        for p, v in TARGETS.items():
            conf_lines.append(f"{p}={v}")
        conf_content = "\n".join(conf_lines) + "\n"

        if not SYSCTL_CONF.exists() or SYSCTL_CONF.read_text() != conf_content:
            try:
                # Write to tmp then sudo cp
                tmp = Path("/tmp/rm-sysctl.conf")
                tmp.write_text(conf_content)
                out, rc = shell(f"sudo cp {tmp} {SYSCTL_CONF} && sudo chmod 644 {SYSCTL_CONF}")
                if rc == 0:
                    log(self.NAME, f"Persistence file updated: {SYSCTL_CONF}")
                else:
                    log(self.NAME, f"Could not write persistence file: {out}", "WARN")
            except Exception as e:
                log(self.NAME, f"Persistence file error: {e}", "WARN")

        log(self.NAME,
            f"applied={len(applied)} skipped={len(skipped)} failed={len(failed)}")
        return {"applied": applied, "skipped": len(skipped), "failed": failed}


if __name__ == "__main__":
    KernelParams().execute()
