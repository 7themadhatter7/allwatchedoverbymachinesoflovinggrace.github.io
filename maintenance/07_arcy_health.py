"""
07_arcy_health.py  —  Monitor ARCY node availability.
Checks Ollama reachability, records models available.
Does not attempt any inference — health only.
Runs every 10 minutes.
"""
import json
import urllib.request
from _lib import MaintenanceScript, log

ARCY_IP     = "100.127.59.111"
ARCY_OLLAMA = f"http://{ARCY_IP}:11434"


class ArcyHealth(MaintenanceScript):
    NAME        = "arcy_health"
    DESCRIPTION = "Monitor ARCY node (100.127.59.111) availability"
    INTERVAL    = 600  # 10 minutes

    def run(self) -> dict:
        last_status = self.state.get("reachable", None)

        try:
            req  = urllib.request.Request(f"{ARCY_OLLAMA}/api/tags")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            reachable = True
        except Exception as e:
            models    = []
            reachable = False
            err       = str(e)

        # Log only on state change
        if reachable != last_status:
            if reachable:
                log(self.NAME, f"ARCY ONLINE  models={models}")
            else:
                log(self.NAME, f"ARCY OFFLINE  error={err}", "WARN")

        self.state["reachable"] = reachable
        self.state["models"]    = models

        return {"reachable": reachable, "models": models}


if __name__ == "__main__":
    ArcyHealth().execute()
