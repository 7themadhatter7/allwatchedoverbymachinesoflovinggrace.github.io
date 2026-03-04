#!/usr/bin/env python3
"""
RM Script 04 — RAM-resident service registry
Lightweight HTTP service on port 8880.
Each RM service registers on startup, sends heartbeats every 30s.
Any service can query: GET /registry -> all known services + status.
Run as: python3 04_service_registry.py &
"""
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path

PORT     = 8880
REGISTRY = {}   # name -> {port, role, pid, last_seen, status}
_lock    = threading.Lock()

LOG = Path("/home/joe/sparky/logs/service_registry.log")
LOG.parent.mkdir(exist_ok=True)

def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][REGISTRY] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class RegistryHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, data, code=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/registry":
            with _lock:
                now = time.time()
                snapshot = {}
                for name, info in REGISTRY.items():
                    info2 = dict(info)
                    info2["alive"] = (now - info.get("last_seen", 0)) < 90
                    info2["age_s"] = int(now - info.get("last_seen", now))
                    snapshot[name] = info2
            self._json(snapshot)

        elif self.path == "/health":
            self._json({"status": "ok", "services": len(REGISTRY)})
        else:
            self._json({"error": "unknown"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/register":
            name = body.get("name","unknown")
            with _lock:
                REGISTRY[name] = {
                    "port":      body.get("port"),
                    "role":      body.get("role",""),
                    "pid":       body.get("pid"),
                    "last_seen": time.time(),
                    "status":    "registered",
                }
            log(f"Registered: {name} port={body.get('port')} pid={body.get('pid')}")
            self._json({"ok": True})

        elif self.path == "/heartbeat":
            name = body.get("name","unknown")
            with _lock:
                if name in REGISTRY:
                    REGISTRY[name]["last_seen"] = time.time()
                    REGISTRY[name]["status"] = body.get("status","alive")
            self._json({"ok": True})

        elif self.path == "/deregister":
            name = body.get("name","unknown")
            with _lock:
                REGISTRY.pop(name, None)
            log(f"Deregistered: {name}")
            self._json({"ok": True})
        else:
            self._json({"error": "unknown"}, 404)


def stale_checker():
    """Log stale services every 60s."""
    while True:
        time.sleep(60)
        with _lock:
            now = time.time()
            for name, info in REGISTRY.items():
                age = now - info.get("last_seen", now)
                if age > 90:
                    log(f"STALE: {name} last seen {int(age)}s ago")


if __name__ == "__main__":
    t = threading.Thread(target=stale_checker, daemon=True)
    t.start()

    # Self-register
    with _lock:
        REGISTRY["service_registry"] = {
            "port": PORT, "role": "RM service registry",
            "pid": __import__("os").getpid(), "last_seen": time.time(),
            "status": "running",
        }

    server = HTTPServer(("0.0.0.0", PORT), RegistryHandler)
    log(f"Service registry running on port {PORT}")
    log(f"  GET  http://localhost:{PORT}/registry")
    log(f"  POST http://localhost:{PORT}/register")
    log(f"  POST http://localhost:{PORT}/heartbeat")
    server.serve_forever()
