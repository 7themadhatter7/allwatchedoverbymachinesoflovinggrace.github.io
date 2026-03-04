#!/usr/bin/env python3
"""
RM Script 07 — ARCY node integration
Establishes minimal integration between SPARKY and ARCY (100.127.59.111):
  1. Health check loop (every 60s)  
  2. Task offload: send Ollama inference requests to ARCY when SPARKY GPU busy
  3. Shared state file via Tailscale for coordination
Run as daemon: nohup python3 07_arcy_integration.py >> logs/arcy.log 2>&1 &
"""
import json
import time
import threading
import urllib.request
import threading as _th
_state_lock = _th.Lock()
from datetime import datetime
from pathlib import Path

ARCY_IP        = "100.127.59.111"
ARCY_OLLAMA    = f"http://{ARCY_IP}:11434"
ARCY_BRIDGE    = f"http://{ARCY_IP}:9090"
STATE_FILE     = Path("/home/joe/sparky/rm_diagnostics/arcy_state.json")
LOG_FILE       = Path("/home/joe/sparky/logs/arcy.log")
LOG_FILE.parent.mkdir(exist_ok=True)

_arcy_state = {
    "reachable":   False,
    "last_check":  None,
    "last_success":None,
    "models":      [],
    "check_count": 0,
    "fail_count":  0,
}


def log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][ARCY] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def arcy_health_check() -> bool:
    """Check if ARCY Ollama is reachable. Returns True if alive."""
    try:
        req = urllib.request.Request(f"{ARCY_OLLAMA}/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        with _state_lock:
            _arcy_state["reachable"] = True
            _arcy_state["last_success"] = datetime.now().isoformat()
        _arcy_state["models"]       = models
        return True
    except Exception as e:
        with _state_lock:
            _arcy_state["reachable"] = False
            _arcy_state["fail_count"] += 1
        return False


def arcy_generate(prompt: str, model: str = None) -> str:
    """
    Send an inference request to ARCY Ollama.
    Returns response text or raises if ARCY unreachable.
    """
    if not _arcy_state["reachable"]:
        raise RuntimeError("ARCY not reachable")

    # Use first available model if none specified
    if model is None:
        if not _arcy_state["models"]:
            raise RuntimeError("No models on ARCY")
        model = _arcy_state["models"][0]

    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{ARCY_OLLAMA}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    return result.get("response", "")


def save_state():
    _arcy_state["last_check"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(_arcy_state, indent=2))


def health_loop():
    """Background loop: check ARCY every 60s, log state changes."""
    last_reachable = None
    while True:
        was_reachable = last_reachable
        is_reachable  = arcy_health_check()
        _arcy_state["check_count"] += 1

        if is_reachable != was_reachable:
            if is_reachable:
                log(f"ARCY ONLINE — models: {_arcy_state['models']}")
            else:
                log("ARCY OFFLINE — cannot reach Ollama on :11434")

        last_reachable = is_reachable
        save_state()
        time.sleep(60)


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        alive = arcy_health_check()
        save_state()
        print(f"ARCY reachable: {alive}")
        print(f"Models: {_arcy_state['models']}")
        print(f"State: {STATE_FILE}")

    elif "--test" in sys.argv:
        # Single inference test
        arcy_health_check()
        if _arcy_state["reachable"]:
            try:
                r = arcy_generate("Hello, this is SPARKY testing ARCY. Reply briefly.")
                print(f"ARCY response: {r[:200]}")
            except Exception as e:
                print(f"Inference failed: {e}")
        else:
            print("ARCY unreachable — cannot test inference")
            print("Check: is Ollama running on ARCY? sudo systemctl start ollama")

    else:
        # Run as daemon
        log(f"ARCY integration daemon starting (target: {ARCY_IP})")
        t = threading.Thread(target=health_loop, daemon=True)
        t.start()
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            log("Daemon stopped")
