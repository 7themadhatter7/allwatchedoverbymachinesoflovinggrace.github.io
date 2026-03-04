#!/usr/bin/env python3
"""
RM Script 06 — Outbound push: RM notifies Joe of priority-1 findings.
Adds /api/push endpoint logic to bridge_server awareness.
Also provides a standalone pusher RM can call directly:
  python3 06_outbound_push.py "message"
Routes: Tailscale (preferred) -> ngrok tunnel -> log fallback.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PUSH_LOG   = Path("/home/joe/sparky/rm_diagnostics/push_log.jsonl")
PUSH_LOG.parent.mkdir(exist_ok=True)

# Delivery targets (in priority order)
TARGETS = [
    # Tailscale direct to Joe's devices — update IP as needed
    # {"type": "tailscale_http", "url": "http://100.x.x.x:9999/notify"},

    # Mother's own chat endpoint — pushes to active session
    {"type": "mother_chat", "url": "http://localhost:8892/api/chat"},

    # Log fallback — always works
    {"type": "log", "path": str(PUSH_LOG)},
]


def push(message: str, priority: int = 1, category: str = "rm_notification"):
    """
    Push a message outbound. Called by RM when she has something urgent.
    Returns True if delivered to at least one non-log target.
    """
    payload = {
        "ts":       datetime.now().isoformat(),
        "priority": priority,
        "category": category,
        "message":  message,
        "from":     "RM",
    }

    delivered = False
    for target in TARGETS:
        try:
            if target["type"] == "mother_chat":
                # Push as a self-observation into RM's own dialog
                body = json.dumps({
                    "message": f"[SELF-OBSERVATION P{priority}] {message}"
                }).encode()
                req = urllib.request.Request(
                    target["url"], data=body,
                    headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=5)
                delivered = True
                print(f"  Delivered via mother_chat")

            elif target["type"] == "log":
                with open(target["path"], "a") as f:
                    f.write(json.dumps(payload) + "\n")
                print(f"  Logged to {target['path']}")

        except Exception as e:
            print(f"  Target {target['type']} failed: {e}")

    return delivered


def push_pending_recommendations():
    """Push any unacknowledged P1/P2 recommendations."""
    recs_file = Path("/home/joe/sparky/rm_diagnostics/recommendations.jsonl")
    if not recs_file.exists():
        return

    acked_file = Path("/home/joe/sparky/rm_diagnostics/push_acked.json")
    acked = set()
    if acked_file.exists():
        acked = set(json.loads(acked_file.read_text()).get("acked", []))

    pushed = 0
    new_acked = set(acked)
    for line in recs_file.read_text().strip().splitlines():
        try:
            rec = json.loads(line)
            key = rec["title"][:60]
            if rec.get("priority", 9) <= 2 and key not in acked:
                push(f"{rec['title']} — {rec['action'][:100]}",
                     priority=rec["priority"],
                     category=rec["category"])
                new_acked.add(key)
                pushed += 1
        except Exception:
            pass

    acked_file.write_text(json.dumps({"acked": list(new_acked)}))
    print(f"Pushed {pushed} pending recommendations")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        delivered = push(msg)
        print(f"Push {'delivered' if delivered else 'logged only'}")
    else:
        push_pending_recommendations()
