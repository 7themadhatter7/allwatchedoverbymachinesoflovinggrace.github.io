"""
05_session_memory.py  —  Consolidate dialog logs into substrate memory.
Reads all mother_dialog_*.jsonl files not yet processed,
extracts word co-occurrence pairs, pushes to mother /api/learn.
Falls back gracefully if endpoint not yet implemented.
Runs nightly (every 24h).
"""
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from _lib import MaintenanceScript, log, load_state, save_state, SPARKY

LOGS_DIR   = SPARKY / "logs"
MOTHER_URL = "http://localhost:8892"

STOP_WORDS = {
    'the','a','an','is','are','was','were','i','you','it','in','on','at',
    'to','of','and','or','but','for','with','that','this','my','me','we',
    'be','do','have','has','he','she','they','what','how','can','will',
    'just','not','so','all','its','as','if','by','up','out','about',
}


def extract_pairs(text: str, window: int = 4) -> Counter:
    words = [w.lower() for w in re.findall(r'[a-z]{3,}', text)
             if w.lower() not in STOP_WORDS]
    pairs = Counter()
    for i, w in enumerate(words):
        for j in range(i + 1, min(i + window + 1, len(words))):
            pairs[tuple(sorted([w, words[j]]))] += 1
    return pairs


def push_pairs(pairs: Counter, source: str) -> int:
    top = pairs.most_common(100)
    if not top:
        return 0
    payload = json.dumps({
        "type":      "association_reinforcement",
        "pairs":     [[a, b, c] for (a, b), c in top],
        "source":    source,
        "timestamp": datetime.now().isoformat(),
    }).encode()
    try:
        req = urllib.request.Request(
            f"{MOTHER_URL}/api/learn", data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        return result.get("pairs_absorbed", len(top))
    except Exception as e:
        log("session_memory", f"Push failed (endpoint may not exist yet): {e}", "WARN")
        return 0


class SessionMemory(MaintenanceScript):
    NAME        = "session_memory"
    DESCRIPTION = "Consolidate dialog logs into substrate co-occurrence memory"
    INTERVAL    = 86400  # 24 hours

    def run(self) -> dict:
        processed_files = set(self.state.get("processed_files", []))
        all_pairs       = Counter()
        new_files       = []

        for dialog_file in sorted(LOGS_DIR.glob("mother_dialog_*.jsonl")):
            if dialog_file.name in processed_files:
                continue
            if dialog_file.stat().st_size == 0:
                continue

            turns_processed = 0
            try:
                for line in dialog_file.read_text().splitlines():
                    try:
                        turn = json.loads(line)
                        text = (turn.get("content") or turn.get("text")
                                or turn.get("message") or str(turn))
                        all_pairs.update(extract_pairs(text))
                        turns_processed += 1
                    except Exception:
                        pass
            except Exception as e:
                log(self.NAME, f"Error reading {dialog_file.name}: {e}", "WARN")
                continue

            new_files.append(dialog_file.name)
            log(self.NAME, f"Read {dialog_file.name}: {turns_processed} turns")

        if not new_files:
            log(self.NAME, "No new dialog files to process")
            return {"new_files": 0, "pairs": 0, "absorbed": 0}

        absorbed = push_pairs(all_pairs, f"session_memory:{len(new_files)}_files")
        log(self.NAME,
            f"Processed {len(new_files)} files, "
            f"{len(all_pairs)} pairs, {absorbed} absorbed by mother")

        self.state["processed_files"] = list(processed_files | set(new_files))
        self.state["total_files"]     = len(self.state["processed_files"])
        self.state["total_pairs_sent"]= self.state.get("total_pairs_sent", 0) + absorbed

        return {"new_files": len(new_files), "pairs": len(all_pairs), "absorbed": absorbed}


if __name__ == "__main__":
    SessionMemory().execute()
