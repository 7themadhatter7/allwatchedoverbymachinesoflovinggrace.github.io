#!/usr/bin/env python3
"""
RM Script 05 — Persistent cross-session memory
Reads dialog_log.jsonl, extracts concept co-occurrences,
updates mother_english_io_v5 substrate association weights via /api/learn.
Run nightly via cron: 0 2 * * * python3 /home/joe/sparky/rm_scripts/05_session_memory.py
"""
import json
import re
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DIALOG_LOG   = Path("/home/joe/sparky/mother_dialog_log.jsonl")
MEMORY_STATE = Path("/home/joe/sparky/rm_diagnostics/session_memory.json")
MOTHER_URL   = "http://localhost:8892"

def load_new_turns(last_processed: int) -> list:
    """Load dialog turns after offset last_processed."""
    if not DIALOG_LOG.exists() or DIALOG_LOG.stat().st_size == 0:
        return []
    turns = []
    with open(DIALOG_LOG) as f:
        for i, line in enumerate(f):
            if i <= last_processed:
                continue
            try:
                turns.append(json.loads(line))
            except Exception:
                pass
    return turns


def extract_concept_pairs(turns: list) -> Counter:
    """
    Extract word co-occurrence pairs from dialog turns.
    Returns Counter of (word_a, word_b) tuples.
    """
    pairs = Counter()
    stop  = {'the','a','an','is','are','was','were','i','you','it',
              'in','on','at','to','of','and','or','but','for','with',
              'that','this','my','me','we','be','do','have','has'}
    window = 4   # co-occurrence window size

    for turn in turns:
        text = turn.get("content","") or turn.get("text","") or str(turn)
        words = [w.lower() for w in re.findall(r'[a-z]{3,}', text)
                 if w.lower() not in stop]
        for i, w in enumerate(words):
            for j in range(i+1, min(i+window+1, len(words))):
                pair = tuple(sorted([w, words[j]]))
                pairs[pair] += 1

    return pairs


def push_to_mother(pairs: Counter) -> int:
    """Send top co-occurrence pairs to mother as association reinforcement."""
    top = pairs.most_common(50)
    if not top:
        return 0

    payload = json.dumps({
        "type":         "association_reinforcement",
        "pairs":        [[a, b, count] for (a,b), count in top],
        "source":       "session_memory",
        "timestamp":    datetime.now().isoformat(),
    }).encode()

    try:
        req = urllib.request.Request(
            f"{MOTHER_URL}/api/learn",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read())
        return result.get("pairs_absorbed", len(top))
    except Exception as e:
        print(f"  Mother push failed (endpoint may not exist yet): {e}")
        return 0


def main():
    state = {}
    if MEMORY_STATE.exists():
        state = json.loads(MEMORY_STATE.read_text())

    last_processed = state.get("last_line_processed", -1)
    turns = load_new_turns(last_processed)
    print(f"New dialog turns: {len(turns)}")

    if not turns:
        print("Nothing new to process.")
        return

    pairs = extract_concept_pairs(turns)
    print(f"Concept pairs extracted: {len(pairs)}")
    for (a,b), c in pairs.most_common(10):
        print(f"  {a} <-> {b}: {c}")

    absorbed = push_to_mother(pairs)
    print(f"Pushed to mother substrate: {absorbed} pairs")

    # Update state
    state["last_line_processed"] = last_processed + len(turns)
    state["last_run"]            = datetime.now().isoformat()
    state["total_turns_seen"]    = state.get("total_turns_seen", 0) + len(turns)
    state["total_pairs_sent"]    = state.get("total_pairs_sent", 0) + absorbed
    MEMORY_STATE.write_text(json.dumps(state, indent=2))
    print(f"State saved. Total turns seen: {state['total_turns_seen']}")


if __name__ == "__main__":
    main()
