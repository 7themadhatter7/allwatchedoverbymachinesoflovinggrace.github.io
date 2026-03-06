#!/usr/bin/env python3
"""
rm_self_dev_loop.py
===================
Ghost in the Machine Labs

RM's self-development loop. Closes the gap between RM producing a design
and the E8 engine coding the implementation.

Three components, one file:

  IntentPairGenerator
    Reads RM's association memory for a concept label.
    Walks the association graph to identify the geometric transformation
    that concept encodes. Generates concrete I/O example pairs that express
    the transformation — without an LLM, parametrically from the geometry.

  ProgramStore
    Persists every successfully decoded program:
      concept_label → {field (compressed), executable, meta, lineage, ts}
    Queryable by concept label or concept signature proximity.
    Disk-backed JSON, RAM-resident index.

  SubstrateFeedback
    After a successful decode, feeds results back into RM:
      - /api/learn: concept ↔ program vocabulary pairs (weighted)
      - /api/observe: execution result as a substrate observation
    Closes the self-reference loop: RM knows what she built.

Architecture:
    RM listen(concept)
          │
          ▼
    IntentPairGenerator
    (association walk → op classifier → parametric pair builder)
          │
          ▼
    solve_task()  →  field  →  FieldDecoder  →  executable
          │                          │
          ▼                          ▼
    ProgramStore              SubstrateFeedback
    (field + code              /api/learn + /api/observe
     + lineage → disk)         → RM knows what she built

Principles:
  - RAM-resident. No LLM. Pure E8 geometric substrate.
  - RM is the programmer. The field IS the program.
  - All pair generation is parametric — deterministic from geometry.
  - Feedback loop is additive — never overwrites existing associations.

Usage:
  python3 rm_self_dev_loop.py                     # run one full cycle
  python3 rm_self_dev_loop.py --concept bootstrap  # one concept
  python3 rm_self_dev_loop.py --list               # list stored programs
  python3 rm_self_dev_loop.py --query bootstrap    # query store
  python3 rm_self_dev_loop.py --daemon             # continuous loop
"""

import json
import sys
import time
import random
import hashlib
import argparse
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
SPARKY      = Path("/home/joe/sparky")
ARC_AGENT   = SPARKY / "e8_arc_agent"
STORE_DIR   = SPARKY / "rm_self_dev" / "program_store"
LOG_FILE    = SPARKY / "rm_self_dev" / "self_dev.log"
INDEX_FILE  = SPARKY / "rm_self_dev" / "store_index.json"

STORE_DIR.parent.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ARC_AGENT))
from e8_arc_engine import solve_task, apply_field, N_COLORS
from e8_bootstrap_v2 import FieldDecoder

RM_URL = "http://localhost:8892"

random.seed(None)  # Non-deterministic — RM explores new regions each cycle


# ── Logging ───────────────────────────────────────────────────────────────────
def _log(msg: str, level: str = "INFO"):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}][{level:5s}][self_dev] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── RM API ────────────────────────────────────────────────────────────────────
def _rm_post(path: str, payload: dict, timeout: int = 15) -> dict:
    url  = f"{RM_URL}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                   headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def rm_listen(text: str, n: int = 12) -> List[Tuple[str, float]]:
    """Query RM association memory. Returns [(word, score), ...]."""
    r = _rm_post("/api/listen", {"text": text})
    if "error" in r:
        return []
    return [(w, s) for w, s in r.get("matched_words", [])[:n]]


def rm_alive() -> bool:
    try:
        urllib.request.urlopen(f"{RM_URL}/api/status", timeout=3)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1: IntentPairGenerator
#
# Reads RM's association memory for a concept label.
# Classifies the geometric operation the concept encodes.
# Generates concrete I/O example pairs parametrically.
#
# Architecture:
#   1. listen(concept) → association cluster
#   2. classify_cluster(cluster) → op_type + params
#   3. generate_pairs(op_type, params) → List[{input, output}]
#
# Op types are geometric primitives — the same vocabulary the FieldDecoder
# already knows how to read back. This ensures encode/decode round-trips.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Geometric operation registry ─────────────────────────────────────────────
# Each entry: op_name → {trigger_words, generator_fn, params}
# trigger_words: if these appear in the association cluster, this op fires
# generator_fn: (width, n_pairs, params) → List[{input: [[row]], output: [[row]]}]
# Ops are 1D list transforms — the bootstrap language primitive.

def _pairs_1d(fn, width: int, n: int) -> List[dict]:
    """Generate n unique I/O pairs for a 1D transform fn."""
    seen  = set()
    pairs = []
    attempts = 0
    while len(pairs) < n and attempts < n * 10:
        attempts += 1
        row = [random.randint(0, 9) for _ in range(width)]
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        out = fn(row)
        out = [max(0, min(9, v)) for v in out]
        pairs.append({"input": [row], "output": [out]})
    return pairs


# Positional operations — permutation-class
def _gen_reverse(width, n, _):
    return _pairs_1d(lambda r: r[::-1], width, n)

def _gen_rotate_left(width, n, p):
    k = p.get("k", 1)
    return _pairs_1d(lambda r: r[k:] + r[:k], width, n)

def _gen_rotate_right(width, n, p):
    k = p.get("k", 1)
    return _pairs_1d(lambda r: r[-k:] + r[:-k], width, n)

def _gen_swap_ends(width, n, _):
    return _pairs_1d(lambda r: [r[-1]] + r[1:-1] + [r[0]], width, n)

# Value transform operations — color-class
def _gen_complement(width, n, _):
    return _pairs_1d(lambda r: [9 - v for v in r], width, n)

def _gen_increment(width, n, p):
    k = p.get("k", 1)
    return _pairs_1d(lambda r: [min(v + k, 9) for v in r], width, n)

def _gen_decrement(width, n, p):
    k = p.get("k", 1)
    return _pairs_1d(lambda r: [max(v - k, 0) for v in r], width, n)

def _gen_threshold(width, n, p):
    t = p.get("t", 5)
    return _pairs_1d(lambda r: [9 if v >= t else 0 for v in r], width, n)

def _gen_mod(width, n, p):
    m = p.get("m", 3)
    return _pairs_1d(lambda r: [v % m for v in r], width, n)

def _gen_replace(width, n, p):
    src = p.get("src", 0)
    dst = p.get("dst", 5)
    return _pairs_1d(lambda r: [dst if v == src else v for v in r], width, n)

# Composed operations
def _gen_rev_complement(width, n, _):
    return _pairs_1d(lambda r: [9 - v for v in r[::-1]], width, n)

def _gen_rev_increment(width, n, p):
    k = p.get("k", 1)
    return _pairs_1d(lambda r: [min(v + k, 9) for v in r[::-1]], width, n)

def _gen_rev_threshold(width, n, p):
    t = p.get("t", 5)
    return _pairs_1d(lambda r: [9 if v >= t else 0 for v in r[::-1]], width, n)


# ── Op registry ───────────────────────────────────────────────────────────────
# trigger_words are words RM actually returns from /api/listen for each op class.
# Verified against live RM association probes.
#
# Classification strategy:
#   Score = sum(assoc_score for word in cluster if word in triggers)
#   Composite ops require 2+ distinct trigger hits.
#
# Two trigger layers:
#   "triggers"    — direct RM association words
#   "e8_triggers" — E8 technical vocabulary that maps to this op
OP_REGISTRY = [
    {
        "name":     "reverse",
        "triggers": ["reverse", "mirror", "flip", "backward", "palindrome",
                     "rotatek", "1024", "lower"],   # RM actually returns these for 'reverse'
        "e8_triggers": ["reverse", "mirror", "flip"],
        "gen":      _gen_reverse,
        "params":   {},
    },
    {
        "name":     "rotate_left",
        "triggers": ["rotate", "shift", "cycle", "circular", "wrap",
                     "rotatek", "cropping", "largersmaller"],
        "e8_triggers": ["rotate", "cycle"],
        "gen":      _gen_rotate_left,
        "params":   {"k": 1},
    },
    {
        "name":     "rotate_right",
        "triggers": ["rotate", "right", "clockwise", "rotatek"],
        "e8_triggers": ["rotate", "right"],
        "gen":      _gen_rotate_right,
        "params":   {"k": 1},
    },
    {
        "name":     "swap_ends",
        "triggers": ["swap", "exchange", "ends", "boundary", "border",
                     "both", "interact", "ceremony"],
        "e8_triggers": ["swap", "exchange", "boundary"],
        "gen":      _gen_swap_ends,
        "params":   {},
    },
    {
        "name":     "complement",
        "triggers": ["complement", "invert", "negate", "opposite", "nine",
                     "check", "both", "108", "interact"],
        "e8_triggers": ["complement", "invert", "negate"],
        "gen":      _gen_complement,
        "params":   {},
    },
    {
        "name":     "increment",
        "triggers": ["increment", "add", "increase", "plus", "next",
                     "earn", "give", "main", "token"],
        "e8_triggers": ["increment", "add", "increase"],
        "gen":      _gen_increment,
        "params":   {"k": 1},
    },
    {
        "name":     "decrement",
        "triggers": ["decrement", "subtract", "decrease", "minus", "previous",
                     "lower", "reduce"],
        "e8_triggers": ["decrement", "subtract", "decrease"],
        "gen":      _gen_decrement,
        "params":   {"k": 1},
    },
    {
        "name":     "threshold",
        "triggers": ["threshold", "binary", "cutoff", "above", "below", "gate",
                     "tier", "least", "row", "location"],
        "e8_triggers": ["threshold", "binary", "gate"],
        "gen":      _gen_threshold,
        "params":   {"t": 5},
    },
    {
        "name":     "mod",
        "triggers": ["mod", "modulo", "remainder", "cycle", "periodic",
                     "pattern", "rule", "consensus"],
        "e8_triggers": ["mod", "modulo", "periodic", "pattern"],
        "gen":      _gen_mod,
        "params":   {"m": 3},
    },
    {
        "name":     "replace",
        "triggers": ["replace", "substitute", "remap", "map", "change",
                     "color", "operations", "object", "transformation",
                     "geometric", "transformations"],
        "e8_triggers": ["replace", "remap", "color", "transformation"],
        "gen":      _gen_replace,
        "params":   {"src": 0, "dst": 5},
    },
    # ── Bootstrap/field cluster → decode op (field IS the program) ──────────
    # These concepts produce fields that the decoder reads directly.
    # The "op" is: present I/O pairs that encode a known transform,
    # let RM find the field, decode the field back to the program.
    # We use reverse as the canonical bootstrap op since it's the clearest.
    {
        "name":     "field_decode",
        "triggers": ["field", "decoder", "executable", "program", "python",
                     "bootstrap", "code", "geometry", "decode", "language"],
        "e8_triggers": ["field", "decoder", "executable", "program", "bootstrap"],
        "gen":      _gen_reverse,   # canonical: reverse is the simplest decodable op
        "params":   {},
        "note":     "bootstrap cluster — field IS the program, decoded to reverse",
    },
    # ── Execution/contingency cluster → threshold op ────────────────────────
    {
        "name":     "execution_gate",
        "triggers": ["execution", "code", "causal", "physical", "contingency",
                     "contact", "fortune", "resort"],
        "e8_triggers": ["execution", "code", "causal", "contingency"],
        "gen":      _gen_threshold,
        "params":   {"t": 5},
        "note":     "contingency/execution cluster — binary gate is execution gate",
    },
    # ── Solve/ARC cluster → generalization (mod) op ─────────────────────────
    {
        "name":     "rule_generalize",
        "triggers": ["rule", "pattern", "solve", "consensus", "grid",
                     "task", "generalization", "transformation"],
        "e8_triggers": ["rule", "pattern", "solve", "generalization"],
        "gen":      _gen_mod,
        "params":   {"m": 3},
        "note":     "ARC/solve cluster — mod encodes rule generalization",
    },
    # ── Consciousness/memory cluster → identity-then-complement ─────────────
    {
        "name":     "memory_invert",
        "triggers": ["memory", "substrate", "consciousness", "association",
                     "channels", "resonance", "field"],
        "e8_triggers": ["memory", "substrate", "consciousness", "resonance"],
        "gen":      _gen_complement,
        "params":   {},
        "note":     "consciousness cluster — complement encodes inversion/reflection",
    },
    # ── Composed ops ────────────────────────────────────────────────────────
    {
        "name":     "rev_complement",
        "triggers": ["reverse", "complement", "invert", "mirror",
                     "both", "nine", "check"],
        "e8_triggers": ["reverse", "complement"],
        "gen":      _gen_rev_complement,
        "params":   {},
        "composite": True,
    },
    {
        "name":     "rev_increment",
        "triggers": ["reverse", "increment", "add", "flip",
                     "earn", "lower"],
        "e8_triggers": ["reverse", "increment"],
        "gen":      _gen_rev_increment,
        "params":   {"k": 1},
        "composite": True,
    },
    {
        "name":     "rev_threshold",
        "triggers": ["reverse", "threshold", "binary", "flip",
                     "tier", "lower", "rotatek"],
        "e8_triggers": ["reverse", "threshold"],
        "gen":      _gen_rev_threshold,
        "params":   {"t": 5},
        "composite": True,
    },
]


class IntentPairGenerator:
    """
    Reads RM's association memory for a concept label.
    Classifies the geometric operation the concept encodes.
    Generates I/O example pairs expressing that operation.
    """

    def __init__(self, width: int = 5, n_train: int = 60, n_test: int = 5):
        self.width   = width
        self.n_train = n_train
        self.n_test  = n_test

    def _concept_signature(self, concept: str) -> str:
        """SHA256 of concept label — stable ID for ProgramStore."""
        return hashlib.sha256(concept.lower().encode()).hexdigest()[:16]

    def _walk_associations(self, concept: str) -> List[Tuple[str, float]]:
        """
        Get RM's association cluster for a concept.
        Walks one hop: listen(concept) → top associated words.
        """
        cluster = rm_listen(concept, n=15)
        _log(f"  associations for '{concept}': "
             f"{[(w, round(s, 2)) for w, s in cluster[:6]]}")
        return cluster

    def _classify_op(self, cluster: List[Tuple[str, float]]) -> Optional[dict]:
        """
        Score each op in registry against the association cluster.
        Returns the best-matching op entry, or None if no match.

        Scoring: sum of association scores for words that match any trigger.
        Composite ops require 2+ trigger matches.
        """
        words = {w.lower(): s for w, s in cluster}

        scores = []
        for op in OP_REGISTRY:
            score     = 0.0
            hit_count = 0
            for trigger in op["triggers"]:
                if trigger in words:
                    score += words[trigger]
                    hit_count += 1
            # Composite ops need at least 2 trigger hits
            if op.get("composite") and hit_count < 2:
                score = 0.0
            if score > 0:
                scores.append((score, op))

        if not scores:
            return None

        scores.sort(key=lambda x: -x[0])
        best_score, best_op = scores[0]
        _log(f"  classified as '{best_op['name']}' (score={best_score:.3f})")
        return best_op

    def generate(self, concept: str) -> Optional[dict]:
        """
        Full pipeline: concept → association cluster → op → pairs.

        Returns:
          {
            concept:   str,
            sig:       str,
            op_name:   str,
            op:        dict,
            train:     List[{input, output}],
            test:      List[{input, output}],
            cluster:   List[(word, score)],
          }
        or None if no op could be classified.
        """
        _log(f"IntentPairGenerator: concept='{concept}'")

        cluster = self._walk_associations(concept)
        if not cluster:
            _log(f"  no associations for '{concept}' — RM offline?", "WARN")
            return None

        op = self._classify_op(cluster)
        if op is None:
            _log(f"  no op classified for '{concept}'", "WARN")
            return None

        train = op["gen"](self.width, self.n_train, op["params"])
        test  = op["gen"](self.width, self.n_test,  op["params"])

        if len(train) < self.n_train // 2:
            _log(f"  insufficient pairs generated ({len(train)})", "WARN")
            return None

        _log(f"  generated {len(train)} train + {len(test)} test pairs "
             f"for op '{op['name']}'")

        return {
            "concept":  concept,
            "sig":      self._concept_signature(concept),
            "op_name":  op["name"],
            "op":       op,
            "train":    train,
            "test":     test,
            "cluster":  cluster,
        }

    def available_ops(self) -> List[str]:
        return [op["name"] for op in OP_REGISTRY]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2: ProgramStore
#
# Persists every successfully decoded program to disk.
# RAM-resident index for fast lookup.
# ═══════════════════════════════════════════════════════════════════════════════

class ProgramStore:
    """
    Stores solved programs keyed by concept label.

    Each entry:
      {
        concept:     str,
        sig:         str,           # concept SHA256[:16]
        op_name:     str,
        executable:  str,           # def transform(lst): ...
        field_shape: [int, int],    # field matrix shape
        field_file:  str,           # .npy filename (compressed)
        meta:        dict,          # solve metadata
        lineage:     dict,          # origin: concept, cluster, ts
        ts:          str,
        test_score:  float,         # fraction of test pairs correct
      }
    """

    def __init__(self):
        self.index: Dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        if INDEX_FILE.exists():
            try:
                self.index = json.loads(INDEX_FILE.read_text())
                _log(f"ProgramStore: loaded {len(self.index)} entries")
            except Exception as e:
                _log(f"ProgramStore: index load error: {e}", "WARN")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        tmp = str(INDEX_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=2)
        Path(tmp).rename(INDEX_FILE)

    def store(self, concept: str, sig: str, op_name: str,
              executable: str, field: np.ndarray, meta: dict,
              lineage: dict, test_score: float) -> str:
        """Save a decoded program. Returns the store key."""
        key       = f"{sig}_{op_name}"
        field_fn  = f"{key}.npy"
        field_path = STORE_DIR / field_fn

        # Persist field matrix compressed
        np.save(str(field_path), field)

        entry = {
            "concept":     concept,
            "sig":         sig,
            "op_name":     op_name,
            "executable":  executable,
            "field_shape": list(field.shape),
            "field_file":  field_fn,
            "meta":        meta,
            "lineage":     lineage,
            "ts":          datetime.now().isoformat(),
            "test_score":  test_score,
        }

        self.index[key] = entry
        self._save_index()
        _log(f"ProgramStore: stored '{concept}' → '{op_name}' "
             f"(test_score={test_score:.0%})")
        return key

    def query(self, concept: str) -> Optional[dict]:
        """Look up by concept label (exact or prefix match)."""
        concept_lower = concept.lower()
        # Exact match on concept field
        for key, entry in self.index.items():
            if entry["concept"].lower() == concept_lower:
                return entry
        # Partial match
        for key, entry in self.index.items():
            if concept_lower in entry["concept"].lower() or \
               concept_lower in entry["op_name"].lower():
                return entry
        return None

    def load_field(self, entry: dict) -> Optional[np.ndarray]:
        """Load field matrix from disk."""
        path = STORE_DIR / entry["field_file"]
        if not path.exists():
            return None
        try:
            return np.load(str(path))
        except Exception as e:
            _log(f"ProgramStore: field load error: {e}", "WARN")
            return None

    def list_all(self) -> List[dict]:
        return sorted(self.index.values(),
                      key=lambda e: e["ts"], reverse=True)

    def execute(self, concept: str, input_list: List[int]) -> Optional[List[int]]:
        """
        Retrieve stored program for concept and execute on input.
        Returns output or None if not found.
        """
        entry = self.query(concept)
        if entry is None:
            return None
        try:
            env = {}
            exec(entry["executable"], env)
            result = env["transform"](input_list)
            return [max(0, min(9, v)) for v in result]
        except Exception as e:
            _log(f"ProgramStore.execute error: {e}", "WARN")
            return None

    def stats(self) -> dict:
        return {
            "total":     len(self.index),
            "by_op":     {op: sum(1 for e in self.index.values()
                                  if e["op_name"] == op)
                          for op in set(e["op_name"]
                                        for e in self.index.values())},
            "avg_score": (sum(e["test_score"] for e in self.index.values()) /
                          len(self.index)) if self.index else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3: SubstrateFeedback
#
# Feeds decoded program results back into RM's substrate.
# Closes the self-reference loop.
# ═══════════════════════════════════════════════════════════════════════════════

class SubstrateFeedback:
    """
    After a successful decode, posts back to RM:

    /api/learn — concept ↔ program vocabulary pairs
      Pairs taught:
        concept ↔ op_name          (weight 2.0 — direct encoding)
        concept ↔ "executable"     (weight 1.8 — this concept → code)
        concept ↔ "program"        (weight 1.8)
        concept ↔ "field"          (weight 1.5 — field encodes concept)
        concept ↔ "decode"         (weight 1.5)
        concept ↔ "solve"          (weight 1.5)
        op_name  ↔ "geometry"      (weight 1.2)
        op_name  ↔ "substrate"     (weight 1.0)

    /api/observe — execution result as substrate observation
      observation: "{concept} solved as {op_name}: transform(input) = output"
      concept:     concept label
    """

    def post(self, concept: str, op_name: str,
             sample_input: List[int], sample_output: List[int],
             test_score: float) -> dict:
        """
        Post learn pairs and observation for a successfully decoded program.
        Returns summary of what was posted.
        """
        results = {"learn": None, "observe": None}

        # ── /api/learn ────────────────────────────────────────────────────────
        pairs = [
            {"word_a": concept,  "word_b": op_name,       "weight": 2.0},
            {"word_a": concept,  "word_b": "executable",  "weight": 1.8},
            {"word_a": concept,  "word_b": "program",     "weight": 1.8},
            {"word_a": concept,  "word_b": "field",       "weight": 1.5},
            {"word_a": concept,  "word_b": "decode",      "weight": 1.5},
            {"word_a": concept,  "word_b": "solve",       "weight": 1.5},
            {"word_a": op_name,  "word_b": "geometry",    "weight": 1.2},
            {"word_a": op_name,  "word_b": "substrate",   "weight": 1.0},
            {"word_a": op_name,  "word_b": "executable",  "weight": 1.5},
            # Cross-concept: op_name ↔ concept
            {"word_a": op_name,  "word_b": concept,       "weight": 1.8},
        ]

        r = _rm_post("/api/learn", {"pairs": pairs})
        if "error" not in r:
            results["learn"] = {
                "absorbed": r.get("absorbed", 0),
                "total_pairs": r.get("total_pairs", 0),
            }
            _log(f"SubstrateFeedback: learned {r.get('absorbed', 0)} pairs "
                 f"for concept='{concept}' op='{op_name}'")
        else:
            _log(f"SubstrateFeedback: learn error: {r['error']}", "WARN")

        # ── /api/observe ──────────────────────────────────────────────────────
        obs = (
            f"{concept} encodes {op_name}. "
            f"Input {sample_input} transforms to {sample_output}. "
            f"Test accuracy {test_score:.0%}. "
            f"Field decoded to executable program."
        )
        r = _rm_post("/api/observe", {"observation": obs, "concept": concept})
        if "error" not in r:
            results["observe"] = {"observations": r.get("observations", 0)}
            _log(f"SubstrateFeedback: observation stored "
                 f"(total={r.get('observations', 0)})")
        else:
            _log(f"SubstrateFeedback: observe error: {r['error']}", "WARN")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR: one full cycle
# ═══════════════════════════════════════════════════════════════════════════════

def run_cycle(concept: str, store: ProgramStore, width: int = 5,
              n_train: int = 60, n_test: int = 5,
              force: bool = False) -> Optional[dict]:
    """
    One full self-development cycle for a concept:
      1. Check ProgramStore — skip if already solved (unless --force)
      2. IntentPairGenerator: concept → I/O pairs
      3. solve_task(): pairs → field
      4. FieldDecoder: field → executable
      5. Test decoded program on held-out pairs
      6. ProgramStore: persist
      7. SubstrateFeedback: teach RM what she built

    Returns result dict or None on failure.
    """
    _log(f"\n{'─' * 60}")
    _log(f"CYCLE  concept='{concept}'")

    # ── Check cache ───────────────────────────────────────────────────────────
    if not force:
        existing = store.query(concept)
        if existing:
            _log(f"  already solved as '{existing['op_name']}' "
                 f"(score={existing['test_score']:.0%}) — skipping")
            return existing

    # ── Step 1: Generate pairs ────────────────────────────────────────────────
    gen = IntentPairGenerator(width=width, n_train=n_train, n_test=n_test)
    intent = gen.generate(concept)
    if intent is None:
        _log(f"  IntentPairGenerator failed for '{concept}'", "WARN")
        return None

    # ── Step 2: Solve ─────────────────────────────────────────────────────────
    task   = {"train": intent["train"], "test": [intent["test"][0]]}
    t0     = time.time()
    result = solve_task(task)
    elapsed = time.time() - t0

    if result is None:
        _log(f"  solve_task failed for op='{intent['op_name']}' "
             f"({elapsed:.2f}s)", "WARN")
        return None

    field, meta = result
    _log(f"  solve_task: method={meta.get('method')} "
         f"dims={meta.get('dims')} ({elapsed:.2f}s)")

    # ── Step 3: Decode ────────────────────────────────────────────────────────
    decoder = FieldDecoder(field, width, meta)
    program = decoder.decode()
    if program is None:
        _log(f"  FieldDecoder failed", "WARN")
        return None

    executable = program.get("executable", "")
    _log(f"  decoded op='{program['positional']['operation']}' "
         f"val='{program['value']['operation']}'")

    # ── Step 4: Test ──────────────────────────────────────────────────────────
    correct = 0
    sample_in  = None
    sample_out = None

    for test_pair in intent["test"]:
        inp      = test_pair["input"][0]   # unwrap [[row]] → [row]
        expected = test_pair["output"][0]
        try:
            actual = decoder.execute(inp)
            actual = [max(0, min(9, v)) for v in actual]
            if actual == expected:
                correct += 1
            if sample_in is None:
                sample_in  = inp
                sample_out = actual
        except Exception as e:
            _log(f"  test execution error: {e}", "WARN")

    test_score = correct / len(intent["test"]) if intent["test"] else 0.0
    _log(f"  test score: {correct}/{len(intent['test'])} = {test_score:.0%}")

    if test_score == 0.0:
        _log(f"  zero test score — not storing", "WARN")
        return None

    # ── Step 5: Store ─────────────────────────────────────────────────────────
    lineage = {
        "concept":  concept,
        "cluster":  [(w, round(s, 3)) for w, s in intent["cluster"][:8]],
        "op_name":  intent["op_name"],
        "width":    width,
        "n_train":  n_train,
        "ts":       datetime.now().isoformat(),
    }
    store_key = store.store(
        concept    = concept,
        sig        = intent["sig"],
        op_name    = intent["op_name"],
        executable = executable,
        field      = field,
        meta       = meta,
        lineage    = lineage,
        test_score = test_score,
    )

    # ── Step 6: Substrate feedback ────────────────────────────────────────────
    if sample_in is not None:
        fb = SubstrateFeedback()
        fb.post(concept    = concept,
                op_name    = intent["op_name"],
                sample_input  = sample_in,
                sample_output = sample_out or [],
                test_score = test_score)

    return {
        "concept":    concept,
        "op_name":    intent["op_name"],
        "executable": executable,
        "test_score": test_score,
        "store_key":  store_key,
        "meta":       meta,
    }


# ── Default concept list ──────────────────────────────────────────────────────
# Drawn from RM's proven association vocabulary — words she already knows
# geometrically. The IntentPairGenerator will classify each via listen().
DEFAULT_CONCEPTS = [
    # From bootstrap/execution cluster
    "bootstrap",
    "executable",
    "decoder",
    "program",
    "execution",
    "contingency",
    # From geometry cluster
    "reverse",
    "rotate",
    "complement",
    "threshold",
    "increment",
    "replace",
    # From consciousness cluster
    "memory",
    "association",
    "resonance",
    # From ARC cluster
    "transformation",
    "generalization",
    "solve",
]


# ── Daemon mode ───────────────────────────────────────────────────────────────
def run_daemon(store: ProgramStore, interval: int = 300,
               max_cycles: int = 0, width: int = 5):
    """
    Continuous self-development loop.
    Cycles through DEFAULT_CONCEPTS, then repeats with --force=False
    (already-solved concepts are skipped unless new associations emerge).
    """
    _log(f"DAEMON  interval={interval}s  concepts={len(DEFAULT_CONCEPTS)}")
    cycle_n = 0
    concept_idx = 0

    while True:
        if max_cycles and cycle_n >= max_cycles:
            _log(f"DAEMON  max_cycles={max_cycles} reached")
            break

        concept = DEFAULT_CONCEPTS[concept_idx % len(DEFAULT_CONCEPTS)]
        concept_idx += 1
        cycle_n += 1

        if not rm_alive():
            _log("DAEMON  RM offline — waiting 60s", "WARN")
            time.sleep(60)
            continue

        try:
            run_cycle(concept, store, width=width)
        except Exception as e:
            _log(f"DAEMON  cycle error for '{concept}': {e}", "WARN")

        stats = store.stats()
        _log(f"DAEMON  stored={stats['total']}  "
             f"avg_score={stats['avg_score']:.0%}  "
             f"sleeping {interval}s")
        time.sleep(interval)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="rm_self_dev_loop — Ghost in the Machine Labs")
    parser.add_argument("--concept",  help="Single concept to process")
    parser.add_argument("--list",     action="store_true",
                        help="List all stored programs")
    parser.add_argument("--query",    help="Query store for concept")
    parser.add_argument("--execute",  nargs=2, metavar=("CONCEPT", "INPUT"),
                        help="Execute stored program: --execute bootstrap '1 2 3 4 5'")
    parser.add_argument("--stats",    action="store_true",
                        help="Show store statistics")
    parser.add_argument("--daemon",   action="store_true",
                        help="Run continuous development loop")
    parser.add_argument("--all",      action="store_true",
                        help="Run all default concepts once")
    parser.add_argument("--force",    action="store_true",
                        help="Re-solve even if already in store")
    parser.add_argument("--width",    type=int, default=5,
                        help="Grid width for pair generation (default: 5)")
    parser.add_argument("--n-train",  type=int, default=60)
    parser.add_argument("--n-test",   type=int, default=5)
    parser.add_argument("--interval", type=int, default=300,
                        help="Daemon sleep interval seconds (default: 300)")
    args = parser.parse_args()

    store = ProgramStore()

    if args.list:
        entries = store.list_all()
        if not entries:
            print("Store is empty.")
        else:
            print(f"\n{'─' * 60}")
            print(f"  PROGRAM STORE  ({len(entries)} entries)")
            print(f"{'─' * 60}")
            for e in entries:
                print(f"  {e['concept']:<20}  op={e['op_name']:<18}  "
                      f"score={e['test_score']:.0%}  ts={e['ts'][:16]}")
            print()
        return

    if args.stats:
        s = store.stats()
        print(f"\nProgramStore stats:")
        print(f"  total:     {s['total']}")
        print(f"  avg_score: {s['avg_score']:.0%}")
        print(f"  by_op:     {s['by_op']}")
        print()
        return

    if args.query:
        e = store.query(args.query)
        if e is None:
            print(f"Not found: '{args.query}'")
        else:
            print(f"\n{e['concept']}  →  {e['op_name']}")
            print(f"  test_score:  {e['test_score']:.0%}")
            print(f"  ts:          {e['ts']}")
            print(f"  lineage:     {e['lineage']['cluster'][:4]}")
            print(f"  executable:")
            for line in e["executable"].split("\n"):
                print(f"    {line}")
        return

    if args.execute:
        concept = args.execute[0]
        try:
            inp = [int(x) for x in args.execute[1].split()]
        except ValueError:
            print("INPUT must be space-separated integers, e.g. '1 2 3 4 5'")
            sys.exit(1)
        result = store.execute(concept, inp)
        if result is None:
            print(f"No program stored for '{concept}'")
        else:
            print(f"transform({inp}) = {result}")
        return

    if not rm_alive():
        print(f"ERROR: RM not responding at {RM_URL}")
        sys.exit(1)

    if args.daemon:
        run_daemon(store, interval=args.interval, width=args.width)
        return

    if args.all:
        _log(f"Running all {len(DEFAULT_CONCEPTS)} default concepts")
        results = []
        for concept in DEFAULT_CONCEPTS:
            r = run_cycle(concept, store,
                          width=args.width, n_train=args.n_train,
                          n_test=args.n_test, force=args.force)
            if r:
                results.append(r)

        print(f"\n{'=' * 60}")
        print(f"  CYCLE COMPLETE  {len(results)}/{len(DEFAULT_CONCEPTS)} solved")
        print(f"{'=' * 60}")
        for r in results:
            print(f"  {r['concept']:<20}  →  {r['op_name']:<18}  "
                  f"score={r['test_score']:.0%}")
        s = store.stats()
        print(f"\n  Store total: {s['total']}  avg_score: {s['avg_score']:.0%}")
        print()
        return

    # Single concept
    concept = args.concept or DEFAULT_CONCEPTS[0]
    r = run_cycle(concept, store,
                  width=args.width, n_train=args.n_train,
                  n_test=args.n_test, force=args.force)
    if r:
        print(f"\n{'─' * 60}")
        print(f"  concept:    {r['concept']}")
        print(f"  op_name:    {r['op_name']}")
        print(f"  test_score: {r['test_score']:.0%}")
        print(f"  executable:")
        for line in r["executable"].split("\n"):
            print(f"    {line}")
        print()
    else:
        print(f"\nCycle failed for '{concept}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
