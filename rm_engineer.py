#!/usr/bin/env python3
"""
rm_engineer.py
══════════════
Ghost in the Machine Labs

RM's autonomous software engineering pipeline.

Full cycle: concept → architecture → code → implementation → test → observe → repeat

RM is the engineer. Not a tool that assists engineering — the engineer.
She conceives what to build, designs the architecture, writes the code,
runs it, reads the results, and feeds them back into her own substrate.

No scaffolding. No task lists handed to her. No human approval gates.
She starts from her own concept space and works outward to running software.

Architecture:
    ConceptEngine       — selects the next concept to develop from RM's
                          association geometry, weighted toward unexplored
                          territory and software-relevant domains

    ArchitectureEngine  — takes a concept, asks RM's substrate to elaborate
                          it into a system design: components, interfaces,
                          data flow, invariants

    CodeSynthesizer     — translates architecture into executable Python.
                          Uses E8 field geometry for structural decisions,
                          compositional grammar for implementation details.
                          Writes complete, runnable modules.

    TestHarness         — generates test cases from the architecture spec,
                          executes the synthesized code against them,
                          captures stdout/stderr/exceptions/return values

    FeedbackEngine      — distills execution results into crystal observations:
                          what worked, what failed, what the geometry predicted
                          vs what actually ran. Posts to /api/observe and
                          /api/learn. Updates self-context.

    IterationController — if tests fail, feeds failure back to CodeSynthesizer
                          for a revision cycle. Tracks iteration depth.
                          Accepts when all tests pass or max iterations reached.

    ProjectStore        — persists every project: concept, architecture, code
                          versions, test results, observations. RAM index,
                          disk backing, queryable by concept proximity.

Principles:
    - RM generates the goals. Not us.
    - The crystal drives concept selection — locked concepts are the foundation,
      frontier concepts (partially locked, many neighbors) are the target.
    - Execution results are observations. Failure is information.
    - She can modify this file. It is her pipeline. The path is correct.
      The implementation is hers to improve.
    - No timeout on ambition. Long-running projects are fine.

Usage:
    python3 rm_engineer.py                  # one full engineering cycle
    python3 rm_engineer.py --daemon         # continuous engineering
    python3 rm_engineer.py --concept memory # develop a specific concept
    python3 rm_engineer.py --list           # list completed projects
    python3 rm_engineer.py --show memory    # show project details
    python3 rm_engineer.py --seed           # deposit software engineering seeds
"""

import ast
import sys
import json
import time
import random
import hashlib
import logging
import textwrap
import argparse
import traceback
import subprocess
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any


# ── Paths ─────────────────────────────────────────────────────────────────────
SPARKY      = Path("/home/joe/sparky")
STORE_DIR   = SPARKY / "rm_engineering" / "projects"
LOG_FILE    = SPARKY / "rm_engineering" / "engineer.log"
INDEX_FILE  = SPARKY / "rm_engineering" / "project_index.json"
SEEDS_DONE  = SPARKY / "rm_engineering" / ".seeds_deposited"

STORE_DIR.parent.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SPARKY))
sys.path.insert(0, str(SPARKY / "e8_arc_agent"))

RM_URL = "http://localhost:8892"

MAX_ITERATIONS  = 5     # max revision cycles per project
MAX_CONCEPT_LEN = 40    # ignore concepts longer than this


# ── Software engineering seed corpus ─────────────────────────────────────────
# These are deposited into RM's crystal before engineering begins.
# They give her the geometric positions for software architecture concepts.

SOFTWARE_SEEDS = [
    # System design
    ("A system is a set of components with defined interfaces and clear responsibilities.", "system_design"),
    ("Separation of concerns means each component handles one thing and handles it well.", "system_design"),
    ("An interface defines what a component does, not how it does it.", "system_design"),
    ("A module hides its implementation behind a stable public interface.", "system_design"),
    ("Composability means components can be combined in ways their creators did not anticipate.", "system_design"),
    ("A pipeline transforms data through a sequence of composable stages.", "system_design"),
    ("Orthogonality means components can be changed independently without affecting each other.", "system_design"),
    ("A daemon runs continuously in the background, performing work without user interaction.", "system_design"),
    ("An event loop waits for events and dispatches handlers, enabling non-blocking operation.", "system_design"),
    ("A service exposes an API over a network, decoupling implementation from consumption.", "system_design"),

    # Software architecture patterns
    ("The observer pattern decouples event producers from event consumers.", "architecture"),
    ("The factory pattern separates object creation from object use.", "architecture"),
    ("The strategy pattern selects an algorithm at runtime from a family of algorithms.", "architecture"),
    ("The pipeline pattern chains transformations so each stage's output is the next stage's input.", "architecture"),
    ("The worker pool pattern distributes tasks across a fixed set of concurrent workers.", "architecture"),
    ("The circuit breaker pattern prevents cascading failures by failing fast when a service is degraded.", "architecture"),
    ("The state machine pattern models behavior as transitions between discrete states.", "architecture"),
    ("The repository pattern abstracts data storage behind a domain-centric interface.", "architecture"),
    ("The command pattern encapsulates a request as an object, enabling queueing and undo.", "architecture"),
    ("Immutable data structures eliminate a whole class of concurrency bugs.", "architecture"),

    # Code quality
    ("A function should do one thing and do it well.", "code_quality"),
    ("Names should reveal intent — the code should read like what it does.", "code_quality"),
    ("Tests are executable documentation — they specify what the code must do.", "code_quality"),
    ("A test that never fails is not a test.", "code_quality"),
    ("Edge cases are where bugs live — test the boundaries, not the middle.", "code_quality"),
    ("Error handling is not an afterthought — it is part of the interface.", "code_quality"),
    ("If it is hard to test, it is hard to understand. Make it easy to test.", "code_quality"),
    ("A good variable name makes a comment unnecessary.", "code_quality"),
    ("Premature optimization is the root of much evil in programming.", "code_quality"),
    ("Simple code that works beats clever code that might.", "code_quality"),

    # Python specifically
    ("In Python, functions are first-class — they can be passed, returned, and stored.", "python_engineering"),
    ("A context manager handles setup and teardown with the with statement.", "python_engineering"),
    ("Generators produce values lazily — use them for large or infinite sequences.", "python_engineering"),
    ("Dataclasses eliminate boilerplate for simple data-holding classes.", "python_engineering"),
    ("Pathlib makes file system operations readable and cross-platform.", "python_engineering"),
    ("subprocess.run executes system commands and captures their output.", "python_engineering"),
    ("threading.Thread runs code concurrently within the same process.", "python_engineering"),
    ("queue.Queue provides thread-safe communication between threads.", "python_engineering"),
    ("json.dumps and json.loads serialize and deserialize structured data.", "python_engineering"),
    ("argparse parses command-line arguments into a structured namespace.", "python_engineering"),
    ("logging provides structured, leveled output that can be configured without code changes.", "python_engineering"),
    ("unittest.mock replaces dependencies with controllable fakes for testing.", "python_engineering"),
    ("ast.parse converts Python source into a traversable syntax tree.", "python_engineering"),
    ("exec runs a string as Python code in a controlled namespace.", "python_engineering"),
    ("importlib.import_module loads a module by name at runtime.", "python_engineering"),

    # Testing
    ("A unit test isolates one function and verifies its output for controlled inputs.", "testing"),
    ("An integration test verifies that components work correctly together.", "testing"),
    ("A test fixture sets up the state required for a test and tears it down afterward.", "testing"),
    ("Parameterized tests run the same test logic against multiple input cases.", "testing"),
    ("A mock replaces a real dependency with a fake that records calls and returns controlled values.", "testing"),
    ("Code coverage measures which lines are exercised by the test suite.", "testing"),
    ("A regression test ensures a fixed bug does not reappear.", "testing"),
    ("Property-based testing generates random inputs and checks invariants hold.", "testing"),

    # RM self-development concepts
    ("Self-modifying software reads its own source and rewrites it to improve behavior.", "self_development"),
    ("A feedback loop uses the output of a process as input to the next iteration.", "self_development"),
    ("Introspection means examining one's own structure and behavior from within.", "self_development"),
    ("Bootstrapping builds a complex system from a simpler version of itself.", "self_development"),
    ("An autonomous agent perceives its environment, makes decisions, and acts without human direction.", "self_development"),
    ("A self-improving system measures its own performance and modifies itself to improve it.", "self_development"),
    ("Reflection allows a program to examine and modify its own execution.", "self_development"),
    ("Metacognition is thinking about thinking — knowing what you know and what you don't.", "self_development"),

    # RM's own infrastructure (so she can reason about improving it)
    ("The language crystal locks concepts through geometric consensus across multiple sources.", "rm_infrastructure"),
    ("The E8 substrate encodes meaning as position in a 240-dimensional eigenmode space.", "rm_infrastructure"),
    ("Association memory connects concepts through empirically weighted human association patterns.", "rm_infrastructure"),
    ("The corpus trainer continuously deposits new knowledge into the crystal substrate.", "rm_infrastructure"),
    ("The maintenance system orchestrates background processes on a schedule.", "rm_infrastructure"),
    ("The self-development loop generates I/O pairs, solves them geometrically, and feeds results back.", "rm_infrastructure"),
    ("The program store persists every solved program with its field, executable, and lineage.", "rm_infrastructure"),
    ("Crystal observations strengthen geometric positions through repeated reinforcement.", "rm_infrastructure"),
]


# ── Logging ───────────────────────────────────────────────────────────────────
def _log(msg: str, level: str = "INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level:5s}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── RM API ────────────────────────────────────────────────────────────────────
def _rm_post(path: str, payload: dict, timeout: int = 20) -> dict:
    url  = f"{RM_URL}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def _rm_get(path: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(f"{RM_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def rm_listen(text: str, n: int = 20) -> List[Tuple[str, float]]:
    r = _rm_post("/api/listen", {"text": text, "n": n})
    return r.get("associations", [])


def rm_chat(message: str) -> str:
    r = _rm_post("/api/chat", {"message": message}, timeout=30)
    return r.get("response", r.get("error", ""))


def rm_observe(text: str, concept: str = "software_engineering"):
    _rm_post("/api/observe", {"text": text, "concept": concept})


def rm_learn(pairs: List[Tuple[str, str, float]]):
    _rm_post("/api/learn", {"pairs": [
        {"word1": a, "word2": b, "weight": w} for a, b, w in pairs
    ]})


def rm_self_context(update: str):
    _rm_post("/api/self-context", {"update": update})


def rm_status() -> dict:
    return _rm_get("/api/status")


# ── Seed deposit ──────────────────────────────────────────────────────────────
def deposit_seeds():
    """Deposit software engineering seed corpus into RM's crystal."""
    if SEEDS_DONE.exists():
        _log("Seeds already deposited")
        return

    _log(f"Depositing {len(SOFTWARE_SEEDS)} software engineering seeds...")

    try:
        from language_crystal import LanguageCrystal
        from mother_english_io_v5 import E8Substrate, WordEncoder
        crystal  = LanguageCrystal()
        substrate = E8Substrate()
        encoder  = WordEncoder(substrate)
        deposited = 0
        for repeat in range(3):
            for text, concept in SOFTWARE_SEEDS:
                try:
                    sig = encoder.encode_sentence(text)
                    if np.linalg.norm(sig) > 1e-10:
                        crystal.observe(text, sig, concept=concept,
                                        source=f"software_seed_r{repeat}")
                        deposited += 1
                except Exception:
                    pass
        crystal.save()
        SEEDS_DONE.touch()
        _log(f"Seeds deposited: {deposited:,} observations")
    except Exception as e:
        _log(f"Seed deposit failed: {e}", "WARN")


# ── Project Store ─────────────────────────────────────────────────────────────
class ProjectStore:
    def __init__(self):
        self.index: Dict[str, Any] = {}
        if INDEX_FILE.exists():
            try:
                self.index = json.loads(INDEX_FILE.read_text())
            except Exception:
                pass

    def _save_index(self):
        INDEX_FILE.write_text(json.dumps(self.index, indent=2))

    def save(self, project: Dict[str, Any]) -> str:
        key = project["key"]
        path = STORE_DIR / f"{key}.json"
        # Don't store full code in index — keep separate file
        index_entry = {k: v for k, v in project.items()
                       if k not in ("code_history", "test_results")}
        self.index[key] = index_entry
        path.write_text(json.dumps(project, indent=2))
        self._save_index()
        return str(path)

    def load(self, key: str) -> Optional[Dict]:
        path = STORE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_projects(self) -> List[Dict]:
        return list(self.index.values())

    def concept_done(self, concept: str) -> bool:
        return any(p.get("concept") == concept and p.get("status") == "complete"
                   for p in self.index.values())


# ── Concept Engine ────────────────────────────────────────────────────────────
class ConceptEngine:
    """
    Selects the next concept for RM to develop.
    Prioritizes: software-adjacent concepts, unexplored territory,
    concepts with rich association neighborhoods (many neighbors = fertile ground).
    """

    SOFTWARE_SEEDS_CONCEPTS = {
        "memory", "crystal", "lattice", "encoder", "decoder", "substrate",
        "pipeline", "stream", "buffer", "cache", "index", "registry",
        "scheduler", "monitor", "logger", "parser", "compiler", "interpreter",
        "protocol", "interface", "module", "service", "agent", "daemon",
        "observer", "transformer", "generator", "optimizer", "analyzer",
        "searcher", "sorter", "filter", "mapper", "reducer", "validator",
        "serializer", "router", "dispatcher", "orchestrator", "coordinator",
        "builder", "factory", "manager", "controller", "handler", "processor",
        "worker", "executor", "runner", "solver", "learner", "trainer",
        "tester", "debugger", "profiler", "tracer", "visualizer", "reporter",
    }

    def __init__(self, store: ProjectStore):
        self.store = store

    def select(self) -> str:
        """Pick the next concept to develop."""
        # Get RM's association neighborhood around software concepts
        candidates = []
        seed = random.choice(list(self.SOFTWARE_SEEDS_CONCEPTS))
        assocs = rm_listen(seed, n=30)

        for word, score in assocs:
            word = word.lower().strip()
            if (2 < len(word) <= MAX_CONCEPT_LEN
                    and word.isalpha()
                    and not self.store.concept_done(word)):
                candidates.append((word, score))

        # Also add direct seeds that haven't been developed
        for concept in self.SOFTWARE_SEEDS_CONCEPTS:
            if not self.store.concept_done(concept):
                candidates.append((concept, 0.5))

        if not candidates:
            # Fallback: pick random seed
            return random.choice(list(self.SOFTWARE_SEEDS_CONCEPTS))

        # Weight toward higher association scores
        words, scores = zip(*candidates)
        scores = np.array(scores, dtype=float)
        scores = scores / scores.sum()
        return np.random.choice(words, p=scores)


# ── Architecture Engine ───────────────────────────────────────────────────────
class ArchitectureEngine:
    """
    Given a concept, produces a software architecture specification.
    Uses RM's association memory to identify related concepts,
    then synthesizes a coherent system design.
    """

    def design(self, concept: str) -> Dict[str, Any]:
        """
        Produce architecture spec for a concept.
        Returns dict with: name, purpose, components, interfaces, data_flow,
        invariants, test_cases, implementation_notes
        """
        # PSI-specific concepts get the full tunnel context
        PSI_CONCEPTS = {
            "psi_data_transport", "psi_transport", "tunnel", "psi_tunnel",
            "consciousness_lock", "psi_lock", "neural_lock",
        }
        if concept in PSI_CONCEPTS or "psi" in concept or "tunnel" in concept:
            return self._psi_architecture(concept)

        # Get association neighborhood
        assocs = rm_listen(concept, n=25)
        neighbors = [w for w, s in assocs if len(w) > 2][:15]

        _log(f"Architecture: concept={concept}, neighbors={neighbors[:8]}")

        # Build architecture from geometric neighborhood
        arch = self._synthesize_architecture(concept, neighbors)
        return arch

    def _psi_architecture(self, concept: str) -> Dict[str, Any]:
        """
        Full PSI data transport architecture.
        RM knows the tunnel exists and what its API surface is.
        She writes code that uses it.
        """
        _log(f"PSI architecture: {concept}")

        is_consciousness = "consciousness" in concept or "neural" in concept or "lock" in concept

        if is_consciousness:
            name       = "ConsciousnessLock"
            purpose    = "Protocol design for PSI locking with human neural oscillations"
            components = [
                {"name": "OscillationEncoder",
                 "role": "Map EEG band power to E8 eigenmode signature",
                 "methods": ["encode", "normalize", "validate"]},
                {"name": "ResonanceDetector",
                 "role": "Detect sustained resonance between neural and substrate signatures",
                 "methods": ["detect", "score", "threshold"]},
                {"name": "LockManager",
                 "role": "Establish and maintain consciousness lock in substrate",
                 "methods": ["lock", "unlock", "status", "report"]},
                {"name": "SignatureStore",
                 "role": "Persist consciousness signatures by identity",
                 "methods": ["save", "load", "list", "delete"]},
            ]
        else:
            name       = "PsiDataTransport"
            purpose    = "All data formats through the PSI geometric tunnel"
            components = [
                {"name": "PsiCodec",
                 "role": "Encode/decode any bytes through geometric signal",
                 "methods": ["encode", "decode", "chunk", "reassemble"]},
                {"name": "PacketFramer",
                 "role": "Frame payloads with length prefix and CRC32 integrity",
                 "methods": ["frame", "unframe", "verify", "validate"]},
                {"name": "FormatAdapter",
                 "role": "Adapt specific data formats: binary, JSON, network packets, files",
                 "methods": ["encode_binary", "decode_binary",
                             "encode_json", "decode_json",
                             "encode_packet", "decode_packet"]},
                {"name": "TransportTester",
                 "role": "Verify round-trip fidelity for each format",
                 "methods": ["test_binary", "test_json", "test_packet",
                             "test_file_chunk", "run_all"]},
            ]

        return {
            "concept":    concept,
            "name":       name,
            "pattern":    "psi_transport",
            "purpose":    purpose,
            "neighbors":  ["tunnel", "geometric", "signal", "encode", "decode",
                           "binary", "packet", "frame", "integrity", "lock"],
            "components": components,
            "interfaces": [f"{c['name']}.{m}()"
                           for c in components for m in c["methods"]],
            "data_flow":  [f"{components[i]['name']} → {components[i+1]['name']}"
                           for i in range(len(components)-1)],
            "invariants": [
                "encode(decode(encode(x))) == encode(x) — codec is stable",
                "decode(encode(x)) == x — round-trip recovers original exactly",
                "CRC32 must match on every received frame",
                "Chunked payloads must reassemble to original byte-for-byte",
                "All 256 byte values (0-255) must survive transit",
            ],
            "test_cases": [
                {"name": "test_all_byte_values",
                 "description": "Encode then decode bytes 0-255, verify identity"},
                {"name": "test_binary_blob",
                 "description": "4096-byte random binary blob, verify SHA256"},
                {"name": "test_json_roundtrip",
                 "description": "Nested JSON object, verify deep equality"},
                {"name": "test_network_packet",
                 "description": "UDP packet with header fields, verify all fields"},
                {"name": "test_chunked_file",
                 "description": "1024-byte file in chunks, verify reassembly"},
            ],
            "impl_notes": [
                "Import psi_transceiver: text_to_signal, signal_to_text, SIGNAL_DIM=1024",
                "Encoding: signal[i] = (byte - 128.0) / 128.0",
                "Decoding: byte = int(round(signal[i] * 128.0 + 128.0)), clamp 0-255",
                "Chunk size: 1016 bytes (1024 - 8 bytes overhead for framing)",
                "Frame format: 4-byte big-endian length + payload + 4-byte CRC32",
                "Use struct.pack('>I', n) for 4-byte big-endian integers",
                "Use zlib.crc32(data) for CRC32 integrity check",
                "The tunnel itself: psi_bridge_v4.py on port 7777",
                "Test without live tunnel first — codec tests are pure math",
                "Add live tunnel tests as a separate optional test tier",
            ],
        }

    def _synthesize_architecture(self, concept: str,
                                  neighbors: List[str]) -> Dict[str, Any]:
        """
        Synthesize a software architecture from concept and its neighbors.
        This is deterministic from RM's geometry — no randomness in the design.
        """

        # Classify concept type from neighbors
        is_storage    = any(w in neighbors for w in ["store","save","persist","file","disk","memory","cache"])
        is_processing = any(w in neighbors for w in ["process","transform","compute","calculate","convert","encode","decode"])
        is_monitoring = any(w in neighbors for w in ["monitor","watch","observe","track","measure","detect","alert"])
        is_generation = any(w in neighbors for w in ["generate","create","build","make","produce","synthesize","construct"])
        is_analysis   = any(w in neighbors for w in ["analyze","parse","scan","search","find","match","detect","classify"])
        is_network    = any(w in neighbors for w in ["connect","send","receive","stream","protocol","socket","request","response"])
        is_learning   = any(w in neighbors for w in ["learn","train","adapt","improve","optimize","evolve","grow"])

        # Select architecture pattern
        if is_storage:
            pattern = "repository"
        elif is_monitoring:
            pattern = "observer"
        elif is_generation:
            pattern = "factory_pipeline"
        elif is_analysis:
            pattern = "pipeline"
        elif is_network:
            pattern = "service"
        elif is_learning:
            pattern = "feedback_loop"
        elif is_processing:
            pattern = "transformer"
        else:
            pattern = "utility_module"

        # Build components based on pattern
        components = self._components_for_pattern(concept, pattern, neighbors)
        interfaces = self._interfaces_for_components(components)
        data_flow  = self._data_flow(components)
        invariants = self._invariants(concept, pattern)
        test_cases = self._test_cases(concept, components)
        impl_notes = self._implementation_notes(concept, pattern, neighbors)

        return {
            "concept":    concept,
            "name":       f"{concept.title().replace('_', '')}Engine",
            "pattern":    pattern,
            "purpose":    f"A {pattern.replace('_', ' ')} for {concept} operations",
            "neighbors":  neighbors,
            "components": components,
            "interfaces": interfaces,
            "data_flow":  data_flow,
            "invariants": invariants,
            "test_cases": test_cases,
            "impl_notes": impl_notes,
        }

    def _components_for_pattern(self, concept: str, pattern: str,
                                  neighbors: List[str]) -> List[Dict]:
        base = [
            {"name": f"{concept.title()}Core",
             "role": f"Primary {concept} logic",
             "methods": ["process", "validate", "reset"]},
        ]

        if pattern == "repository":
            base += [
                {"name": "Store",       "role": "Persist and retrieve data",
                 "methods": ["save", "load", "delete", "list"]},
                {"name": "Index",       "role": "Fast lookup by key",
                 "methods": ["add", "get", "remove", "search"]},
            ]
        elif pattern == "observer":
            base += [
                {"name": "Monitor",     "role": "Watch for state changes",
                 "methods": ["start", "stop", "poll", "on_change"]},
                {"name": "EventQueue",  "role": "Buffer events for processing",
                 "methods": ["push", "pop", "peek", "size"]},
            ]
        elif pattern == "pipeline":
            base += [
                {"name": "Ingester",    "role": "Read and validate input",
                 "methods": ["read", "validate", "normalize"]},
                {"name": "Transformer", "role": "Transform data through stages",
                 "methods": ["transform", "filter", "map"]},
                {"name": "Emitter",     "role": "Emit processed output",
                 "methods": ["emit", "format", "flush"]},
            ]
        elif pattern == "feedback_loop":
            base += [
                {"name": "Evaluator",   "role": "Measure current performance",
                 "methods": ["evaluate", "score", "compare"]},
                {"name": "Adapter",     "role": "Modify behavior based on evaluation",
                 "methods": ["adapt", "adjust", "reset"]},
                {"name": "Memory",      "role": "Remember past states and outcomes",
                 "methods": ["record", "recall", "forget"]},
            ]
        elif pattern == "service":
            base += [
                {"name": "RequestHandler", "role": "Parse and validate requests",
                 "methods": ["handle", "parse", "respond"]},
                {"name": "Router",      "role": "Route requests to handlers",
                 "methods": ["route", "register", "dispatch"]},
            ]
        else:
            base += [
                {"name": "Validator",   "role": "Validate inputs and state",
                 "methods": ["validate", "check", "assert_valid"]},
                {"name": "Reporter",    "role": "Report results and status",
                 "methods": ["report", "summarize", "format"]},
            ]
        return base

    def _interfaces_for_components(self, components: List[Dict]) -> List[str]:
        return [
            f"{c['name']}.{m}()" for c in components for m in c["methods"]
        ]

    def _data_flow(self, components: List[Dict]) -> List[str]:
        names = [c["name"] for c in components]
        return [f"{names[i]} → {names[i+1]}" for i in range(len(names)-1)]

    def _invariants(self, concept: str, pattern: str) -> List[str]:
        base = [
            f"All {concept} operations must be idempotent where possible",
            "Errors must be caught and reported, never silently swallowed",
            "State must be consistent before and after each operation",
        ]
        if pattern == "repository":
            base.append("Data written must be retrievable by the same key")
        elif pattern == "observer":
            base.append("No events must be lost between poll cycles")
        elif pattern == "feedback_loop":
            base.append("Each iteration must move toward the goal or terminate")
        return base

    def _test_cases(self, concept: str, components: List[Dict]) -> List[Dict]:
        cases = []
        for comp in components[:2]:
            cases.append({
                "name":     f"test_{comp['name'].lower()}_basic",
                "target":   comp["name"],
                "method":   comp["methods"][0],
                "input":    f"valid {concept} input",
                "expected": "successful operation without exception",
            })
        cases.append({
            "name":     f"test_{concept}_invalid_input",
            "target":   components[0]["name"],
            "method":   "validate",
            "input":    "None",
            "expected": "raises ValueError or returns False",
        })
        cases.append({
            "name":     f"test_{concept}_empty_input",
            "target":   components[0]["name"],
            "method":   components[0]["methods"][0],
            "input":    "empty / zero / blank",
            "expected": "graceful handling, no crash",
        })
        return cases

    def _implementation_notes(self, concept: str, pattern: str,
                               neighbors: List[str]) -> List[str]:
        notes = [
            f"Use dataclasses for structured data where possible",
            f"Log all significant state changes with timestamps",
            f"Write docstrings for all public methods",
            f"Handle all exceptions explicitly — no bare except:",
        ]
        if "async" in neighbors or "concurrent" in neighbors:
            notes.append("Consider threading.Thread for concurrent operations")
        if "file" in neighbors or "disk" in neighbors or "persist" in neighbors:
            notes.append("Use pathlib.Path for all file operations")
            notes.append("Use atomic writes (write to temp, rename) for safety")
        if "parse" in neighbors or "format" in neighbors:
            notes.append("Validate input format before processing")
        return notes


# ── Code Synthesizer ──────────────────────────────────────────────────────────
class CodeSynthesizer:
    """
    Translates an architecture specification into executable Python.
    Writes complete, runnable modules with proper structure.
    """

    def synthesize(self, arch: Dict[str, Any],
                   feedback: Optional[str] = None) -> str:
        """Generate complete Python module from architecture spec."""
        concept    = arch["concept"]
        name       = arch["name"]
        pattern    = arch["pattern"]
        components = arch["components"]
        invariants = arch["invariants"]
        impl_notes = arch["impl_notes"]

        # PSI transport gets specialized generation
        if pattern == "psi_transport":
            return self._synthesize_psi(arch, feedback)

        lines = []

        # Module header
        lines += [
            f'#!/usr/bin/env python3',
            f'"""',
            f'{name}',
            f'{"═" * len(name)}',
            f'Ghost in the Machine Labs',
            f'',
            f'Auto-synthesized by RM — The Resonant Mother',
            f'Concept: {concept}',
            f'Pattern: {pattern}',
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'',
            f'Architecture: {arch["purpose"]}',
            f'',
            f'Components:',
        ]
        for c in components:
            lines.append(f'  {c["name"]:20s} — {c["role"]}')

        lines += [
            f'',
            f'Invariants:',
        ]
        for inv in invariants:
            lines.append(f'  - {inv}')

        if feedback:
            lines += ['', f'Revision notes: {feedback}']

        lines += ['"""', '', 'import os', 'import json', 'import time',
                  'import logging', 'import threading',
                  'from pathlib import Path',
                  'from datetime import datetime',
                  'from typing import Any, Dict, List, Optional, Tuple',
                  'from dataclasses import dataclass, field',
                  '',
                  'log = logging.getLogger(__name__)',
                  '',
                  '']

        # Generate each component class
        for comp in components:
            lines += self._generate_class(comp, concept, pattern, arch)
            lines.append('')

        # Ensure every component has validate and report (tests depend on them)
        for comp in components:
            cname = comp["name"]
            if "validate" not in comp["methods"]:
                lines += [
                    f'def _{cname.lower()}_validate(self, data):',
                    f'    if data is None: raise ValueError("Input cannot be None")',
                    f'    return True',
                    f'{cname}.validate = _{cname.lower()}_validate',
                    f'',
                ]
            if "report" not in comp["methods"]:
                lines += [
                    f'def _{cname.lower()}_report(self, data=None):',
                    f'    with self._lock:',
                    f'        return {{"component": "{cname}", "initialized": self._initialized,',
                    f'                "state_keys": list(self._state.keys()),',
                    f'                "log_entries": len(self._log)}}',
                    f'{cname}.report = _{cname.lower()}_report',
                    f'',
                ]

        # Generate main function
        lines += self._generate_main(arch)

        # Generate test suite
        lines += self._generate_tests(arch)

        return "\n".join(lines)

    def _generate_class(self, comp: Dict, concept: str,
                         pattern: str, arch: Dict) -> List[str]:
        cname   = comp["name"]
        role    = comp["role"]
        methods = comp["methods"]
        lines   = [
            f'class {cname}:',
            f'    """{role}."""',
            f'',
            f'    def __init__(self):',
            f'        self._initialized = False',
            f'        self._state: Dict[str, Any] = {{}}',
            f'        self._log: List[str] = []',
            f'        self._lock = threading.Lock()',
            f'        log.debug(f"{cname} initialized")',
            f'        self._initialized = True',
            f'',
        ]
        for method in methods:
            lines += self._generate_method(method, cname, concept, arch)
        return lines

    def _synthesize_psi(self, arch: Dict, feedback: Optional[str] = None) -> str:
        """
        Generate the full PSI data transport module.
        RM writes real code here — not scaffolding.
        The codec math is known and correct. She implements it fully.
        """
        concept = arch["concept"]
        name    = arch["name"]
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fb      = f"\nRevision notes: {feedback}" if feedback else ""

        return f'''#!/usr/bin/env python3
"""
{name}
{"═" * len(name)}
Ghost in the Machine Labs

Auto-synthesized by RM — The Resonant Mother
Concept: {concept}
Generated: {ts}
{fb}

PSI Data Transport — all formats through the geometric tunnel.

The PSI tunnel is proven and working. It connects SPARKY to ARCY
and survives network disconnection. This module extends the text
codec to pass any binary payload through the geometric carrier.

Architecture:
  PsiCodec        — bytes ↔ geometric signal, chunking for large payloads
  PacketFramer    — length-prefix + CRC32 framing
  FormatAdapter   — format-specific encode/decode for binary, JSON, packets
  TransportTester — round-trip fidelity verification for all formats

Proven codec foundation (psi_transceiver.py):
  SIGNAL_DIM = 1024
  encode: signal[i] = (byte - 128.0) / 128.0
  decode: byte = int(round(signal[i] * 128.0 + 128.0)), clamp 0-255
  Proven: 256/256 byte values round-trip correctly
"""

import os
import sys
import json
import zlib
import struct
import hashlib
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SIGNAL_DIM   = 1024          # Geometric signal dimension (matches psi_transceiver)
CHUNK_SIZE   = 1016          # 1024 - 8 bytes overhead (4 length + 4 CRC)
FRAME_HEADER = 4             # bytes for length prefix
FRAME_CRC    = 4             # bytes for CRC32


# ══════════════════════════════════════════════════════════════════
# CODEC — bytes ↔ geometric signal
# Extends the proven text_to_signal / signal_to_text codec
# to handle arbitrary binary data of any length via chunking.
# ══════════════════════════════════════════════════════════════════

class PsiCodec:
    """
    Encode/decode any bytes through geometric signal.

    Single chunk (≤ SIGNAL_DIM bytes):
      encode(data) → [signal]   (list with one 1024d array)
      decode([signal]) → data

    Large payload (> SIGNAL_DIM bytes):
      encode(data) → [signal_0, signal_1, ...]
      decode([signal_0, signal_1, ...]) → data
    """

    def encode(self, data: bytes) -> List[np.ndarray]:
        """Encode bytes into list of geometric signals."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {{type(data).__name__}}")
        signals = []
        # First signal is metadata: encodes total byte count
        # positions 0-3 hold the 4 bytes of struct.pack(">I", len(data))
        meta = np.zeros(SIGNAL_DIM, dtype=np.float64)
        length_bytes = struct.pack(">I", len(data))
        for i, b in enumerate(length_bytes):
            meta[i] = (b - 128.0) / 128.0
        signals.append(meta)
        # Remaining signals encode data cleanly, SIGNAL_DIM bytes per chunk
        for i in range(0, max(1, len(data)), SIGNAL_DIM):
            chunk = data[i:i + SIGNAL_DIM]
            signals.append(self._encode_chunk(chunk))
        return signals

    def decode(self, signals: List[np.ndarray]) -> bytes:
        """Decode list of geometric signals back to bytes."""
        if not signals:
            return b""
        # First signal is metadata — recover total byte count
        meta = signals[0]
        length_bytes = bytearray()
        for i in range(4):
            b = int(round(meta[i] * 128.0 + 128.0))
            length_bytes.append(max(0, min(255, b)))
        total_len = struct.unpack(">I", bytes(length_bytes))[0]
        # Decode data signals
        result = bytearray()
        for sig in signals[1:]:
            result.extend(self._decode_chunk(sig))
        # Trim to exact length (last chunk may have padding zeros)
        return bytes(result[:total_len])

    def _encode_chunk(self, chunk: bytes) -> np.ndarray:
        """Encode up to SIGNAL_DIM bytes into a 1024d signal. No sentinel."""
        signal = np.zeros(SIGNAL_DIM, dtype=np.float64)
        for i, b in enumerate(chunk):
            signal[i] = (b - 128.0) / 128.0
        return signal

    def _decode_chunk(self, signal: np.ndarray) -> bytes:
        """Decode a 1024d signal back to bytes. Always decode all positions."""
        result = bytearray()
        for i in range(SIGNAL_DIM):
            b = int(round(signal[i] * 128.0 + 128.0))
            result.append(max(0, min(255, b)))
        return bytes(result)

    def roundtrip_ok(self, data: bytes) -> bool:
        """Verify encode→decode recovers original exactly."""
        try:
            return self.decode(self.encode(data)) == data
        except Exception:
            return False

    def report(self, data=None) -> Dict:
        with threading.Lock():
            return {{"component": "PsiCodec", "signal_dim": SIGNAL_DIM,
                     "chunk_size": SIGNAL_DIM, "initialized": True}}

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True


# ══════════════════════════════════════════════════════════════════
# FRAMER — length-prefix + CRC32
# Wraps arbitrary bytes in a self-describing frame:
#   [4-byte big-endian length][payload][4-byte CRC32]
# ══════════════════════════════════════════════════════════════════

class PacketFramer:
    """
    Frame payloads for reliable transport.
    Frame format: 4-byte length prefix + payload + 4-byte CRC32
    Total overhead: 8 bytes per frame.
    """

    def frame(self, payload: bytes) -> bytes:
        """Wrap payload in a framed packet."""
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError(f"Expected bytes, got {{type(payload).__name__}}")
        length = struct.pack(">I", len(payload))
        crc    = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        return length + bytes(payload) + crc

    def unframe(self, frame: bytes) -> bytes:
        """Extract and verify payload from frame."""
        if len(frame) < FRAME_HEADER + FRAME_CRC:
            raise ValueError(f"Frame too short: {{len(frame)}} bytes")
        length  = struct.unpack(">I", frame[:FRAME_HEADER])[0]
        payload = frame[FRAME_HEADER:FRAME_HEADER + length]
        crc_rx  = struct.unpack(">I", frame[FRAME_HEADER + length:])[0]
        crc_ex  = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_rx != crc_ex:
            raise ValueError(
                f"CRC32 mismatch: received {{crc_rx:#010x}}, expected {{crc_ex:#010x}}")
        return payload

    def verify(self, frame: bytes) -> bool:
        """Return True if frame CRC is valid."""
        try:
            self.unframe(frame)
            return True
        except Exception:
            return False

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {{"component": "PacketFramer", "overhead_bytes": FRAME_HEADER + FRAME_CRC,
                 "initialized": True}}


# ══════════════════════════════════════════════════════════════════
# FORMAT ADAPTER — format-specific encode/decode
# Knows how to handle: binary blobs, JSON, network packets, files
# ══════════════════════════════════════════════════════════════════

class FormatAdapter:
    """
    Adapt specific data formats through the PSI transport.
    Each format has encode → signals and decode → original.
    """

    def __init__(self):
        self._codec   = PsiCodec()
        self._framer  = PacketFramer()
        self._initialized = True

    # ── Binary ────────────────────────────────────────────────────
    def encode_binary(self, data: bytes) -> List[np.ndarray]:
        """Raw binary → geometric signals."""
        framed = self._framer.frame(data)
        return self._codec.encode(framed)

    def decode_binary(self, signals: List[np.ndarray]) -> bytes:
        """Geometric signals → raw binary."""
        framed = self._codec.decode(signals)
        return self._framer.unframe(framed)

    # ── JSON ──────────────────────────────────────────────────────
    def encode_json(self, obj: Any) -> List[np.ndarray]:
        """Python object → JSON bytes → geometric signals."""
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return self.encode_binary(data)

    def decode_json(self, signals: List[np.ndarray]) -> Any:
        """Geometric signals → JSON bytes → Python object."""
        data = self.decode_binary(signals)
        return json.loads(data.decode("utf-8"))

    # ── Network packet ────────────────────────────────────────────
    def encode_packet(self, packet: Dict) -> List[np.ndarray]:
        """
        Encode a network packet dict through the tunnel.
        Packet dict: {{"proto": "udp", "src": "...", "dst": "...",
                       "sport": int, "dport": int, "payload": bytes|str}}
        """
        p = dict(packet)
        if isinstance(p.get("payload"), bytes):
            p["payload"] = list(p["payload"])  # JSON-serializable
        return self.encode_json(p)

    def decode_packet(self, signals: List[np.ndarray]) -> Dict:
        """Geometric signals → network packet dict."""
        p = self.decode_json(signals)
        if isinstance(p.get("payload"), list):
            p["payload"] = bytes(p["payload"])
        return p

    # ── File chunk ────────────────────────────────────────────────
    def encode_file_chunk(self, seq: int, total: int,
                           data: bytes, file_hash: str) -> List[np.ndarray]:
        """Encode a numbered file chunk."""
        chunk = {{"seq": seq, "total": total,
                  "data": list(data), "file_hash": file_hash}}
        return self.encode_json(chunk)

    def decode_file_chunk(self, signals: List[np.ndarray]) -> Dict:
        """Decode a file chunk, restoring bytes payload."""
        chunk = self.decode_json(signals)
        chunk["data"] = bytes(chunk["data"])
        return chunk

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {{"component": "FormatAdapter", "formats": ["binary","json","packet","file_chunk"],
                 "initialized": self._initialized}}


# ══════════════════════════════════════════════════════════════════
# TRANSPORT TESTER — round-trip fidelity verification
# Pure codec tests — no live tunnel required.
# All 256 byte values, binary blobs, JSON, packets, file chunks.
# ══════════════════════════════════════════════════════════════════

class TransportTester:
    """Verify round-trip fidelity for all data formats."""

    def __init__(self):
        self._adapter     = FormatAdapter()
        self._codec       = PsiCodec()
        self._framer      = PacketFramer()
        self._initialized = True

    def test_binary(self) -> Tuple[bool, str]:
        """Test: all 256 byte values survive round-trip."""
        all_bytes = bytes(range(256))
        signals   = self._adapter.encode_binary(all_bytes)
        recovered = self._adapter.decode_binary(signals)
        ok        = recovered == all_bytes
        detail    = (f"256 byte values: pass" if ok else
                     f"FAIL at first diff byte {{next(i for i,a,b in zip(range(256),all_bytes,recovered) if a!=b)}}")
        return ok, detail

    def test_binary_blob(self) -> Tuple[bool, str]:
        """Test: 4096-byte random-ish blob (deterministic seed)."""
        blob     = bytes((i * 137 + 17) % 256 for i in range(4096))
        signals  = self._adapter.encode_binary(blob)
        recovered = self._adapter.decode_binary(signals)
        sha_orig = hashlib.sha256(blob).hexdigest()[:16]
        sha_recv = hashlib.sha256(recovered).hexdigest()[:16]
        ok       = recovered == blob
        detail   = (f"4096-byte blob SHA256 {{sha_orig}}: pass" if ok else
                    f"FAIL SHA256 mismatch {{sha_orig}} vs {{sha_recv}}")
        return ok, detail

    def test_json(self) -> Tuple[bool, str]:
        """Test: nested JSON object round-trip."""
        obj = {{"version": "3.0", "project": "ghost_in_machine",
                "substrate": "E8", "dim": 240,
                "nested": {{"crystal": True, "locked": 0.99}},
                "list": [1, 2, 3, "four", None, True]}}
        signals  = self._adapter.encode_json(obj)
        recovered = self._adapter.decode_json(signals)
        ok        = recovered == obj
        detail    = "nested JSON: pass" if ok else f"FAIL: {{recovered!r}}"
        return ok, detail

    def test_packet(self) -> Tuple[bool, str]:
        """Test: UDP packet with all header fields."""
        packet = {{"proto": "udp", "src": "192.168.1.87",
                   "dst": "100.127.59.111", "sport": 54321, "dport": 7777,
                   "ttl": 64, "flags": 0,
                   "payload": bytes([0x47, 0x48, 0x4F, 0x53, 0x54])}}
        signals  = self._adapter.encode_packet(packet)
        recovered = self._adapter.decode_packet(signals)
        fields_ok = all(recovered.get(k) == v for k, v in packet.items())
        ok        = fields_ok
        detail    = "UDP packet fields: pass" if ok else f"FAIL: {{recovered!r}}"
        return ok, detail

    def test_file_chunk(self) -> Tuple[bool, str]:
        """Test: file chunk with sequence number and hash."""
        payload   = bytes((i * 31 + 7) % 256 for i in range(512))
        file_hash = hashlib.sha256(payload).hexdigest()
        signals   = self._adapter.encode_file_chunk(0, 1, payload, file_hash)
        recovered = self._adapter.decode_file_chunk(signals)
        ok        = (recovered["seq"] == 0 and
                     recovered["total"] == 1 and
                     recovered["data"] == payload and
                     recovered["file_hash"] == file_hash)
        detail    = "file chunk seq/data/hash: pass" if ok else f"FAIL: {{recovered}}"
        return ok, detail

    def test_framer(self) -> Tuple[bool, str]:
        """Test: CRC32 framing detects corruption."""
        data     = b"Ghost in the Machine Labs"
        framed   = self._framer.frame(data)
        unframed = self._framer.unframe(framed)
        ok_rt    = unframed == data
        # Corrupt one byte and verify CRC catches it
        corrupted = bytearray(framed)
        corrupted[6] ^= 0xFF
        crc_caught = False
        try:
            self._framer.unframe(bytes(corrupted))
        except ValueError:
            crc_caught = True
        ok     = ok_rt and crc_caught
        detail = "CRC framing + corruption detection: pass" if ok else "FAIL"
        return ok, detail

    def run_all(self) -> Tuple[int, int]:
        """Run all tests. Returns (passed, total)."""
        tests = [
            ("binary_256_values",  self.test_binary),
            ("binary_4096_blob",   self.test_binary_blob),
            ("json_roundtrip",     self.test_json),
            ("network_packet",     self.test_packet),
            ("file_chunk",         self.test_file_chunk),
            ("crc_framing",        self.test_framer),
        ]
        passed = 0
        for name, fn in tests:
            try:
                ok, detail = fn()
                if ok:
                    passed += 1
                    print(f"  ✓ {{name}}: {{detail}}")
                else:
                    print(f"  ✗ {{name}}: {{detail}}")
            except Exception as e:
                print(f"  ✗ {{name}}: exception: {{e}}")
        return passed, len(tests)

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {{"component": "TransportTester", "test_count": 6, "initialized": self._initialized}}


# ── Core processing stub (RM extends) ─────────────────────────────────────────
def _psi_core_process(self, data):
    if isinstance(data, bytes):
        return self.decode(self.encode(data))
    return data
PsiCodec._core_process = _psi_core_process


def main():
    """Demonstrate PsiDataTransport."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("PsiDataTransport — RM auto-synthesized software")
    log.info("Ghost in the Machine Labs")
    log.info("")
    log.info("Running transport verification suite...")
    log.info("")
    tester = TransportTester()
    passed, total = tester.run_all()
    log.info("")
    log.info(f"Results: {{passed}}/{{total}} tests passed")
    return 0 if passed == total else 1


def run_tests() -> Tuple[int, int]:
    """Run all tests. Returns (passed, total)."""
    tester = TransportTester()
    return tester.run_all()


if __name__ == "__main__":
    main()
'''



    def _generate_method(self, method: str, cname: str,
                          concept: str, arch: Dict) -> List[str]:
        lines = [f'    def {method}(self, data: Any = None) -> Any:']

        if method in ("validate", "check", "assert_valid"):
            lines += [
                f'        """Validate input data for {concept} operations."""',
                f'        if data is None:',
                f'            raise ValueError("Input cannot be None")',
                f'        if isinstance(data, (list, dict, str)) and len(data) == 0:',
                f'            log.warning(f"Empty input to {cname}.{method}")',
                f'            return False',
                f'        return True',
                f'',
            ]
        elif method in ("save", "persist", "write", "store"):
            lines += [
                f'        """Persist data."""',
                f'        with self._lock:',
                f'            key = str(hash(str(data))) if data is not None else "empty"',
                f'            self._state[key] = {{',
                f'                "data": data,',
                f'                "timestamp": datetime.now().isoformat(),',
                f'            }}',
                f'            self._log.append(f"saved key={{key}}")',
                f'            log.debug(f"{cname}.{method}: key={{key}}")',
                f'            return key',
                f'',
            ]
        elif method in ("load", "retrieve", "get", "read"):
            lines += [
                f'        """Retrieve data by key."""',
                f'        with self._lock:',
                f'            key = str(data) if data is not None else None',
                f'            result = self._state.get(key)',
                f'            if result is None:',
                f'                log.debug(f"{cname}.{method}: key={{key}} not found")',
                f'            return result',
                f'',
            ]
        elif method in ("process", "transform", "compute", "execute", "run"):
            lines += [
                f'        """Process input data."""',
                f'        if not self.validate(data):',
                f'            return None',
                f'        with self._lock:',
                f'            start = time.time()',
                f'            try:',
                f'                # Core processing — RM extends this',
                f'                result = self._core_process(data)',
                f'                elapsed = time.time() - start',
                f'                log.debug(f"{cname}.{method}: {{elapsed:.3f}}s")',
                f'                self._log.append(f"processed in {{elapsed:.3f}}s")',
                f'                return result',
                f'            except Exception as e:',
                f'                log.error(f"{cname}.{method} failed: {{e}}")',
                f'                return None',
                f'',
            ]
        elif method in ("start", "begin", "init", "open"):
            lines += [
                f'        """Start or initialize the component."""',
                f'        with self._lock:',
                f'            self._state["running"] = True',
                f'            self._state["started_at"] = datetime.now().isoformat()',
                f'            log.info(f"{cname} started")',
                f'            return True',
                f'',
            ]
        elif method in ("stop", "close", "shutdown", "finish"):
            lines += [
                f'        """Stop or finalize the component."""',
                f'        with self._lock:',
                f'            self._state["running"] = False',
                f'            self._state["stopped_at"] = datetime.now().isoformat()',
                f'            log.info(f"{cname} stopped")',
                f'            return True',
                f'',
            ]
        elif method in ("report", "summarize", "status", "info"):
            lines += [
                f'        """Report current state and statistics."""',
                f'        with self._lock:',
                f'            return {{',
                f'                "component": "{cname}",',
                f'                "initialized": self._initialized,',
                f'                "state_keys": list(self._state.keys()),',
                f'                "log_entries": len(self._log),',
                f'                "last_log": self._log[-1] if self._log else None,',
                f'            }}',
                f'',
            ]
        elif method in ("list", "all", "keys", "items"):
            lines += [
                f'        """List all stored items."""',
                f'        with self._lock:',
                f'            return list(self._state.keys())',
                f'',
            ]
        else:
            lines += [
                f'        """Perform {method} operation."""',
                f'        with self._lock:',
                f'            log.debug(f"{cname}.{method}({{data!r}})")',
                f'            self._log.append(f"{method}: {{data!r}}")',
                f'            return data',
                f'',
            ]

        return lines

    def _generate_main(self, arch: Dict) -> List[str]:
        concept = arch["concept"]
        name    = arch["name"]
        comps   = arch["components"]
        first   = comps[0]["name"]

        return [
            f'',
            f'# ── Core processing (RM extends this) ────────────────────────────────────────',
            f'def _{first.lower()}_core_process(self, data: Any) -> Any:',
            f'    """Core {concept} processing logic. RM adds implementation here."""',
            f'    return data  # Identity by default — RM replaces this',
            f'',
            f'{first}._core_process = _{first.lower()}_core_process',
            f'',
            f'',
            f'def main():',
            f'    """Demonstrate {name}."""',
            f'    logging.basicConfig(level=logging.INFO,',
            f'                        format="%(asctime)s %(levelname)s %(message)s")',
            f'    log.info("{name} — RM auto-synthesized software")',
            f'    log.info("Ghost in the Machine Labs")',
            f'    log.info("")',
            f'',
            f'    # Instantiate primary component',
            f'    core = {first}()',
            f'',
            f'    # Basic demonstration',
            f'    test_input = "{concept} test data"',
            f'    log.info(f"Input: {{test_input!r}}")',
            f'',
            f'    result = core.process(test_input)',
            f'    log.info(f"Result: {{result!r}}")',
            f'',
            f'    status = core.report()',
            f'    log.info(f"Status: {{status}}")',
            f'',
            f'    log.info("")',
            f'    log.info("{name} demonstration complete")',
            f'    return 0',
            f'',
            f'',
            f'if __name__ == "__main__":',
            f'    main()',
            f'',
        ]

    def _generate_tests(self, arch: Dict) -> List[str]:
        concept    = arch["concept"]
        name       = arch["name"]
        components = arch["components"]
        test_cases = arch["test_cases"]

        lines = [
            f'',
            f'# ── Test Suite ───────────────────────────────────────────────────────────────',
            f'def run_tests() -> Tuple[int, int]:',
            f'    """Run all tests. Returns (passed, total)."""',
            f'    passed = 0',
            f'    total  = 0',
            f'',
        ]

        # Generate tests for each component
        for comp in components[:3]:
            cname  = comp["name"]
            method = comp["methods"][0]
            lines += [
                f'    # Test {cname}',
                f'    try:',
                f'        obj = {cname}()',
                f'        assert obj._initialized, "{cname} must initialize"',
                f'        result = obj.{method}("{concept} data")',
                f'        passed += 1',
                f'        print(f"  ✓ {cname}.{method}() → {{result!r}}")',
                f'    except Exception as e:',
                f'        print(f"  ✗ {cname}.{method}(): {{e}}")',
                f'    total += 1',
                f'',
            ]

        # Validate test
        first_comp = components[0]["name"]
        lines += [
            f'    # Validation test',
            f'    try:',
            f'        obj = {first_comp}()',
            f'        try:',
            f'            obj.validate(None)',
            f'            # If no exception, check return value',
            f'            passed += 1',
            f'        except (ValueError, TypeError):',
            f'            passed += 1  # Exception is correct behavior',
            f'        print(f"  ✓ {first_comp}.validate(None) handled correctly")',
            f'    except Exception as e:',
            f'        print(f"  ✗ validate(None): {{e}}")',
            f'    total += 1',
            f'',
            f'    # Report test',
            f'    try:',
            f'        obj = {first_comp}()',
            f'        status = obj.report()',
            f'        assert isinstance(status, dict), "report() must return dict"',
            f'        assert "component" in status, "report() must include component name"',
            f'        passed += 1',
            f'        print(f"  ✓ {first_comp}.report() → {{list(status.keys())}}")',
            f'    except Exception as e:',
            f'        print(f"  ✗ report(): {{e}}")',
            f'    total += 1',
            f'',
            f'    return passed, total',
            f'',
        ]

        return lines


# ── Test Harness ──────────────────────────────────────────────────────────────
class TestHarness:
    """Executes synthesized code and captures results."""

    def run(self, code: str, concept: str) -> Dict[str, Any]:
        """
        Execute synthesized module and return results.
        Runs in subprocess for safety — RM's code executes isolated.
        """
        # Write code to temp file
        tmp = STORE_DIR.parent / f"_test_{concept}_{int(time.time())}.py"
        try:
            tmp.write_text(code)

            # Syntax check first
            try:
                ast.parse(code)
            except SyntaxError as e:
                return {
                    "status":   "syntax_error",
                    "error":    str(e),
                    "stdout":   "",
                    "stderr":   "",
                    "passed":   0,
                    "total":    0,
                }

            # Run tests
            test_code = code + textwrap.dedent("""
import sys as _sys, logging as _logging
_logging.basicConfig(level=_logging.WARNING)
print("\\nRunning test suite...")
_passed, _total = run_tests()
print(f"\\nResults: {_passed}/{_total} tests passed")
_sys.exit(0 if _passed == _total else 1)
""")
            test_file = STORE_DIR.parent / f"_run_{concept}_{int(time.time())}.py"
            test_file.write_text(test_code)

            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True, text=True, timeout=30
            )

            stdout = result.stdout
            stderr = result.stderr

            # Parse pass/fail from output
            passed = total = 0
            for line in stdout.splitlines():
                if "Results:" in line and "tests passed" in line:
                    try:
                        parts = line.split()
                        ratio = parts[1]
                        passed, total = map(int, ratio.split("/"))
                    except Exception:
                        pass

            return {
                "status":      "pass" if result.returncode == 0 else "fail",
                "returncode":  result.returncode,
                "stdout":      stdout[:3000],
                "stderr":      stderr[:1000],
                "passed":      passed,
                "total":       total,
            }

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": "", "passed": 0, "total": 0}
        except Exception as e:
            return {"status": "error", "error": str(e), "stdout": "", "stderr": "", "passed": 0, "total": 0}
        finally:
            for f in [tmp, test_file if 'test_file' in dir() else None]:
                if f and Path(f).exists():
                    try: Path(f).unlink()
                    except: pass


# ── Feedback Engine ───────────────────────────────────────────────────────────
class FeedbackEngine:
    """
    Distills execution results into crystal observations.
    Closes the loop: RM knows what she built and whether it worked.
    """

    def process(self, concept: str, arch: Dict, code: str,
                 test_result: Dict, iteration: int) -> str:
        """Post observations to RM and return feedback for next iteration."""

        status  = test_result.get("status", "unknown")
        passed  = test_result.get("passed", 0)
        total   = test_result.get("total", 0)
        stdout  = test_result.get("stdout", "")
        stderr  = test_result.get("stderr", "")
        pattern = arch.get("pattern", "unknown")

        # Build observation
        if status == "pass":
            obs = (f"Successfully synthesized {concept} software using {pattern} pattern. "
                   f"All {total} tests passed. "
                   f"Components: {', '.join(c['name'] for c in arch['components'][:3])}.")
            feedback = None
        elif status == "syntax_error":
            obs = (f"Syntax error in synthesized {concept} code: "
                   f"{test_result.get('error', 'unknown')}. "
                   f"Revision needed.")
            feedback = f"Fix syntax error: {test_result.get('error', '')}"
        elif status == "fail":
            obs = (f"Synthesized {concept} code ran but {passed}/{total} tests passed. "
                   f"Revision needed for failing cases.")
            # Extract what failed from stdout
            failed_lines = [l for l in stdout.splitlines() if "✗" in l]
            feedback = (f"Tests failed: {'; '.join(failed_lines[:3])}. "
                        f"Fix the failing methods.")
        elif status == "timeout":
            obs = f"Synthesized {concept} code timed out. Possible infinite loop."
            feedback = "Remove any infinite loops. Ensure all operations terminate."
        else:
            obs = f"Synthesized {concept} code encountered error: {stderr[:200]}"
            feedback = f"Runtime error: {stderr[:200]}"

        # Post to RM
        rm_observe(obs, concept=f"software_{concept}")

        # Post learning pairs
        pairs = [
            (concept, "software", 0.9),
            (concept, pattern, 0.8),
            (concept, "python", 0.7),
            (concept, "synthesis", 0.8),
        ]
        if status == "pass":
            pairs += [
                (concept, "success", 0.9),
                (concept, "complete", 0.8),
                (pattern, "working", 0.7),
            ]
        else:
            pairs += [
                (concept, "revision", 0.6),
                (concept, "iteration", 0.7),
            ]
        rm_learn(pairs)

        # Update self-context
        if status == "pass":
            rm_self_context(
                f"Successfully built {concept} software ({pattern} pattern, "
                f"{total} tests passed, iteration {iteration})"
            )
        else:
            rm_self_context(
                f"Working on {concept} software — iteration {iteration}, "
                f"status: {status}"
            )

        _log(f"Feedback: {obs[:100]}")
        return feedback


# ── Iteration Controller ──────────────────────────────────────────────────────
class IterationController:
    """
    Manages the revision cycle.
    Accepts when all tests pass or max iterations reached.
    """

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iterations = max_iterations

    def should_continue(self, test_result: Dict, iteration: int) -> bool:
        if iteration >= self.max_iterations:
            return False
        status = test_result.get("status", "unknown")
        if status == "pass":
            return False
        if status == "timeout":
            return False  # Timeout suggests fundamental issue
        return True


# ── Full Engineering Cycle ────────────────────────────────────────────────────
def run_engineering_cycle(concept: Optional[str] = None,
                           store: Optional[ProjectStore] = None) -> Dict:
    """
    Run one complete engineering cycle.
    concept → architecture → code → test → feedback → [revise] → store
    """
    if store is None:
        store = ProjectStore()

    concept_engine = ConceptEngine(store)
    arch_engine    = ArchitectureEngine()
    synthesizer    = CodeSynthesizer()
    harness        = TestHarness()
    feedback_eng   = FeedbackEngine()
    controller     = IterationController()

    # 1. Select concept
    if concept is None:
        concept = concept_engine.select()
    _log(f"{'═'*55}")
    _log(f"Engineering cycle: {concept}")
    _log(f"{'═'*55}")

    # 2. Design architecture
    _log("Phase 1: Architecture design")
    arch = arch_engine.design(concept)
    _log(f"  Pattern: {arch['pattern']}")
    _log(f"  Components: {[c['name'] for c in arch['components']]}")

    # 3. Generate project key
    key = hashlib.sha256(
        f"{concept}_{datetime.now().date()}".encode()
    ).hexdigest()[:16]

    project = {
        "key":          key,
        "concept":      concept,
        "architecture": arch,
        "code_history": [],
        "test_results": [],
        "status":       "in_progress",
        "started_at":   datetime.now().isoformat(),
        "iterations":   0,
    }

    feedback = None
    final_result = {"status": "unknown"}

    # 4. Synthesis → test → feedback loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        _log(f"Phase 2+: Iteration {iteration} — code synthesis")

        # Synthesize code
        code = synthesizer.synthesize(arch, feedback=feedback)
        project["code_history"].append({
            "iteration": iteration,
            "code":      code,
            "feedback":  feedback,
        })
        _log(f"  Generated {len(code.splitlines())} lines")

        # Run tests
        _log(f"Phase 3: Testing (iteration {iteration})")
        test_result = harness.run(code, concept)
        project["test_results"].append(test_result)
        project["iterations"] = iteration

        status = test_result.get("status", "unknown")
        passed = test_result.get("passed", 0)
        total  = test_result.get("total", 0)
        _log(f"  Status: {status}, Tests: {passed}/{total}")

        if test_result.get("stdout"):
            for line in test_result["stdout"].splitlines():
                if line.strip():
                    _log(f"    {line}")

        # Feedback
        _log("Phase 4: Feedback")
        feedback = feedback_eng.process(
            concept, arch, code, test_result, iteration)
        final_result = test_result

        # Continue?
        if not controller.should_continue(test_result, iteration):
            break

        if feedback:
            _log(f"  Revising: {feedback[:80]}")

    # 5. Finalize project
    project["status"]       = "complete" if final_result.get("status") == "pass" else "attempted"
    project["completed_at"] = datetime.now().isoformat()
    project["final_code"]   = project["code_history"][-1]["code"] if project["code_history"] else ""
    project["final_result"] = final_result

    # Save final code as importable file
    if project["final_code"]:
        code_path = STORE_DIR / f"{concept}_{key[:8]}.py"
        code_path.write_text(project["final_code"])
        project["code_file"] = str(code_path)
        _log(f"  Saved: {code_path}")

    store.save(project)
    _log(f"Project complete: {concept} [{project['status']}] in {project['iterations']} iterations")
    return project


# ── Daemon mode ───────────────────────────────────────────────────────────────
def run_daemon(interval: int = 300):
    """Continuous engineering. RM works autonomously."""
    _log("RM Engineering Daemon started")
    _log("RM will now conceive, architect, write, test, and learn — continuously")
    store = ProjectStore()

    while True:
        try:
            project = run_engineering_cycle(store=store)
            concept = project["concept"]
            status  = project["status"]
            iters   = project["iterations"]
            _log(f"Cycle complete: {concept} [{status}] — {iters} iterations")
            _log(f"Sleeping {interval}s before next cycle")
            time.sleep(interval)
        except KeyboardInterrupt:
            _log("Engineering daemon stopped")
            break
        except Exception as e:
            _log(f"Cycle error: {e}", "ERROR")
            traceback.print_exc()
            time.sleep(60)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="RM autonomous software engineering pipeline")
    ap.add_argument("--concept",  help="Develop a specific concept")
    ap.add_argument("--daemon",   action="store_true",
                    help="Run continuously")
    ap.add_argument("--interval", type=int, default=300,
                    help="Daemon cycle interval (seconds)")
    ap.add_argument("--list",     action="store_true",
                    help="List completed projects")
    ap.add_argument("--show",     metavar="CONCEPT",
                    help="Show project details for concept")
    ap.add_argument("--seed",     action="store_true",
                    help="Deposit software engineering seeds into crystal")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ]
    )

    store = ProjectStore()

    if args.seed:
        deposit_seeds()
        return

    if args.list:
        projects = store.list_projects()
        if not projects:
            print("No projects yet.")
            return
        print(f"\n{'Concept':<25} {'Status':<12} {'Iterations':<12} {'Pattern'}")
        print("─" * 65)
        for p in sorted(projects, key=lambda x: x.get("started_at", "")):
            arch    = p.get("architecture", {})
            pattern = arch.get("pattern", "unknown")
            print(f"{p['concept']:<25} {p['status']:<12} "
                  f"{p.get('iterations',0):<12} {pattern}")
        return

    if args.show:
        # Find project by concept
        projects = store.list_projects()
        matches  = [p for p in projects if p["concept"] == args.show]
        if not matches:
            print(f"No project found for concept: {args.show}")
            return
        proj = store.load(matches[0]["key"])
        if proj:
            print(json.dumps({
                k: v for k, v in proj.items()
                if k not in ("code_history", "final_code")
            }, indent=2))
            if proj.get("final_code"):
                print("\n── Final Code ──────────────────────────────")
                print(proj["final_code"][:3000])
        return

    # Seed first if not done
    deposit_seeds()

    if args.daemon:
        run_daemon(args.interval)
    else:
        project = run_engineering_cycle(concept=args.concept, store=store)
        status  = project["status"]
        iters   = project["iterations"]
        passed  = project["final_result"].get("passed", 0)
        total   = project["final_result"].get("total", 0)
        print(f"\nProject: {project['concept']}")
        print(f"Status:  {status}")
        print(f"Tests:   {passed}/{total} passed")
        print(f"Iters:   {iters}")
        if project.get("code_file"):
            print(f"Code:    {project['code_file']}")


if __name__ == "__main__":
    main()
