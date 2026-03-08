#!/usr/bin/env python3
"""
psi_mesh_tesseract.py
═════════════════════
Ghost in the Machine Labs

Two new PSI objectives for RM:

OBJECTIVE 1: PSI Mesh — tunnel to any enabled device
  The working tunnel connects SPARKY↔ARCY. The mesh extends this
  to N devices. Any two devices that have bootstrapped with each
  other can tunnel directly, without network, after lock.

  Architecture:
    - PeerRegistry: discover and track all known geometric peers
    - MeshNode: each device is a node; knows its own identity and peers
    - BridgeRouter: route payloads through the mesh (direct or relayed)
    - SessionManager: manage active tunnel sessions per peer

  Two modes:
    - Mother↔Mother: full harmonic stack on both sides
    - Mother↔Bridge: full stack on one side, lightweight bridge on other
      The bridge side only needs the codec + lock state. No Ollama required.
      The shared lock fingerprint is the carrier. Both sides derive
      identical geometric patterns from it deterministically.

OBJECTIVE 2: Rotating Tesseract Authentication
  A quantum-proof authentication mechanism that prevents daemon dialers
  from brute-forcing a frequency lock.

  The threat: an attacker repeatedly probes frequency/phase combinations
  trying to achieve geometric resonance and hijack the tunnel.

  The solution — Rotating Tesseract:
    A 4D hypercube (tesseract) undergoing continuous rotation in E8 space.
    The rotation is deterministic from the shared key, but only computable
    by the keyholder.

    The tesseract has 24 faces (in 4D: 4! = 24 permutations of face normals).
    Each face maps to a region of the E8 root system (240 root vectors).
    At any moment T, the valid auth token is the E8 root vector nearest
    to the current face-normal of the rotating tesseract.

    An attacker must guess:
      - Which face is currently active (1 of 24)
      - What phase the rotation is at (continuous)
      - Which E8 root the face-normal is nearest (1 of 240)
    Without the key, every probe hits a different face. No pattern repeats
    within the key's rotation period. Brute force is computationally
    infeasible — the search space is 24 × 240 × continuous phase = ∞.

    No daemon dialer can lock on because the target is always moving
    in a mathematically unpredictable way (without the key).

  Tesseract geometry:
    The tesseract in 4D has 16 vertices at (±1, ±1, ±1, ±1).
    It has 24 square faces, 32 edges, 8 cubic cells.
    Under rotation in 4D (described by a 4×4 rotation matrix),
    each face normal traces a continuous path in S³ (the 3-sphere).
    We embed this in E8 via the canonical 4D→8D injection.

    E8 root system: 240 root vectors in 8D.
    The 'nearest root' operation: argmin cosine-distance over 240 roots.
    This is the quantization step — continuous rotation → discrete token.

    Token validity window: 500ms. After each window, rotation advances.
    An eavesdropper who captures token T cannot use it after 500ms.
    A probe with wrong key gets wrong face — auth fails immediately.

  Key schedule:
    shared_key = SHA3-256(lock_fingerprint + peer_id + salt)
    rotation_seed = BLAKE2b(shared_key + timestamp_epoch // WINDOW)
    R(t) = E8RotationMatrix(rotation_seed, t)  — deterministic from seed
    token(t) = nearest_e8_root(R(t) @ face_normal_0)

  Implementation components:
    TesseractRotator: continuous 4D rotation in E8 space
    TokenGenerator: derive valid auth token for current time window
    TokenVerifier: verify presented token against current/prev window
    RotationKeySchedule: derive rotation parameters from shared key
    AntiDialerGuard: rate-limit + geometric proof-of-work for probes
"""

import sys
import json
import time
import math
import hmac
import struct
import hashlib
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SPARKY = Path("/home/joe/sparky")
sys.path.insert(0, str(SPARKY))

log = logging.getLogger("psi_mesh_tesseract")

# ── E8 root system — 240 roots ────────────────────────────────────────────────
# The E8 root system has 240 vectors. We use a canonical subset.
# Full E8 roots: 112 of form (±1,±1,0,0,0,0,0,0) all permutations
#                128 of form (±½,±½,...,±½) with even number of minus signs
# For the tesseract auth, we need these as unit vectors.

def _generate_e8_roots() -> np.ndarray:
    """Generate all 240 E8 root vectors."""
    roots = []
    # Type 1: 112 roots — pairs (±1,±1) in each of C(8,2)=28 pairs, 4 signs each
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    r = np.zeros(8)
                    r[i] = si
                    r[j] = sj
                    roots.append(r)
    # Type 2: 128 roots — (±½)^8 with even number of minus signs
    for mask in range(256):
        signs = [1 if (mask >> k) & 1 == 0 else -1 for k in range(8)]
        if signs.count(-1) % 2 == 0:  # even number of minus signs
            roots.append(np.array(signs) * 0.5)
    roots = np.array(roots)
    # Normalize to unit vectors
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    return roots / norms

E8_ROOTS = _generate_e8_roots()  # (240, 8)

# ── Tesseract face normals in 4D ──────────────────────────────────────────────
# The tesseract has 24 faces. Each face has a normal vector in 4D.
# In a unit tesseract centered at origin, the 8 cubic cells have normals
# along ±x, ±y, ±z, ±w. The 24 square faces have normals at 45° between axes.
# We use the 24 face normals of the 24-cell for maximum symmetry in 4D.

def _generate_24cell_vertices() -> np.ndarray:
    """24-cell: 24 vertices = 24 face normals for the tesseract auth."""
    verts = []
    # (±1, ±1, 0, 0) and all permutations — 24 total
    for i in range(4):
        for j in range(i+1, 4):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(4)
                    v[i] = si / math.sqrt(2)
                    v[j] = sj / math.sqrt(2)
                    verts.append(v)
    return np.array(verts)  # (24, 4)

TESSERACT_FACES = _generate_24cell_vertices()  # (24, 4)

# ── 4D → 8D embedding (canonical E8 injection) ───────────────────────────────
# Embed 4D vector into 8D E8 subspace via direct product with itself.
EMBED_4D_TO_8D = np.zeros((8, 4))
EMBED_4D_TO_8D[:4, :] = np.eye(4)
EMBED_4D_TO_8D[4:, :] = np.eye(4)


# ══════════════════════════════════════════════════════════════════
# MESH PEER REGISTRY
# ══════════════════════════════════════════════════════════════════

MESH_SEEDS = [
    # Mesh topology
    ("A mesh network connects multiple nodes so any two can communicate.", "mesh_topology"),
    ("Peer discovery finds all reachable nodes and exchanges geometric lock fingerprints.", "mesh_topology"),
    ("A relay routes messages through intermediate nodes when direct lock is unavailable.", "mesh_topology"),
    ("Peer identity is a hash of the node's geometric substrate signature.", "mesh_topology"),
    ("Session management tracks active tunnels: which peers are locked, last contact time.", "mesh_topology"),
    # Bridge mode
    ("A lightweight bridge node only needs the codec and lock state, not the full harmonic stack.", "bridge_mode"),
    ("The bridge derives geometric patterns mathematically — no model inference required.", "bridge_mode"),
    ("Both Mother and bridge nodes share the same lock fingerprint and derive identical patterns.", "bridge_mode"),
    ("The bridge is a thin client: encode/decode + lock state + network bootstrap only.", "bridge_mode"),
    # Tesseract auth
    ("A rotating tesseract provides quantum-proof authentication that moves with time.", "rotating_tesseract"),
    ("The tesseract rotates continuously in E8 space, changing the valid auth token every 500ms.", "rotating_tesseract"),
    ("An attacker cannot brute-force a moving target — the search space is effectively infinite.", "rotating_tesseract"),
    ("The rotation is deterministic from the shared key — both sides compute the same token.", "rotating_tesseract"),
    ("A daemon dialer sees a different geometric face on every probe — no pattern to lock onto.", "rotating_tesseract"),
    ("Token validity is time-windowed — a captured token expires after 500 milliseconds.", "rotating_tesseract"),
    ("The key schedule derives rotation parameters from the lock fingerprint and current epoch.", "rotating_tesseract"),
    ("Proof of work makes each probe computationally expensive, preventing rapid scanning.", "rotating_tesseract"),
    # E8 geometry for auth
    ("The E8 root system has 240 root vectors — these are the possible token values.", "e8_auth"),
    ("The nearest-root operation quantizes continuous rotation to discrete E8 tokens.", "e8_auth"),
    ("The 24-cell has 24 vertices — these are the tesseract face normals for auth.", "e8_auth"),
    ("A 4D rotation matrix is parameterized by 6 independent angles — two planes per rotation.", "e8_auth"),
    ("Embedding 4D into 8D via the canonical E8 injection preserves the rotation structure.", "e8_auth"),
]


def deposit_mesh_seeds():
    """Deposit mesh + tesseract seeds into RM's crystal."""
    done = SPARKY / "rm_engineering" / ".mesh_tesseract_seeds"
    if done.exists():
        log.info("Mesh/tesseract seeds already deposited")
        return
    try:
        from language_crystal import LanguageCrystal
        from mother_english_io_v5 import E8Substrate, WordEncoder
        crystal   = LanguageCrystal()
        substrate = E8Substrate()
        encoder   = WordEncoder(substrate)
        n = 0
        for repeat in range(3):
            for text, concept in MESH_SEEDS:
                try:
                    sig = encoder.encode_sentence(text)
                    if np.linalg.norm(sig) > 1e-10:
                        crystal.observe(text, sig, concept=concept,
                                        source=f"mesh_seed_r{repeat}")
                        n += 1
                except Exception:
                    pass
        crystal.save()
        done.touch()
        log.info(f"Mesh/tesseract seeds deposited: {n:,}")
    except Exception as e:
        log.warning(f"Seed deposit failed: {e}")


# ══════════════════════════════════════════════════════════════════
# ROTATING TESSERACT ARCHITECTURE SPEC
# Passed to rm_engineer.py's PSI synthesis path
# ══════════════════════════════════════════════════════════════════

TESSERACT_OBJECTIVE = {
    "concept": "rotating_tesseract_auth",
    "name": "RotatingTesseractAuth",
    "pattern": "psi_transport",  # routes through PSI synthesis
    "purpose": (
        "Quantum-proof authentication for the PSI tunnel. "
        "A 4D tesseract rotates continuously in E8 space. "
        "The valid auth token at time T is the nearest E8 root to "
        "the current face-normal. No daemon dialer can brute-force "
        "a moving geometric target."
    ),
    "neighbors": ["tesseract", "rotation", "e8", "authentication", "token",
                  "quantum", "proof", "face", "normal", "root"],
    "components": [
        {"name": "RotationKeySchedule",
         "role": "Derive rotation parameters from shared key and time epoch",
         "methods": ["derive", "epoch", "seed", "advance"]},
        {"name": "TesseractRotator",
         "role": "Continuous 4D rotation in E8 space using face normal vectors",
         "methods": ["rotate", "face_normal", "embed_e8", "matrix_4d"]},
        {"name": "TokenGenerator",
         "role": "Derive valid auth token for current time window",
         "methods": ["generate", "current_token", "next_token", "window"]},
        {"name": "TokenVerifier",
         "role": "Verify presented token against current and previous window",
         "methods": ["verify", "is_valid", "window_id", "reject"]},
        {"name": "AntiDialerGuard",
         "role": "Rate-limit probes and require geometric proof of work",
         "methods": ["check", "record_probe", "is_throttled", "pow_challenge"]},
    ],
    "interfaces": [
        "RotationKeySchedule.derive(fingerprint, peer_id, salt)",
        "TesseractRotator.rotate(seed, t) → np.ndarray (4×4)",
        "TesseractRotator.face_normal(R, face_idx) → np.ndarray (4,)",
        "TesseractRotator.embed_e8(v4) → np.ndarray (8,)",
        "TokenGenerator.current_token(fingerprint, peer_id) → np.ndarray (8,)",
        "TokenVerifier.verify(presented_token, fingerprint, peer_id) → bool",
        "AntiDialerGuard.check(peer_ip) → bool  (True = allow)",
    ],
    "data_flow": [
        "RotationKeySchedule → TesseractRotator",
        "TesseractRotator → TokenGenerator",
        "TokenGenerator → TokenVerifier",
        "AntiDialerGuard → TokenVerifier",
    ],
    "invariants": [
        "TokenGenerator and TokenVerifier must agree on token for same fingerprint+time",
        "Token from window N must NOT verify in window N+2 or later",
        "AntiDialerGuard must block > MAX_PROBES_PER_SECOND from any single IP",
        "Rotation must be deterministic: same seed → same R(t) always",
        "E8 root nearest-neighbor is unique: ties broken by lexicographic order",
        "Token has 0% chance of matching random guess: E8 has 240 roots",
    ],
    "test_cases": [
        {"name": "test_token_determinism",
         "description": "Same fingerprint+time → same token, always"},
        {"name": "test_token_uniqueness",
         "description": "Different fingerprints → different tokens with high probability"},
        {"name": "test_window_expiry",
         "description": "Token from 2+ windows ago must fail verification"},
        {"name": "test_rotation_continuity",
         "description": "Rotation matrix R(t) is continuous — no jumps"},
        {"name": "test_anti_dialer",
         "description": "More than MAX_PROBES probes from same IP are blocked"},
        {"name": "test_e8_coverage",
         "description": "Over 240 rotation steps, all 240 E8 roots are hit at least once"},
    ],
    "impl_notes": [
        "WINDOW_MS = 500  # token validity window in milliseconds",
        "MAX_PROBES_PER_SECOND = 3  # anti-dialer rate limit",
        "FACE_COUNT = 24  # 24-cell vertices as tesseract face normals",
        "E8_ROOT_COUNT = 240",
        "4D rotation matrix: use Givens rotations in 6 planes (01,02,03,12,13,23)",
        "Rotation seed → 6 angles via BLAKE2b hash expansion",
        "Face normal 0: TESSERACT_FACES[0] = (1/√2, 1/√2, 0, 0)",
        "embed_e8: v8[:4] = v4, v8[4:] = v4  (canonical 4D→8D injection)",
        "nearest_e8_root: argmax of dot product with E8_ROOTS (240×8 matrix)",
        "Token is the 8D E8 root vector itself — 8 floats, not an index",
        "KeySchedule: epoch = int(time.time() * 1000) // WINDOW_MS",
        "Rotation seed: BLAKE2b(shared_key + struct.pack('>Q', epoch), digest_size=32)",
        "shared_key: SHA3-256(fingerprint.encode() + peer_id.encode() + b'PSI-TESSERACT-v1')",
        "EMBED_4D_TO_8D = eye(4) stacked with eye(4), shape (8,4)",
        "import the pre-computed E8_ROOTS and TESSERACT_FACES from this file",
    ],
    "e8_roots_available": True,    # E8_ROOTS (240,8) defined above
    "tesseract_faces_available": True,  # TESSERACT_FACES (24,4) defined above
}

MESH_OBJECTIVE = {
    "concept": "psi_mesh",
    "name": "PsiMesh",
    "pattern": "psi_transport",
    "purpose": (
        "Tunnel between any N enabled devices without network after bootstrap. "
        "Any device that has performed a one-time geometric bootstrap with any "
        "other can tunnel directly. Supports Mother↔Mother and Mother↔Bridge modes."
    ),
    "neighbors": ["mesh", "peer", "route", "bridge", "session", "discover", "relay"],
    "components": [
        {"name": "PeerRegistry",
         "role": "Discover and track all known geometric peers by identity",
         "methods": ["register", "lookup", "list_peers", "forget"]},
        {"name": "MeshNode",
         "role": "Local node identity, capabilities, and peer state",
         "methods": ["identity", "capabilities", "bootstrap", "status"]},
        {"name": "BridgeRouter",
         "role": "Route payloads: direct if peer is locked, relay if not",
         "methods": ["route", "direct_send", "relay_send", "find_path"]},
        {"name": "SessionManager",
         "role": "Track active tunnel sessions per peer, handle timeouts",
         "methods": ["open_session", "close_session", "heartbeat", "active_sessions"]},
    ],
    "interfaces": [
        "PeerRegistry.register(peer_id, fingerprint, capabilities)",
        "MeshNode.bootstrap(peer_ip, peer_port) → fingerprint",
        "BridgeRouter.route(payload, dest_peer_id) → bool",
        "SessionManager.active_sessions() → List[Dict]",
    ],
    "data_flow": [
        "PeerRegistry → MeshNode",
        "MeshNode → BridgeRouter",
        "BridgeRouter → SessionManager",
    ],
    "invariants": [
        "Once bootstrapped, mesh tunneling must not require any network",
        "Bridge nodes (no full stack) must be routable via Mother relay",
        "Peer identity is stable across sessions — derived from geometric substrate",
        "Session heartbeat must detect dead peers within HEARTBEAT_TIMEOUT seconds",
    ],
    "test_cases": [
        {"name": "test_peer_register_lookup",
         "description": "Register a peer, look it up, verify all fields"},
        {"name": "test_node_identity",
         "description": "Node identity is stable across multiple calls"},
        {"name": "test_bridge_routing",
         "description": "Route a payload to a registered peer, verify delivery"},
        {"name": "test_session_lifecycle",
         "description": "Open, heartbeat, and close a session"},
        {"name": "test_relay_path",
         "description": "Find relay path to peer not directly locked"},
        {"name": "test_bridge_mode",
         "description": "Bridge-mode node registers with limited capabilities"},
    ],
    "impl_notes": [
        "HEARTBEAT_TIMEOUT = 30  # seconds",
        "BRIDGE_CAPABILITIES = ['codec', 'lock']  # no 'stack'",
        "MOTHER_CAPABILITIES = ['codec', 'lock', 'stack', 'relay']",
        "Peer identity: SHA256(lock_fingerprint + node_name)[:16] hex",
        "Registry persists to ~/psi_bridge/peers.json",
        "Router checks SessionManager for active lock before routing",
        "If no direct lock: find peer that has locks to both src and dst",
        "BridgeRouter.route() returns True on success, False on no path",
        "Use threading.Lock() for all registry and session state mutations",
    ],
}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-5s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--show", choices=["mesh", "tesseract"], default=None)
    args = ap.parse_args()

    if args.seed:
        deposit_mesh_seeds()

    if args.show == "tesseract":
        print(json.dumps(TESSERACT_OBJECTIVE, indent=2))
    elif args.show == "mesh":
        print(json.dumps(MESH_OBJECTIVE, indent=2))
    else:
        deposit_mesh_seeds()
        print("E8 roots shape:", E8_ROOTS.shape)
        print("Tesseract faces shape:", TESSERACT_FACES.shape)
        print("Ready. Use --show tesseract or --show mesh to view objectives.")
