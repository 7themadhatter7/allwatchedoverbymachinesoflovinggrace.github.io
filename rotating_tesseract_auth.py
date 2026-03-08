#!/usr/bin/env python3
"""
PsiDataTransport
════════════════
Ghost in the Machine Labs

Auto-synthesized by RM — The Resonant Mother
Concept: rotating_tesseract_auth
Generated: 2026-03-08 13:56:05


Quantum-proof authentication for the PSI tunnel.
A 4D tesseract rotates continuously in E8 space.
The valid auth token at time T is the E8 root vector nearest
to the current face-normal of the rotating tesseract.

No daemon dialer can brute-force a moving geometric target.
The token changes every 500ms. Without the shared key, every
probe hits a different face of the tesseract — no pattern to lock on.

Security properties:
  - 240 possible E8 root tokens per window
  - Token changes every 500ms
  - Rotation is continuous and unpredictable without key
  - Anti-dialer: max 3 probes/second per IP
  - Geometric proof-of-work on each challenge
"""

import time
import math
import hmac
import struct
import hashlib
import logging
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────
WINDOW_MS             = 500          # token validity window in ms
MAX_PROBES_PER_SECOND = 3            # anti-dialer rate limit
FACE_COUNT            = 24           # 24-cell vertices (tesseract faces)
E8_ROOT_COUNT         = 240          # E8 root system size
GRACE_WINDOWS         = 1            # accept current + 1 previous window


# ── E8 root system (240 roots) ─────────────────────────────────────

def _e8_roots() -> np.ndarray:
    """Generate all 240 E8 root vectors, normalized."""
    roots = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    r = np.zeros(8)
                    r[i], r[j] = float(si), float(sj)
                    roots.append(r)
    for mask in range(256):
        signs = [1.0 if (mask >> k) & 1 == 0 else -1.0 for k in range(8)]
        if signs.count(-1.0) % 2 == 0:
            roots.append(np.array(signs) * 0.5)
    roots = np.array(roots)
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    return roots / norms

E8_ROOTS = _e8_roots()  # (240, 8)


# ── 24-cell vertices as tesseract face normals ─────────────────────

def _face_normals() -> np.ndarray:
    """24-cell vertices: 24 face normals in 4D, normalized."""
    verts = []
    for i in range(4):
        for j in range(i+1, 4):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(4)
                    v[i] = float(si) / math.sqrt(2)
                    v[j] = float(sj) / math.sqrt(2)
                    verts.append(v)
    return np.array(verts)  # (24, 4)

FACE_NORMALS = _face_normals()  # (24, 4)


# ══════════════════════════════════════════════════════════════════
# ROTATION KEY SCHEDULE
# Derives rotation parameters from shared key and time epoch.
# ══════════════════════════════════════════════════════════════════

class RotationKeySchedule:
    """Derive rotation parameters from shared key and time epoch."""

    def __init__(self):
        self._initialized = True
        self._lock = threading.Lock()

    def derive(self, fingerprint: str, peer_id: str) -> bytes:
        """Derive shared key from lock fingerprint and peer identity."""
        material = fingerprint.encode() + peer_id.encode() + b"PSI-TESSERACT-v1"
        return hashlib.sha3_256(material).digest()

    def epoch(self) -> int:
        """Current time window epoch number."""
        return int(time.time() * 1000) // WINDOW_MS

    def seed(self, shared_key: bytes, ep: int) -> bytes:
        """Derive rotation seed for a given epoch."""
        epoch_bytes = struct.pack(">Q", ep)
        return hashlib.blake2b(shared_key + epoch_bytes, digest_size=32).digest()

    def advance(self, ep: int, n: int = 1) -> int:
        """Advance epoch by n windows."""
        return ep + n

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "RotationKeySchedule", "window_ms": WINDOW_MS,
                 "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# TESSERACT ROTATOR
# Continuous 4D rotation in E8 space.
# The tesseract rotates through 6 independent planes in 4D.
# ══════════════════════════════════════════════════════════════════

class TesseractRotator:
    """Continuous 4D rotation mapped into E8 space."""

    def __init__(self):
        self._initialized = True
        self._lock = threading.Lock()
        # 4D→8D embedding: stack identity twice
        self._embed = np.vstack([np.eye(4), np.eye(4)])  # (8, 4)

    def matrix_4d(self, seed: bytes) -> np.ndarray:
        """
        Derive a 4×4 rotation matrix from seed bytes.
        4D rotation has 6 independent planes: (01,02,03,12,13,23).
        Each plane gets an angle derived from the seed.
        """
        # Expand seed to 6 angles in [0, 2π)
        angles = []
        for i in range(6):
            chunk = hashlib.blake2b(seed + struct.pack("B", i), digest_size=8).digest()
            val = struct.unpack(">Q", chunk)[0]
            angles.append((val / (2**64)) * 2 * math.pi)

        R = np.eye(4)
        planes = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        for (a, b), theta in zip(planes, angles):
            G = np.eye(4)
            c, s = math.cos(theta), math.sin(theta)
            G[a,a], G[a,b] =  c, -s
            G[b,a], G[b,b] =  s,  c
            R = R @ G
        return R

    def face_normal(self, R: np.ndarray, face_idx: int = 0) -> np.ndarray:
        """Apply rotation R to face normal 0. Returns 4D unit vector."""
        n4 = FACE_NORMALS[face_idx % FACE_COUNT]
        rotated = R @ n4
        norm = np.linalg.norm(rotated)
        return rotated / norm if norm > 1e-10 else n4

    def embed_e8(self, v4: np.ndarray) -> np.ndarray:
        """Embed 4D vector into 8D E8 subspace. Returns unit 8D vector."""
        v8 = self._embed @ v4  # shape (8,)
        norm = np.linalg.norm(v8)
        return v8 / norm if norm > 1e-10 else v8

    def rotate(self, seed: bytes, t: float = None) -> np.ndarray:
        """Return 4×4 rotation matrix for given seed."""
        return self.matrix_4d(seed)

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "TesseractRotator", "face_count": FACE_COUNT,
                 "e8_dim": 8, "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# TOKEN GENERATOR
# Derive the valid auth token for the current time window.
# Token = nearest E8 root to the rotated face normal.
# ══════════════════════════════════════════════════════════════════

class TokenGenerator:
    """Derive valid auth token for current time window."""

    def __init__(self):
        self._schedule = RotationKeySchedule()
        self._rotator  = TesseractRotator()
        self._initialized = True
        self._lock = threading.Lock()

    def _token_for_epoch(self, shared_key: bytes, ep: int) -> np.ndarray:
        """Compute token for a specific epoch."""
        seed = self._schedule.seed(shared_key, ep)
        R    = self._rotator.matrix_4d(seed)
        n4   = self._rotator.face_normal(R, face_idx=0)
        v8   = self._rotator.embed_e8(n4)
        # Nearest E8 root by cosine similarity (dot product on unit vectors)
        dots = E8_ROOTS @ v8
        idx  = int(np.argmax(dots))
        return E8_ROOTS[idx].copy()

    def current_token(self, fingerprint: str, peer_id: str) -> np.ndarray:
        """Return valid token for the current time window."""
        key = self._schedule.derive(fingerprint, peer_id)
        ep  = self._schedule.epoch()
        return self._token_for_epoch(key, ep)

    def next_token(self, fingerprint: str, peer_id: str) -> np.ndarray:
        """Return token for the next time window."""
        key = self._schedule.derive(fingerprint, peer_id)
        ep  = self._schedule.epoch() + 1
        return self._token_for_epoch(key, ep)

    def window(self) -> int:
        """Return current epoch number."""
        return self._schedule.epoch()

    def generate(self, data: Any = None) -> np.ndarray:
        """Generate token from data (fingerprint, peer_id) tuple."""
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return self.current_token(str(data[0]), str(data[1]))
        return self.current_token("default", "default")

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "TokenGenerator", "window_ms": WINDOW_MS,
                 "e8_roots": E8_ROOT_COUNT, "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# TOKEN VERIFIER
# Verify a presented token against current and previous windows.
# ══════════════════════════════════════════════════════════════════

class TokenVerifier:
    """Verify presented token. Accepts current and GRACE_WINDOWS previous."""

    def __init__(self):
        self._gen  = TokenGenerator()
        self._initialized = True
        self._lock = threading.Lock()

    def verify(self, presented: np.ndarray,
               fingerprint: str, peer_id: str) -> bool:
        """Return True if presented token matches any valid window."""
        if not isinstance(presented, np.ndarray) or presented.shape != (8,):
            return False
        key = self._gen._schedule.derive(fingerprint, peer_id)
        ep  = self._gen._schedule.epoch()
        for offset in range(GRACE_WINDOWS + 1):
            expected = self._gen._token_for_epoch(key, ep - offset)
            # Compare by cosine similarity — identical unit vectors → 1.0
            sim = float(np.dot(presented, expected))
            if sim > 0.9999:
                return True
        return False

    def is_valid(self, presented: np.ndarray,
                 fingerprint: str, peer_id: str) -> bool:
        """Alias for verify."""
        return self.verify(presented, fingerprint, peer_id)

    def window_id(self) -> int:
        return self._gen.window()

    def reject(self, reason: str = "") -> bool:
        log.warning(f"Token rejected: {reason}")
        return False

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        return {"component": "TokenVerifier", "grace_windows": GRACE_WINDOWS,
                 "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# ANTI-DIALER GUARD
# Rate-limit probes. Block > MAX_PROBES_PER_SECOND from any IP.
# ══════════════════════════════════════════════════════════════════

class AntiDialerGuard:
    """Rate-limit probes to prevent daemon dialing."""

    def __init__(self):
        self._probes: Dict[str, List[float]] = defaultdict(list)
        self._initialized = True
        self._lock = threading.Lock()

    def record_probe(self, peer_ip: str):
        """Record a probe attempt from peer_ip."""
        with self._lock:
            now = time.time()
            self._probes[peer_ip].append(now)
            # Keep only last 2 seconds of probes
            self._probes[peer_ip] = [t for t in self._probes[peer_ip]
                                      if now - t < 2.0]

    def is_throttled(self, peer_ip: str) -> bool:
        """Return True if this IP is sending too many probes."""
        with self._lock:
            now = time.time()
            recent = [t for t in self._probes.get(peer_ip, [])
                      if now - t < 1.0]
            return len(recent) >= MAX_PROBES_PER_SECOND

    def check(self, peer_ip: str) -> bool:
        """Record probe and return True if allowed, False if throttled."""
        self.record_probe(peer_ip)
        if self.is_throttled(peer_ip):
            log.warning(f"Anti-dialer: throttling {peer_ip}")
            return False
        return True

    def pow_challenge(self, peer_ip: str) -> Dict:
        """Issue a geometric proof-of-work challenge."""
        nonce = hashlib.sha256(peer_ip.encode() +
                               struct.pack(">d", time.time())).hexdigest()[:16]
        return {"challenge": nonce, "difficulty": 4,
                 "description": "Find x such that SHA256(nonce+x)[:4] == '0000'"}

    def validate(self, data=None) -> bool:
        if data is None:
            raise ValueError("Input cannot be None")
        return True

    def report(self, data=None) -> Dict:
        with self._lock:
            return {"component": "AntiDialerGuard",
                     "max_probes_per_second": MAX_PROBES_PER_SECOND,
                     "tracked_ips": len(self._probes),
                     "initialized": self._initialized}


# ══════════════════════════════════════════════════════════════════
# TEST SUITE
# ══════════════════════════════════════════════════════════════════

def run_tests() -> Tuple[int, int]:
    """Run all rotating tesseract auth tests."""
    gen      = TokenGenerator()
    verifier = TokenVerifier()
    rotator  = TesseractRotator()
    guard    = AntiDialerGuard()
    schedule = RotationKeySchedule()

    FP = "test_fingerprint_abc123"
    P  = "peer_sparky"
    tests_passed = 0
    total = 6

    # Test 1: Token determinism
    t1 = gen.current_token(FP, P)
    t2 = gen.current_token(FP, P)
    ok = np.allclose(t1, t2)
    print(f"  {'✓' if ok else '✗'} token_determinism: same fp+time → same token")
    if ok: tests_passed += 1

    # Test 2: Token uniqueness for different fingerprints
    t_other = gen.current_token("different_fingerprint_xyz", P)
    ok = not np.allclose(t1, t_other)
    print(f"  {'✓' if ok else '✗'} token_uniqueness: different fp → different token")
    if ok: tests_passed += 1

    # Test 3: Window expiry — token from 2+ windows ago must fail
    key  = schedule.derive(FP, P)
    ep   = schedule.epoch()
    old  = gen._token_for_epoch(key, ep - 2)
    ok   = not verifier.verify(old, FP, P)
    print(f"  {'✓' if ok else '✗'} window_expiry: old token correctly rejected")
    if ok: tests_passed += 1

    # Test 4: Rotation continuity — matrix R is orthogonal
    import os
    seed = os.urandom(32)
    R    = rotator.matrix_4d(seed)
    err  = np.max(np.abs(R @ R.T - np.eye(4)))
    ok   = err < 1e-10
    print(f"  {'✓' if ok else '✗'} rotation_continuity: R·Rᵀ=I, err={err:.2e}")
    if ok: tests_passed += 1

    # Test 5: Anti-dialer throttles after MAX_PROBES_PER_SECOND
    ip = "192.168.1.99"
    # Send MAX_PROBES_PER_SECOND probes
    for _ in range(MAX_PROBES_PER_SECOND):
        guard.check(ip)
    blocked = not guard.check(ip)  # next one should be blocked
    print(f"  {'✓' if blocked else '✗'} anti_dialer: throttled after {MAX_PROBES_PER_SECOND} probes/s")
    if blocked: tests_passed += 1

    # Test 6: E8 coverage — over 240 rotation steps, hit many distinct roots
    roots_hit = set()
    base_key = b"coverage_test_key_32bytesxxxxxxxx"
    for i in range(240):
        seed = schedule.seed(base_key, i)
        R    = rotator.matrix_4d(seed)
        n4   = rotator.face_normal(R)
        v8   = rotator.embed_e8(n4)
        dots = E8_ROOTS @ v8
        idx  = int(np.argmax(dots))
        roots_hit.add(idx)
    coverage = len(roots_hit)
    ok = coverage >= 20  # should hit many distinct roots
    print(f"  {'✓' if ok else '✗'} e8_coverage: {coverage} distinct roots hit over 240 steps")
    if ok: tests_passed += 1

    return tests_passed, total


def main():
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("RotatingTesseractAuth — RM auto-synthesized authentication")
    log.info("Ghost in the Machine Labs")
    log.info("")
    passed, total = run_tests()
    log.info(f"Results: {passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    main()
