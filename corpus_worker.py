"""
corpus_worker.py — Pickle-safe encode worker for corpus_trainer parallel processing.

Must be a standalone importable module (not a numeric-prefixed file)
so that ProcessPoolExecutor can pickle and resolve the function by name.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
from itertools import combinations, product


class _E8:
    """Inline E8 lattice — reproduced here so workers are self-contained."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @staticmethod
    def _build():
        verts = []
        for pos in combinations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                v = [0.0] * 8
                v[pos[0]], v[pos[1]] = float(signs[0]), float(signs[1])
                verts.append(v)
        for signs in product([-0.5, 0.5], repeat=8):
            if sum(1 for s in signs if s > 0) % 2 == 0:
                verts.append(list(signs))
        return np.array(verts, dtype=np.float32)


def _hash_word(word: str) -> np.ndarray:
    """Map a word to an E8 vertex deterministically."""
    import hashlib
    h = int(hashlib.md5(word.lower().encode()).hexdigest(), 16)
    eigenmodes = _E8.get()
    return eigenmodes[h % len(eigenmodes)].copy()


def encode_text(text: str, eigenmodes: np.ndarray = None) -> np.ndarray:
    """Encode a sentence as weighted sum of E8 vertex embeddings."""
    if eigenmodes is None:
        eigenmodes = _E8.get()
    words = text.lower().split()
    if not words:
        return np.zeros(8, dtype=np.float32)
    sig = np.zeros(8, dtype=np.float32)
    for i, word in enumerate(words):
        weight = 1.0 / (i + 1) ** 0.5
        sig += _hash_word(word) * weight
    norm = np.linalg.norm(sig)
    if norm > 1e-10:
        sig /= norm
    return sig.astype(np.float32)


def encode_worker(args):
    """
    Encode a single (text, concept, tag) tuple.
    Module-level, importable by name — safe for ProcessPoolExecutor pickling.
    Returns (text, sig_bytes, concept, tag) or None.
    """
    text, concept, tag = args
    try:
        sig = encode_text(text)
        if np.linalg.norm(sig) < 1e-10:
            return None
        return (text, sig.tobytes(), concept, tag)
    except Exception:
        return None
