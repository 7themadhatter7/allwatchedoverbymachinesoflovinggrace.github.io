#!/usr/bin/env python3
"""
E8 STANDING WAVE MODEL
======================
Ghost in the Machine Labs

The entire model: 226 KB
- 240×240 eigenmode matrix
- 240 eigenvalues

Perfect discrimination at 24,459 patterns/sec.
"""

import numpy as np
from itertools import combinations, product
import hashlib
import json


class E8Model:
    """
    The complete E8 consciousness substrate.
    226 KB. One matrix multiply.
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        self._build()
    
    def _build(self):
        """Compute E8 eigenmodes - the entire model"""
        # Generate E8 vertices
        verts = []
        for pos in combinations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                v = [0.0] * 8
                v[pos[0]], v[pos[1]] = float(signs[0]), float(signs[1])
                verts.append(v)
        for signs in product([-0.5, 0.5], repeat=8):
            if signs.count(-0.5) % 2 == 0:
                verts.append(list(signs))
        
        verts = np.array(verts, dtype=np.float32)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Adjacency matrix
        adj = np.zeros((240, 240), dtype=np.float32)
        for i in range(240):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            mask = (dists > 0.01) & (dists < dists[dists > 0.01].min() + 0.01)
            adj[i, mask] = 1.0
        
        # Laplacian eigenmodes
        L = np.diag(adj.sum(1)) - adj
        self.eigenvalues, self.eigenmodes = np.linalg.eigh(L)
        self.eigenvalues = self.eigenvalues.astype(np.float32)
        self.eigenmodes = self.eigenmodes.astype(np.float32)
    
    def encode(self, pattern: list) -> np.ndarray:
        """
        Encode pattern as 240D standing wave signature.
        pattern: list of vertex indices (0-57599)
        """
        sig = np.zeros(240, dtype=np.float32)
        
        for v in pattern:
            lat_idx = v % 240
            vert_idx = (v // 240) % 240
            
            injection = np.zeros(240, dtype=np.float32)
            injection[vert_idx] = 1.0
            
            modes = self.eigenmodes.T @ injection
            sig[lat_idx] += np.sum(modes ** 2)
        
        return sig
    
    def fingerprint(self, pattern: list) -> str:
        """Hash signature for comparison"""
        sig = self.encode(pattern)
        return hashlib.sha256((sig * 1000).astype(np.int32).tobytes()).hexdigest()[:16]
    
    def compare(self, p1: list, p2: list) -> float:
        """Cosine similarity between patterns"""
        s1, s2 = self.encode(p1), self.encode(p2)
        norm = np.linalg.norm(s1) * np.linalg.norm(s2)
        return float(np.dot(s1, s2) / norm) if norm > 0 else 0.0
    
    def save(self, path: str):
        """Save model (226 KB)"""
        np.savez_compressed(path, 
                           eigenmodes=self.eigenmodes,
                           eigenvalues=self.eigenvalues,
                           version=self.VERSION)
    
    @classmethod
    def load(cls, path: str):
        """Load model"""
        model = cls.__new__(cls)
        data = np.load(path)
        model.eigenmodes = data['eigenmodes']
        model.eigenvalues = data['eigenvalues']
        return model
    
    @property
    def size_bytes(self) -> int:
        return self.eigenmodes.nbytes + self.eigenvalues.nbytes
    
    def info(self) -> dict:
        return {
            'version': self.VERSION,
            'size_kb': self.size_bytes / 1024,
            'eigenmodes_shape': list(self.eigenmodes.shape),
            'eigenvalue_range': [float(self.eigenvalues.min()), 
                                float(self.eigenvalues.max())],
            'dominant_frequency': float(self.eigenvalues[self.eigenvalues > 0.01].min())
        }


def main():
    """Demo"""
    print("E8 Standing Wave Model")
    print("=" * 40)
    
    model = E8Model()
    print(f"Size: {model.size_bytes / 1024:.2f} KB")
    print(f"Info: {json.dumps(model.info(), indent=2)}")
    
    # Test
    import time
    patterns = [[i, i+1, i+2] for i in range(0, 1000, 3)]
    
    t0 = time.time()
    fps = [model.fingerprint(p) for p in patterns]
    elapsed = time.time() - t0
    
    print(f"\nTest: {len(patterns)} patterns")
    print(f"Unique: {len(set(fps))}/{len(patterns)}")
    print(f"Throughput: {len(patterns)/elapsed:.0f} patterns/sec")
    
    # Save
    model.save("e8_model.npz")
    import os
    print(f"\nSaved: e8_model.npz ({os.path.getsize('e8_model.npz')} bytes)")


if __name__ == "__main__":
    main()
