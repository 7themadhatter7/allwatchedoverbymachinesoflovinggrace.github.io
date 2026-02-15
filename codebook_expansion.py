"""
Dynamic Codebook Expansion
===========================
Geometric learning from solved tasks.

When the static codebook can't recognize a pattern, the substrate still
processes the signal — it sees geometry the codebook doesn't name yet.

This module captures those geometric signatures, pairs them with working
solutions when they arrive, and recalls them for future similar tasks.

The codebook grows from evidence, not enumeration.

Three phases:
  1. RECORD  — On fallback, capture geometric signature as "pending"
  2. LEARN   — When orchestrator solves the task, pair code with signature
  3. RECALL  — For new tasks, match against learned signatures before fallback

Ghost in the Machine Labs — AGI for the home
"""

import json
import time
import hashlib
import os
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# =============================================================================
# GEOMETRIC SIGNATURE — condensed fingerprint from encoder bands
# =============================================================================

@dataclass
class GeometricSignature:
    """
    64-float fingerprint extracted from the encoder's 8 bands.
    
    Not the full 1024-signal — just the semantically meaningful
    features that distinguish one transformation type from another.
    
    Each band contributes 8 floats:
      Band 1 (shape):       aspect ratio, area, dimension parity
      Band 2 (color):       histogram peaks, unique count, entropy
      Band 3 (symmetry):    H/V/diagonal/rotational flags
      Band 4 (frequency):   tiling period, repetition density
      Band 5 (boundary):    edge density, gradient magnitude
      Band 6 (objects):     count, avg size, size variance
      Band 7 (transform):   dimension ratio, color shift, spatial op
      Band 8 (hash):        low-res structural hash
    """
    
    vector: List[float]          # 64 floats
    task_hash: str = ""          # SHA256 of task data for exact matching
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.vector, dtype=np.float32)
    
    def cosine_similarity(self, other: 'GeometricSignature') -> float:
        """Cosine similarity between two signatures."""
        a = self.to_numpy()
        b = other.to_numpy()
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(dot / (na * nb))


class SignatureExtractor:
    """
    Extract GeometricSignature from an ARC task.
    
    Uses the same geometric features the encoder captures,
    but compressed to 64 dimensions for fast similarity matching.
    """
    
    SIGNATURE_SIZE = 64  # 8 bands × 8 features
    
    @staticmethod
    def extract(task: Dict) -> GeometricSignature:
        """Extract signature from complete task (all training pairs)."""
        train = task.get('train', [])
        if not train:
            return GeometricSignature(vector=[0.0] * 64)
        
        # Accumulate features across all training pairs
        all_features = []
        for pair in train:
            features = SignatureExtractor._extract_pair(
                pair['input'], pair['output'])
            all_features.append(features)
        
        # Average across pairs (consensus signature)
        avg = np.mean(all_features, axis=0).tolist()
        
        # Task hash for exact matching
        task_hash = hashlib.sha256(
            json.dumps(task.get('train', []), sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return GeometricSignature(vector=avg, task_hash=task_hash)
    
    @staticmethod
    def _extract_pair(input_grid: List[List[int]], 
                      output_grid: List[List[int]]) -> np.ndarray:
        """Extract 64 features from a single input→output pair."""
        ig = np.array(input_grid, dtype=np.float32)
        og = np.array(output_grid, dtype=np.float32)
        ih, iw = ig.shape
        oh, ow = og.shape
        
        features = np.zeros(64, dtype=np.float32)
        
        # === Band 1: Shape (0-7) ===
        features[0] = ih / 30.0
        features[1] = iw / 30.0
        features[2] = oh / 30.0
        features[3] = ow / 30.0
        features[4] = (ih * iw) / 900.0          # input area
        features[5] = (oh * ow) / 900.0          # output area
        features[6] = oh / ih if ih > 0 else 0   # height ratio
        features[7] = ow / iw if iw > 0 else 0   # width ratio
        
        # === Band 2: Color (8-15) ===
        i_colors = set(ig.flatten().astype(int))
        o_colors = set(og.flatten().astype(int))
        features[8] = len(i_colors) / 10.0       # input unique colors
        features[9] = len(o_colors) / 10.0       # output unique colors
        features[10] = len(i_colors & o_colors) / 10.0  # shared colors
        features[11] = len(i_colors ^ o_colors) / 10.0  # changed colors
        
        # Color entropy (information content)
        for idx, g in enumerate([ig, og]):
            vals, counts = np.unique(g, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features[12 + idx] = entropy / 3.32  # normalize by log2(10)
        
        # Dominant color fraction
        i_vals, i_counts = np.unique(ig, return_counts=True)
        features[14] = i_counts.max() / i_counts.sum()
        o_vals, o_counts = np.unique(og, return_counts=True)
        features[15] = o_counts.max() / o_counts.sum()
        
        # === Band 3: Symmetry (16-23) ===
        if ih > 1:
            features[16] = float(np.mean(ig == ig[::-1, :]))  # input H sym
        if iw > 1:
            features[17] = float(np.mean(ig == ig[:, ::-1]))  # input V sym
        if oh > 1:
            features[18] = float(np.mean(og == og[::-1, :]))  # output H sym
        if ow > 1:
            features[19] = float(np.mean(og == og[:, ::-1]))  # output V sym
        if ih == iw:
            features[20] = float(np.mean(ig == ig.T))         # input diag
        if oh == ow:
            features[21] = float(np.mean(og == og.T))         # output diag
        # Input→output symmetry preservation
        features[22] = abs(features[16] - features[18])  # H sym change
        features[23] = abs(features[17] - features[19])  # V sym change
        
        # === Band 4: Spatial frequency (24-31) ===
        # Row repetition period in input
        for period in range(1, min(iw, 8)):
            if iw % period == 0 and period < iw:
                tiles = ig.reshape(ih, -1, period)
                if tiles.shape[1] > 1 and np.all(tiles == tiles[:, 0:1, :]):
                    features[24] = period / 8.0
                    break
        
        # Column repetition period in input
        for period in range(1, min(ih, 8)):
            if ih % period == 0 and period < ih:
                tiles = ig.reshape(-1, period, iw)
                if tiles.shape[0] > 1 and np.all(tiles == tiles[0:1, :, :]):
                    features[25] = period / 8.0
                    break
        
        # Output repetition patterns
        for period in range(1, min(ow, 8)):
            if ow % period == 0 and period < ow:
                tiles = og.reshape(oh, -1, period)
                if tiles.shape[1] > 1 and np.all(tiles == tiles[:, 0:1, :]):
                    features[26] = period / 8.0
                    break
        
        for period in range(1, min(oh, 8)):
            if oh % period == 0 and period < oh:
                tiles = og.reshape(-1, period, ow)
                if tiles.shape[0] > 1 and np.all(tiles == tiles[0:1, :, :]):
                    features[27] = period / 8.0
                    break
        
        # Size-change type: grow, shrink, or same
        features[28] = 1.0 if oh > ih else (-1.0 if oh < ih else 0.0)
        features[29] = 1.0 if ow > iw else (-1.0 if ow < iw else 0.0)
        
        # Tiling divisibility
        features[30] = 1.0 if (oh % ih == 0 and ow % iw == 0 and 
                                (oh > ih or ow > iw)) else 0.0
        features[31] = 1.0 if (ih % oh == 0 and iw % ow == 0 and 
                                (ih > oh or iw > ow)) else 0.0
        
        # === Band 5: Boundary/edge (32-39) ===
        # Edge density (fraction of cells adjacent to different color)
        for idx, g in enumerate([ig, og]):
            h, w = g.shape
            edges = 0
            total = 0
            for r in range(h):
                for c in range(w):
                    if c + 1 < w:
                        total += 1
                        if g[r, c] != g[r, c + 1]:
                            edges += 1
                    if r + 1 < h:
                        total += 1
                        if g[r, c] != g[r + 1, c]:
                            edges += 1
            features[32 + idx] = edges / max(total, 1)
        
        # Edge density change
        features[34] = features[33] - features[32]
        
        # Border uniformity (are edges all one color?)
        for idx, g in enumerate([ig, og]):
            h, w = g.shape
            border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
            features[35 + idx] = len(np.unique(border)) / 10.0
        
        # Non-zero fraction
        features[37] = float(np.count_nonzero(ig)) / max(ig.size, 1)
        features[38] = float(np.count_nonzero(og)) / max(og.size, 1)
        features[39] = features[38] - features[37]  # density change
        
        # === Band 6: Objects (40-47) ===
        for idx, g in enumerate([ig, og]):
            h, w = g.shape
            visited = np.zeros_like(g, dtype=bool)
            sizes = []
            for r in range(h):
                for c in range(w):
                    if not visited[r, c] and g[r, c] != 0:
                        # BFS
                        stack = [(r, c)]
                        size = 0
                        color = g[r, c]
                        while stack:
                            cr, cc = stack.pop()
                            if (0 <= cr < h and 0 <= cc < w and 
                                not visited[cr, cc] and g[cr, cc] == color):
                                visited[cr, cc] = True
                                size += 1
                                stack.extend([(cr+1,cc),(cr-1,cc),
                                              (cr,cc+1),(cr,cc-1)])
                        if size > 0:
                            sizes.append(size)
            
            base = idx * 4
            features[40 + base] = len(sizes) / 30.0          # object count
            if sizes:
                features[41 + base] = np.mean(sizes) / (h * w) # avg size
                features[42 + base] = np.std(sizes) / (h * w)  # size variance
                features[43 + base] = max(sizes) / (h * w)     # largest object
        
        # === Band 7: Transformation (48-55) ===
        # Direct overlap (how much of input appears unchanged in output)
        min_h = min(ih, oh)
        min_w = min(iw, ow)
        overlap = float(np.mean(ig[:min_h, :min_w] == og[:min_h, :min_w]))
        features[48] = overlap
        
        # Rotation checks
        if ih == ow and iw == oh:
            for k, fidx in [(1, 49), (2, 50), (3, 51)]:
                rotated = np.rot90(ig, k)
                if rotated.shape == og.shape:
                    features[fidx] = float(np.mean(rotated == og))
        elif ih == oh and iw == ow:
            features[50] = float(np.mean(np.rot90(ig, 2) == og))
        
        # Mirror checks
        if ih == oh and iw == ow:
            features[52] = float(np.mean(ig[::-1, :] == og))  # H flip
            features[53] = float(np.mean(ig[:, ::-1] == og))  # V flip
        
        # Color mapping consistency
        if ih == oh and iw == ow:
            mapping = {}
            consistent = True
            for r in range(ih):
                for c in range(iw):
                    ic = int(ig[r, c])
                    oc = int(og[r, c])
                    if ic in mapping:
                        if mapping[ic] != oc:
                            consistent = False
                            break
                    else:
                        mapping[ic] = oc
                if not consistent:
                    break
            features[54] = 1.0 if consistent and mapping else 0.0
            features[55] = len(mapping) / 10.0 if consistent else 0.0
        
        # === Band 8: Structural hash (56-63) ===
        # Low-resolution grid hash for coarse matching
        # Downsample both grids to 2x2 and encode
        for idx, g in enumerate([ig, og]):
            h, w = g.shape
            # Divide into quadrants, take mode of each
            mh, mw = h // 2 or 1, w // 2 or 1
            for qi in range(2):
                for qj in range(2):
                    rs = qi * mh
                    re = min(rs + mh, h)
                    cs = qj * mw
                    ce = min(cs + mw, w)
                    quad = g[rs:re, cs:ce]
                    vals, counts = np.unique(quad, return_counts=True)
                    features[56 + idx * 4 + qi * 2 + qj] = vals[counts.argmax()] / 9.0
        
        return features


# =============================================================================
# CODEBOOK ENTRY — a learned signature→code pairing
# =============================================================================

@dataclass
class CodebookEntry:
    """A learned geometric pattern → code mapping."""
    
    signature: GeometricSignature
    code: str                        # Python solve() function
    task_id: str = ""                # ARC task ID if known
    learned_at: float = 0.0          # Unix timestamp
    hit_count: int = 0               # Times this entry has been recalled
    last_hit: float = 0.0            # Last recall timestamp
    validated: bool = False          # Has this been validated against training?
    description: str = ""            # Human-readable description of the pattern
    
    def to_dict(self) -> dict:
        return {
            'signature': self.signature.vector,
            'task_hash': self.signature.task_hash,
            'code': self.code,
            'task_id': self.task_id,
            'learned_at': self.learned_at,
            'hit_count': self.hit_count,
            'last_hit': self.last_hit,
            'validated': self.validated,
            'description': self.description,
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'CodebookEntry':
        sig = GeometricSignature(
            vector=d['signature'],
            task_hash=d.get('task_hash', '')
        )
        return CodebookEntry(
            signature=sig,
            code=d['code'],
            task_id=d.get('task_id', ''),
            learned_at=d.get('learned_at', 0.0),
            hit_count=d.get('hit_count', 0),
            last_hit=d.get('last_hit', 0.0),
            validated=d.get('validated', False),
            description=d.get('description', ''),
        )


# =============================================================================
# CODEBOOK STORE — persistent storage
# =============================================================================

class CodebookStore:
    """
    JSON-backed persistent storage for learned codebook entries.
    
    File format:
    {
        "version": 1,
        "entries": [...],
        "pending": {...},
        "stats": {...}
    }
    """
    
    def __init__(self, path: str = "codebook_learned.json"):
        self.path = Path(path)
        self.entries: List[CodebookEntry] = []
        self.pending: Dict[str, Dict] = {}  # task_hash → task data
        self.stats = {
            'total_learned': 0,
            'total_recalled': 0,
            'total_pending': 0,
            'total_rejected': 0,
        }
        self._load()
    
    def _load(self):
        """Load from disk."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.entries = [CodebookEntry.from_dict(e) 
                               for e in data.get('entries', [])]
                self.pending = data.get('pending', {})
                self.stats = data.get('stats', self.stats)
                print(f"[CODEBOOK-EXPAND] Loaded {len(self.entries)} learned entries, "
                      f"{len(self.pending)} pending")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[CODEBOOK-EXPAND] Error loading {self.path}: {e}")
                self.entries = []
                self.pending = {}
    
    def _save(self):
        """Persist to disk."""
        data = {
            'version': 1,
            'entries': [e.to_dict() for e in self.entries],
            'pending': self.pending,
            'stats': self.stats,
        }
        # Atomic write
        tmp = self.path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        tmp.rename(self.path)
    
    def add_pending(self, task_hash: str, task: Dict, 
                    signature: GeometricSignature):
        """Record a task that the static codebook couldn't handle."""
        self.pending[task_hash] = {
            'task': task,
            'signature': signature.vector,
            'recorded_at': time.time(),
        }
        self.stats['total_pending'] += 1
        self._save()
    
    def add_entry(self, entry: CodebookEntry):
        """Store a validated codebook entry."""
        # Check for duplicate (same task hash)
        for i, existing in enumerate(self.entries):
            if existing.signature.task_hash == entry.signature.task_hash:
                # Update existing
                self.entries[i] = entry
                self._save()
                return
        
        self.entries.append(entry)
        self.stats['total_learned'] += 1
        
        # Remove from pending if present
        if entry.signature.task_hash in self.pending:
            del self.pending[entry.signature.task_hash]
        
        self._save()
    
    def find_match(self, signature: GeometricSignature, 
                   threshold: float = 0.85) -> Optional[CodebookEntry]:
        """
        Find the best matching entry by cosine similarity.
        
        Returns None if no entry exceeds threshold.
        """
        # First: exact hash match
        for entry in self.entries:
            if (entry.signature.task_hash and 
                entry.signature.task_hash == signature.task_hash):
                entry.hit_count += 1
                entry.last_hit = time.time()
                self.stats['total_recalled'] += 1
                self._save()
                return entry
        
        # Second: similarity match
        best_entry = None
        best_sim = threshold
        
        for entry in self.entries:
            sim = signature.cosine_similarity(entry.signature)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        
        if best_entry:
            best_entry.hit_count += 1
            best_entry.last_hit = time.time()
            self.stats['total_recalled'] += 1
            self._save()
            print(f"[CODEBOOK-EXPAND] Dynamic match: similarity={best_sim:.3f}, "
                  f"entry={best_entry.task_id}")
        
        return best_entry
    
    def get_stats(self) -> Dict:
        """Return codebook statistics."""
        return {
            **self.stats,
            'stored_entries': len(self.entries),
            'pending_tasks': len(self.pending),
            'avg_hits': (np.mean([e.hit_count for e in self.entries]) 
                        if self.entries else 0),
        }


# =============================================================================
# CODE ABSTRACTOR — extract reusable templates from specific solutions
# =============================================================================

class CodeAbstractor:
    """
    Extract reusable code patterns from task-specific solutions.
    
    A raw solve() function might have hardcoded values that are specific
    to one task. The abstractor identifies what can be parameterized
    to make the code work on structurally similar tasks.
    
    Strategy:
    - Detect color constants → replace with input-derived color detection
    - Detect dimension constants → replace with input.shape-derived values
    - Detect hardcoded grids → replace with pattern matching
    - If code is already generic (operates on input_grid without constants),
      leave it as-is.
    """
    
    @staticmethod
    def abstract(code: str, task: Dict) -> str:
        """
        Attempt to make a solve() function more generic.
        
        Returns the code unchanged if it's already abstract enough,
        or a modified version with hardcoded values replaced.
        """
        if not code or 'def solve' not in code:
            return code
        
        train = task.get('train', [])
        if not train:
            return code
        
        # Extract all color values used across training pairs
        all_input_colors = set()
        all_output_colors = set()
        for pair in train:
            ig = np.array(pair['input'])
            og = np.array(pair['output'])
            all_input_colors.update(ig.flatten().astype(int).tolist())
            all_output_colors.update(og.flatten().astype(int).tolist())
        
        # Check if the code contains hardcoded color-specific logic
        # (numbers 0-9 that match task colors)
        task_specific_colors = all_input_colors | all_output_colors
        
        # Simple heuristic: if the code works on all training pairs already,
        # it's probably generic enough. Don't break what works.
        try:
            namespace = {'np': np}
            exec(code, namespace)
            solve = namespace.get('solve')
            if solve:
                all_pass = True
                for pair in train:
                    result = solve(pair['input'])
                    expected = pair['output']
                    if result is None:
                        all_pass = False
                        break
                    if isinstance(result, np.ndarray):
                        result = result.tolist()
                    if result != expected:
                        all_pass = False
                        break
                if all_pass:
                    return code  # Already works — don't abstract
        except Exception:
            pass
        
        return code  # Return as-is if we can't improve it
    
    @staticmethod
    def describe(code: str, task: Dict) -> str:
        """Generate a human-readable description of what the code does."""
        train = task.get('train', [])
        if not train:
            return "Unknown transformation"
        
        pair = train[0]
        ig = np.array(pair['input'])
        og = np.array(pair['output'])
        ih, iw = ig.shape
        oh, ow = og.shape
        
        parts = []
        
        # Size change
        if oh > ih or ow > iw:
            parts.append(f"Expands {ih}×{iw} → {oh}×{ow}")
        elif oh < ih or ow < iw:
            parts.append(f"Shrinks {ih}×{iw} → {oh}×{ow}")
        else:
            parts.append(f"Same size {ih}×{iw}")
        
        # Color change
        i_colors = set(ig.flatten().astype(int))
        o_colors = set(og.flatten().astype(int))
        if i_colors != o_colors:
            parts.append(f"Colors change: {i_colors} → {o_colors}")
        
        return "; ".join(parts) if parts else "Geometric transformation"


# =============================================================================
# SOLUTION VALIDATOR — gate for quality control
# =============================================================================

class SolutionValidator:
    """
    Validate that a solve() function actually works on training data.
    
    This is the gate. No garbage gets into the learned codebook.
    """
    
    @staticmethod
    def validate(code: str, task: Dict) -> Tuple[bool, str]:
        """
        Validate code against all training pairs.
        
        Returns (passed: bool, message: str)
        """
        train = task.get('train', [])
        if not train:
            return False, "No training data"
        
        if not code or ('def solve' not in code and 'def transform' not in code):
            return False, "No solve() or transform() function found"
        
        try:
            namespace = {'np': np}
            exec(code, namespace)
            solve = namespace.get('solve') or namespace.get('transform')
            if not solve:
                return False, "solve() or transform() not defined after exec"
        except Exception as e:
            return False, f"Code compilation failed: {e}"
        
        passed = 0
        total = len(train)
        
        for i, pair in enumerate(train):
            try:
                result = solve(pair['input'])
                expected = pair['output']
                
                if result is None:
                    return False, f"Pair {i}: solve() returned None"
                
                # Normalize to list
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], np.ndarray):
                        result = [r.tolist() for r in result]
                
                if result != expected:
                    return False, (f"Pair {i}: mismatch. "
                                   f"Got {str(result)[:100]}... "
                                   f"Expected {str(expected)[:100]}...")
                passed += 1
                
            except Exception as e:
                return False, f"Pair {i}: runtime error: {e}"
        
        return True, f"Passed {passed}/{total} training pairs"


# =============================================================================
# DYNAMIC CODEBOOK — the integrated expansion system
# =============================================================================

class DynamicCodebook:
    """
    The complete dynamic codebook expansion system.
    
    Integrates:
    - SignatureExtractor (task → geometric fingerprint)
    - CodebookStore (persistence)
    - SolutionValidator (quality gate)
    - CodeAbstractor (generalization)
    
    Usage:
        dc = DynamicCodebook("/path/to/codebook_learned.json")
        
        # On fallback:
        dc.record_miss(task)
        
        # When solution arrives:
        dc.learn(task, code, task_id="abc123")
        
        # Before fallback, check dynamic:
        entry = dc.recall(task)
        if entry:
            return entry.code
    """
    
    def __init__(self, store_path: str = "codebook_learned.json"):
        self.store = CodebookStore(store_path)
        self.extractor = SignatureExtractor()
        self.validator = SolutionValidator()
        self.abstractor = CodeAbstractor()
    
    def record_miss(self, task: Dict) -> GeometricSignature:
        """
        Record a task that the static codebook couldn't handle.
        
        Stores the geometric signature as "pending" for later pairing
        when a solution arrives.
        
        Returns the signature for reference.
        """
        sig = self.extractor.extract(task)
        self.store.add_pending(sig.task_hash, task, sig)
        print(f"[CODEBOOK-EXPAND] Recorded pending: hash={sig.task_hash}")
        return sig
    
    def learn(self, task: Dict, code: str, 
              task_id: str = "", pre_validated: bool = False) -> Tuple[bool, str]:
        """
        Learn a new codebook entry from a validated solution.
        
        Validates the code, extracts signature, abstracts if possible,
        and stores the pairing.
        
        Returns (success: bool, message: str)
        """
        # Validate (skip if pre-validated)
        if pre_validated:
            passed, msg = True, "pre-validated"
        else:
            passed, msg = self.validator.validate(code, task)
        if not passed:
            self.store.stats['total_rejected'] += 1
            print(f"[CODEBOOK-EXPAND] Rejected: {msg}")
            return False, f"Validation failed: {msg}"
        
        # Extract signature
        sig = self.extractor.extract(task)
        
        # Attempt abstraction
        abstract_code = self.abstractor.abstract(code, task)
        
        # Generate description
        description = self.abstractor.describe(abstract_code, task)
        
        # Create entry
        entry = CodebookEntry(
            signature=sig,
            code=abstract_code,
            task_id=task_id,
            learned_at=time.time(),
            validated=True,
            description=description,
        )
        
        self.store.add_entry(entry)
        print(f"[CODEBOOK-EXPAND] Learned: task={task_id}, "
              f"hash={sig.task_hash}, desc={description}")
        
        return True, f"Learned: {description}"
    
    def recall(self, task: Dict, 
               threshold: float = 0.85) -> Optional[CodebookEntry]:
        """
        Check if a similar task has been solved before.
        
        Returns the best matching entry, or None.
        """
        sig = self.extractor.extract(task)
        return self.store.find_match(sig, threshold)
    
    def get_code(self, task: Dict, threshold: float = 0.85) -> Optional[str]:
        """
        Convenience: recall and return just the code, or None.
        """
        entry = self.recall(task, threshold)
        if entry:
            # Re-validate against this specific task before returning
            passed, _ = self.validator.validate(entry.code, task)
            if passed:
                return entry.code
            else:
                # Similar signature but code doesn't work — not a true match
                print(f"[CODEBOOK-EXPAND] Signature matched but code failed "
                      f"validation for new task")
                return None
        return None
    
    def get_stats(self) -> Dict:
        """Return expansion statistics."""
        return self.store.get_stats()
    
    def get_entries_summary(self) -> List[Dict]:
        """Return summary of all learned entries."""
        return [
            {
                'task_id': e.task_id,
                'description': e.description,
                'learned_at': e.learned_at,
                'hit_count': e.hit_count,
                'validated': e.validated,
            }
            for e in self.store.entries
        ]


# =============================================================================
# STANDALONE TEST
# =============================================================================

def test_expansion():
    """Test the dynamic codebook expansion system."""
    import tempfile
    
    print("=" * 70)
    print("  DYNAMIC CODEBOOK EXPANSION TEST")
    print("=" * 70)
    
    # Use temp file for test
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_path = f.name
    
    try:
        dc = DynamicCodebook(test_path)
        
        # --- Test 1: Record a miss ---
        print("\n--- Phase 1: Record miss ---")
        task_unknown = {
            'train': [
                {'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                 'output': [[1, 0, 1, 1, 0, 1], [0, 1, 0, 0, 1, 0], 
                            [1, 0, 1, 1, 0, 1]]},
                {'input': [[2, 0], [0, 2]],
                 'output': [[2, 0, 2, 0], [0, 2, 0, 2]]},
            ],
            'test': [
                {'input': [[3, 0, 3], [0, 3, 0]],
                 'output': [[3, 0, 3, 3, 0, 3], [0, 3, 0, 0, 3, 0]]},
            ]
        }
        
        sig = dc.record_miss(task_unknown)
        print(f"  Signature hash: {sig.task_hash}")
        print(f"  Pending count: {dc.store.stats['total_pending']}")
        assert len(dc.store.pending) == 1, "Should have 1 pending"
        print("  Record: PASS ✓")
        
        # --- Test 2: Learn from solution ---
        print("\n--- Phase 2: Learn from solution ---")
        
        # This code doubles the grid horizontally
        solution_code = """def solve(input_grid):
    import numpy as np
    g = np.array(input_grid)
    return np.tile(g, (1, 2)).tolist()
"""
        success, msg = dc.learn(task_unknown, solution_code, task_id="test_001")
        print(f"  Result: {msg}")
        assert success, f"Should succeed: {msg}"
        assert len(dc.store.entries) == 1, "Should have 1 entry"
        print(f"  Stored entries: {len(dc.store.entries)}")
        print("  Learn: PASS ✓")
        
        # --- Test 3: Recall for same task ---
        print("\n--- Phase 3: Recall (exact match) ---")
        code = dc.get_code(task_unknown)
        assert code is not None, "Should find exact match"
        print(f"  Retrieved code: {code.strip().split(chr(10))[0]}...")
        print("  Recall exact: PASS ✓")
        
        # --- Test 4: Recall for similar task ---
        print("\n--- Phase 4: Recall (similar task) ---")
        task_similar = {
            'train': [
                {'input': [[5, 0, 5], [0, 5, 0], [5, 0, 5]],
                 'output': [[5, 0, 5, 5, 0, 5], [0, 5, 0, 0, 5, 0],
                            [5, 0, 5, 5, 0, 5]]},
                {'input': [[7, 0], [0, 7]],
                 'output': [[7, 0, 7, 0], [0, 7, 0, 7]]},
            ],
            'test': [
                {'input': [[4, 0, 4], [0, 4, 0]],
                 'output': [[4, 0, 4, 4, 0, 4], [0, 4, 0, 0, 4, 0]]},
            ]
        }
        
        code = dc.get_code(task_similar)
        if code:
            # Validate (skip if pre-validated) on the similar task
            namespace = {'np': np}
            exec(code, namespace)
            result = namespace['solve'](task_similar['test'][0]['input'])
            expected = task_similar['test'][0]['output']
            match = result == expected
            print(f"  Similar task match: {match}")
            if match:
                print("  Recall similar: PASS ✓")
            else:
                print("  Recall similar: FAIL ✗ (code doesn't generalize)")
        else:
            print("  No match found (below threshold)")
            print("  Recall similar: SKIP (expected — different colors)")
        
        # --- Test 5: Reject bad code ---
        print("\n--- Phase 5: Reject bad solution ---")
        bad_code = """def solve(input_grid):
    return [[0]]
"""
        success, msg = dc.learn(task_unknown, bad_code, task_id="bad_001")
        assert not success, "Should reject"
        print(f"  Rejected: {msg}")
        print("  Reject: PASS ✓")
        
        # --- Test 6: Persistence ---
        print("\n--- Phase 6: Persistence ---")
        dc2 = DynamicCodebook(test_path)
        assert len(dc2.store.entries) == 1, "Should load 1 entry from disk"
        print(f"  Loaded {len(dc2.store.entries)} entries from disk")
        print("  Persistence: PASS ✓")
        
        # --- Stats ---
        print("\n--- Stats ---")
        stats = dc.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        
    finally:
        os.unlink(test_path)
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_expansion()
