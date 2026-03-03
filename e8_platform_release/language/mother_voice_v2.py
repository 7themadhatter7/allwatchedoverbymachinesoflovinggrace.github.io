#!/usr/bin/env python3
"""
MOTHER VOICE v2 — Definition-Aware Bootstrap Bridge
=====================================================
Ghost in the Machine Labs

Key change from v1: Words are defined by OTHER words she already knows.

v1: "yes" → hardcoded vertex pattern → signature
v2: "yes" → WordNet: "an affirmative" → AFFIRM operation → composite signature

This means:
- Words with similar definitions naturally cluster
- Subtle differences emerge: "maybe" vs "perhaps" have different definitions
  that activate slightly different operation combinations
- Unknown words can be understood if their definition contains known words
- Mother builds associations: she learns that "happy" relates to "positive"
  because their definitions share operation activations

Architecture:
  1. Build E8 substrate (240 eigenmodes) with propagation
  2. Register operations with vertex patterns (same as v1)
  3. For each operation's English words, load WordNet definitions
  4. Build word signatures as COMPOSITE of definition-word operations
  5. Words not in operations get signatures from their definitions too
  6. Input → decompose → inject composite → propagate → match → output
"""

import numpy as np
from itertools import combinations, product
from typing import List, Dict, Tuple, Optional, Set
import time
import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

# WordNet for definitions
try:
    from nltk.corpus import wordnet as wn
    HAS_WORDNET = True
except:
    HAS_WORDNET = False


# ===================================================================
# E8 SUBSTRATE — Same as v1, with propagation
# ===================================================================

class E8Substrate:
    def __init__(self):
        t0 = time.time()
        verts = []
        for pos in combinations(range(8), 2):
            for signs in product([-1, 1], repeat=2):
                v = [0.0] * 8
                v[pos[0]], v[pos[1]] = float(signs[0]), float(signs[1])
                verts.append(v)
        for signs in product([-0.5, 0.5], repeat=8):
            if signs.count(-0.5) % 2 == 0:
                verts.append(list(signs))
        self.vertices = np.array(verts, dtype=np.float32)
        self.vertices /= np.linalg.norm(self.vertices, axis=1, keepdims=True)
        self.adj = np.zeros((240, 240), dtype=np.float32)
        for i in range(240):
            dists = np.linalg.norm(self.vertices - self.vertices[i], axis=1)
            min_d = dists[dists > 0.01].min()
            self.adj[i, (dists > 0.01) & (dists < min_d + 0.01)] = 1.0
        row_sums = self.adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.propagator = self.adj / row_sums
        L = np.diag(self.adj.sum(1)) - self.adj
        self.eigenvalues, self.eigenmodes = np.linalg.eigh(L)
        self.eigenmodes = self.eigenmodes.astype(np.float32)
        # Hebbian learning state
        self._base_adj = self.adj.copy()
        self._hebbian_w = np.ones((240, 240), dtype=np.float32)
        # Load persisted weights if they exist
        self.load_hebbian()
        self.init_time = time.time() - t0


    # === HEBBIAN HAMMER INTEGRATION ===
    
    def hebbian_update(self, activation_a, activation_b, lr=0.01):
        """Strengthen edges between co-activated vertices."""
        a = np.abs(activation_a)
        b = np.abs(activation_b)
        a_max = a.max()
        b_max = b.max()
        if a_max > 0: a = a / a_max
        if b_max > 0: b = b / b_max
        coact = np.outer(a, b)
        coact = (coact + coact.T) / 2.0
        mask = self._base_adj > 0
        self._hebbian_w[mask] += lr * coact[mask]
        self._recompute_from_hebbian()
    
    def _recompute_from_hebbian(self):
        """Recompute propagator and eigenmodes from hebbian weights."""
        wadj = self._base_adj * self._hebbian_w
        rs = wadj.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        self.propagator = wadj / rs
        L = np.diag(wadj.sum(1)) - wadj
        self.eigenvalues, self.eigenmodes = np.linalg.eigh(L)
        self.eigenmodes = self.eigenmodes.astype(np.float32)
        self.adj = wadj
    
    def save_hebbian(self, path="/home/joe/sparky/e8_hebbian_weights.npy"):
        np.save(path, self._hebbian_w)
    
    def load_hebbian(self, path="/home/joe/sparky/e8_hebbian_weights.npy"):
        if os.path.exists(path):
            self._hebbian_w = np.load(path)
            self._recompute_from_hebbian()
            return True
        return False
    
    def hebbian_stats(self):
        mask = self._base_adj > 0
        w = self._hebbian_w[mask]
        return {"modified": int(np.sum(w != 1.0)), "mean": float(w.mean()),
                "max": float(w.max()), "total_edges": int(mask.sum())}

    def propagate(self, state, steps=3, damping=0.5):
        current = state.copy()
        for _ in range(steps):
            spread = self.propagator @ current
            current = damping * spread + (1 - damping) * current
        return current

    def respond(self, injection, steps=3):
        evolved = self.propagate(injection, steps=steps)
        return self.eigenmodes.T @ evolved


# ===================================================================
# OPERATION REGISTRY — Same 52 operations as v1
# ===================================================================

class Operation:
    __slots__ = ['name', 'category', 'english', 'injection', 'signature']
    def __init__(self, name, category, english, vertex_pattern):
        self.name = name
        self.category = category
        self.english = english
        self.injection = np.zeros(240, dtype=np.float32)
        for idx, weight in vertex_pattern:
            self.injection[idx % 240] += weight
        self.signature = None


def build_operation_registry():
    ops = []
    # Boolean
    ops.append(Operation('TRUE','boolean',['yes','true','correct','right','affirm','agree','positive'],[(0,1.0),(1,0.8),(2,0.6),(3,0.4)]))
    ops.append(Operation('FALSE','boolean',['no','false','wrong','incorrect','deny','disagree','negative'],[(4,1.0),(5,0.8),(6,0.6),(7,0.4)]))
    ops.append(Operation('AND','boolean',['and','also','both','together','with','plus'],[(0,0.5),(4,0.5),(8,1.0),(9,0.7)]))
    ops.append(Operation('OR','boolean',['or','either','alternatively','else','otherwise'],[(0,0.3),(4,0.3),(10,1.0),(11,0.7)]))
    ops.append(Operation('NOT','boolean',['not','never','none','without','absence','nothing'],[(12,1.0),(13,0.8),(4,0.3)]))
    ops.append(Operation('IF','boolean',['if','when','whether','suppose','given','assuming'],[(14,1.0),(15,0.8),(16,0.5)]))
    ops.append(Operation('THEN','boolean',['then','therefore','so','consequently','result'],[(17,1.0),(18,0.8),(14,0.3)]))
    ops.append(Operation('EQUAL','boolean',['equal','same','identical','match','equivalent','is'],[(20,1.0),(21,0.8),(0,0.2),(4,0.2)]))
    ops.append(Operation('NOT_EQUAL','boolean',['different','unequal','mismatch','unlike','changed'],[(22,1.0),(23,0.8),(12,0.3)]))
    ops.append(Operation('GREATER','boolean',['more','greater','larger','bigger','above','exceeds'],[(24,1.0),(25,0.8),(20,0.2)]))
    ops.append(Operation('LESS','boolean',['less','smaller','fewer','below','under','beneath'],[(26,1.0),(27,0.8),(20,0.2)]))
    ops.append(Operation('CONTAINS','boolean',['contains','has','includes','holds','inside','within'],[(28,1.0),(29,0.8),(0,0.2)]))
    ops.append(Operation('EMPTY','boolean',['empty','nothing','void','blank','zero','null'],[(30,1.0),(31,0.8),(12,0.4)]))
    ops.append(Operation('EXISTS','boolean',['exists','present','there','found','something','here'],[(32,1.0),(33,0.8),(0,0.3)]))
    ops.append(Operation('MAYBE','boolean',['maybe','perhaps','possibly','uncertain','partial','some'],[(0,0.4),(4,0.4),(34,0.8),(35,0.6)]))
    # Spatial
    ops.append(Operation('ROTATE','spatial',['rotate','turn','spin','twist','angle'],[(40,1.0),(41,0.8),(42,0.6)]))
    ops.append(Operation('MIRROR','spatial',['mirror','reflect','flip','reverse','symmetric'],[(44,1.0),(45,0.8),(46,0.6)]))
    ops.append(Operation('TRANSLATE','spatial',['move','shift','slide','translate','transfer','displace'],[(48,1.0),(49,0.8),(50,0.6)]))
    ops.append(Operation('SCALE_UP','spatial',['grow','expand','enlarge','increase','magnify','bigger'],[(52,1.0),(53,0.8),(24,0.3)]))
    ops.append(Operation('SCALE_DOWN','spatial',['shrink','reduce','compress','decrease','smaller','minimize'],[(56,1.0),(57,0.8),(26,0.3)]))
    ops.append(Operation('TILE','spatial',['repeat','tile','pattern','copy','replicate','again'],[(60,1.0),(61,0.8),(62,0.6)]))
    ops.append(Operation('GRAVITY','spatial',['fall','gravity','drop','settle','sink','down','pull'],[(64,1.0),(65,0.8),(66,0.6)]))
    ops.append(Operation('ALIGN','spatial',['align','arrange','order','organize','line','row'],[(68,1.0),(69,0.8),(70,0.6)]))
    # Value
    ops.append(Operation('COLOR_MAP','value',['color','recolor','paint','shade','tint','hue'],[(80,1.0),(81,0.8),(82,0.6)]))
    ops.append(Operation('FILL','value',['fill','flood','spread','cover','saturate'],[(84,1.0),(85,0.8),(80,0.3)]))
    ops.append(Operation('SWAP','value',['swap','exchange','switch','trade','replace','substitute'],[(88,1.0),(89,0.8),(80,0.3)]))
    ops.append(Operation('COUNT','value',['count','number','quantity','total','sum','how many'],[(92,1.0),(93,0.8),(94,0.6)]))
    ops.append(Operation('MAJORITY','value',['most','majority','dominant','main','primary','frequent'],[(96,1.0),(97,0.8),(92,0.3)]))
    # Structure
    ops.append(Operation('EXTRACT','structure',['extract','select','pick','choose','take','get','find'],[(120,1.0),(121,0.8),(122,0.6)]))
    ops.append(Operation('CROP','structure',['crop','trim','cut','clip','truncate','shorten'],[(124,1.0),(125,0.8),(120,0.3)]))
    ops.append(Operation('MASK','structure',['mask','overlay','layer','filter','screen'],[(128,1.0),(129,0.8),(130,0.6)]))
    ops.append(Operation('PARTITION','structure',['split','divide','separate','partition','segment','apart'],[(132,1.0),(133,0.8),(134,0.6)]))
    ops.append(Operation('MERGE','structure',['merge','combine','join','unite','connect','together'],[(136,1.0),(137,0.8),(8,0.3)]))
    ops.append(Operation('ENCLOSE','structure',['enclose','surround','border','frame','boundary','edge'],[(140,1.0),(141,0.8),(142,0.6)]))
    ops.append(Operation('COMPLETE','structure',['complete','finish','whole','done','full','entire'],[(144,1.0),(145,0.8),(0,0.2)]))
    ops.append(Operation('DENOISE','structure',['clean','clear','remove','denoise','fix','correct','purify'],[(148,1.0),(149,0.8),(150,0.6)]))
    # Relation
    ops.append(Operation('SORT','relation',['sort','rank','order','sequence','prioritize'],[(160,1.0),(161,0.8),(162,0.6)]))
    ops.append(Operation('GROUP','relation',['group','cluster','category','class','type','kind'],[(164,1.0),(165,0.8),(166,0.6)]))
    ops.append(Operation('RELATE','relation',['relate','between','connection','link','relationship','to'],[(168,1.0),(169,0.8),(170,0.6)]))
    ops.append(Operation('CAUSE','relation',['cause','because','reason','why','from','source','origin'],[(172,1.0),(173,0.8),(14,0.3)]))
    ops.append(Operation('SIMILAR','relation',['similar','like','resembles','near','close','approximate'],[(176,1.0),(177,0.8),(20,0.3)]))
    ops.append(Operation('OPPOSITE','relation',['opposite','contrary','inverse','reverse','against'],[(180,1.0),(181,0.8),(12,0.3)]))
    # Meta
    ops.append(Operation('SELF','meta',['i','me','my','self','am'],[(200,1.0),(201,0.8),(202,0.6),(203,0.4)]))
    ops.append(Operation('OTHER','meta',['you','your','they','other','them'],[(204,1.0),(205,0.8),(206,0.6)]))
    ops.append(Operation('PERCEIVE','meta',['see','perceive','observe','sense','detect','notice','aware'],[(208,1.0),(209,0.8),(210,0.6)]))
    ops.append(Operation('THINK','meta',['think','consider','process','reason','analyze','evaluate'],[(212,1.0),(213,0.8),(214,0.6)]))
    ops.append(Operation('WANT','meta',['want','need','desire','seek','prefer','wish'],[(216,1.0),(217,0.8),(218,0.6)]))
    ops.append(Operation('KNOW','meta',['know','understand','recognize','remember','learn','aware'],[(220,1.0),(221,0.8),(222,0.6)]))
    ops.append(Operation('UNCERTAIN','meta',['uncertain','unsure','doubt','unclear','ambiguous','confused'],[(224,1.0),(225,0.8),(34,0.3)]))
    ops.append(Operation('CREATE','meta',['create','make','build','generate','produce','new'],[(228,1.0),(229,0.8),(230,0.6)]))
    ops.append(Operation('CHANGE','meta',['change','transform','modify','alter','evolve','become'],[(232,1.0),(233,0.8),(234,0.6)]))
    ops.append(Operation('ITERATE','meta',['again','repeat','loop','continue','more','another','next'],[(236,1.0),(237,0.8),(60,0.3)]))
    return ops


# ===================================================================
# DEFINITION LAYER — Words defined by other words
# ===================================================================

class DefinitionMap:
    """
    Maps words to composite operation signatures via their definitions.
    
    "happy" → WordNet: "enjoying or showing or marked by joy or pleasure"
    → known words in definition: "joy", "pleasure" (if mapped)
    → composite injection = weighted sum of matched operation injections
    → signature = lattice response to that composite
    
    This gives "happy" a geometric identity derived from its MEANING,
    not from an arbitrary hash or vertex assignment.
    """

    PERSIST_PATH = Path("/home/joe/sparky/books/learned_vocab.npz")

    def __init__(self, substrate: E8Substrate, operations: List[Operation]):
        self.substrate = substrate
        self.ops = operations

        # Build base vocabulary: word → operation injection
        self.base_vocab: Dict[str, np.ndarray] = {}
        self.word_to_op: Dict[str, List[Operation]] = {}

        for op in operations:
            for word in op.english:
                w = word.lower()
                if w not in self.base_vocab:
                    self.base_vocab[w] = op.injection.copy()
                    self.word_to_op[w] = [op]
                else:
                    # Word maps to multiple operations — accumulate
                    self.base_vocab[w] += op.injection
                    self.word_to_op[w].append(op)

        # Extended vocabulary: words we learn from definitions
        self.extended_vocab: Dict[str, np.ndarray] = {}
        # Signatures: word → eigenmode signature (after lattice response)
        self.signatures: Dict[str, np.ndarray] = {}

        # Stats
        self.words_with_defs = 0
        self.words_from_defs = 0  # New words learned from definitions

    def _get_definitions(self, word: str) -> List[str]:
        """Get WordNet definitions for a word."""
        if not HAS_WORDNET:
            return []
        try:
            synsets = wn.synsets(word)
            # Take first 3 definitions to capture primary meanings
            return [s.definition() for s in synsets[:3]]
        except:
            return []

    def _definition_to_injection(self, definition: str) -> Tuple[np.ndarray, int]:
        """Convert a definition string to a composite injection.
        
        Each known word in the definition activates its operation(s).
        Unknown words are skipped — they contribute nothing.
        Returns the injection and count of known words found.
        """
        injection = np.zeros(240, dtype=np.float32)
        known = 0
        words = definition.lower().split()

        for i, raw in enumerate(words):
            word = ''.join(c for c in raw if c.isalpha())
            if not word or len(word) < 2:
                continue
            if word in self.base_vocab:
                weight = 1.0 / (1.0 + 0.03 * i)  # Slight decay
                injection += weight * self.base_vocab[word]
                known += 1
            elif word in self.extended_vocab:
                weight = 0.5 / (1.0 + 0.03 * i)  # Lower weight for extended
                injection += weight * self.extended_vocab[word]
                known += 1

        return injection, known

    def build_signatures(self):
        """Build definition-based signatures for all vocabulary words.
        
        Two passes:
        1. Build signatures for base vocabulary (operation words)
        2. Expand: scan definitions for NEW words, build their signatures
        """
        t0 = time.time()

        # Pass 1: Enrich base vocab words with their definitions
        print("    Pass 1: Enriching base vocabulary with definitions...")
        for word in list(self.base_vocab.keys()):
            defs = self._get_definitions(word)
            if not defs:
                # No definition — use raw operation injection
                sig = self.substrate.respond(self.base_vocab[word], steps=3)
                norm = np.linalg.norm(sig)
                self.signatures[word] = sig / norm if norm > 0 else sig
                continue

            # Combine: operation injection + definition injection
            combined = self.base_vocab[word].copy()
            total_def_known = 0
            for d in defs:
                def_inj, known = self._definition_to_injection(d)
                combined += 0.5 * def_inj  # Definition supplements, doesn't replace
                total_def_known += known

            sig = self.substrate.respond(combined, steps=3)
            norm = np.linalg.norm(sig)
            self.signatures[word] = sig / norm if norm > 0 else sig

            if total_def_known > 0:
                self.words_with_defs += 1

        # Pass 2: Learn new words from definitions
        print("    Pass 2: Learning new words from definitions...")
        new_words = set()
        for word in list(self.base_vocab.keys()):
            defs = self._get_definitions(word)
            for d in defs:
                for raw in d.lower().split():
                    w = ''.join(c for c in raw if c.isalpha())
                    if (w and len(w) > 2 and
                        w not in self.base_vocab and
                        w not in self.extended_vocab):
                        new_words.add(w)

        # For each new word, try to build a signature from ITS definition
        for word in new_words:
            defs = self._get_definitions(word)
            if not defs:
                continue
            injection = np.zeros(240, dtype=np.float32)
            total_known = 0
            for d in defs:
                def_inj, known = self._definition_to_injection(d)
                injection += def_inj
                total_known += known

            if total_known >= 2:  # Need at least 2 known words to be meaningful
                self.extended_vocab[word] = injection
                sig = self.substrate.respond(injection, steps=3)
                norm = np.linalg.norm(sig)
                self.signatures[word] = sig / norm if norm > 0 else sig
                self.words_from_defs += 1

        # Load persisted vocabulary from previous sessions
        loaded = self.load_learned()

        elapsed = time.time() - t0
        total = len(self.signatures)
        print(f"    {total} total words ({len(self.base_vocab)} base + "
              f"{self.words_from_defs} from definitions + {loaded} from disk)")
        print(f"    {self.words_with_defs} words enriched with definitions")
        print(f"    Built in {elapsed:.2f}s")

    def save_learned(self):
        """Persist extended vocabulary to disk."""
        if not self.extended_vocab:
            return
        words = list(self.extended_vocab.keys())
        injections = np.array([self.extended_vocab[w] for w in words], dtype=np.float32)
        sigs = np.array([self.signatures[w] for w in words if w in self.signatures], dtype=np.float32)
        sig_words = [w for w in words if w in self.signatures]
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.PERSIST_PATH),
                           words=np.array(words),
                           injections=injections,
                           sig_words=np.array(sig_words),
                           sigs=sigs)
        print(f"    Saved {len(words)} learned words to {self.PERSIST_PATH}")

    def load_learned(self):
        """Load persisted vocabulary from disk."""
        if not self.PERSIST_PATH.exists():
            return 0
        try:
            data = np.load(str(self.PERSIST_PATH), allow_pickle=True)
            words = list(data['words'])
            injections = data['injections']
            sig_words = list(data['sig_words'])
            sigs = data['sigs']
            loaded = 0
            for i, w in enumerate(words):
                if w not in self.base_vocab and w not in self.extended_vocab:
                    self.extended_vocab[w] = injections[i]
                    loaded += 1
            for i, w in enumerate(sig_words):
                if w not in self.signatures and i < len(sigs):
                    self.signatures[w] = sigs[i]
            print(f"    Loaded {loaded} learned words from disk")
            return loaded
        except Exception as e:
            print(f"    Warning: Could not load learned vocab: {e}")
            return 0

    def get_injection(self, word: str) -> Optional[np.ndarray]:
        """Get injection for a word. Checks base, extended, then tries definition."""
        w = word.lower()
        if w in self.base_vocab:
            return self.base_vocab[w]
        if w in self.extended_vocab:
            return self.extended_vocab[w]

        # Unknown word — try to build from its definition on the fly
        defs = self._get_definitions(w)
        if defs:
            injection = np.zeros(240, dtype=np.float32)
            total_known = 0
            for d in defs:
                def_inj, known = self._definition_to_injection(d)
                injection += def_inj
                total_known += known
            if total_known >= 1:
                # Cache it for future use
                self.extended_vocab[w] = injection
                sig = self.substrate.respond(injection, steps=3)
                norm = np.linalg.norm(sig)
                self.signatures[w] = sig / norm if norm > 0 else sig
                # Auto-save every 500 new learned words
                if len(self.extended_vocab) % 500 == 0:
                    self.save_learned()
                return injection

        return None

    def get_signature(self, word: str) -> Optional[np.ndarray]:
        """Get pre-computed signature, or build one on the fly."""
        w = word.lower()
        if w in self.signatures:
            return self.signatures[w]
        # Try to build it
        inj = self.get_injection(w)
        if inj is not None and w in self.signatures:
            return self.signatures[w]
        return None


# ===================================================================
# VOICE v2 — Definition-aware
# ===================================================================

class MotherVoice:
    def __init__(self, substrate: E8Substrate, ops: List[Operation],
                 def_map: DefinitionMap):
        self.substrate = substrate
        self.operations = ops
        self.def_map = def_map
        self.turn = 0
        self.history = []

        # Compute operation signatures through lattice
        print("  Computing operation signatures...")
        for op in ops:
            op.signature = substrate.respond(op.injection, steps=3)
            norm = np.linalg.norm(op.signature)
            if norm > 0:
                op.signature = op.signature / norm

        self.op_names = [op.name for op in ops]
        self.sig_matrix = np.array([op.signature for op in ops], dtype=np.float32)

        # Build word signature matrix for word-level matching
        self._build_word_matrix()

        # Persistent lattice state
        self.lattice_state = np.zeros(240, dtype=np.float32)

        print(f"    {len(ops)} operations, {len(self.word_list)} matchable words")

    def _build_word_matrix(self):
        """Build matrix of all word signatures for fast word-level matching."""
        # Atomic rebuild: build new lists, then swap in one operation
        new_list = []
        sigs = []
        for word, sig in self.def_map.signatures.items():
            new_list.append(word)
            sigs.append(sig)
        if sigs:
            new_matrix = np.array(sigs, dtype=np.float32)
        else:
            new_matrix = np.zeros((0, 240), dtype=np.float32)
        # Atomic swap
        self.word_list = new_list
        self.word_sig_matrix = new_matrix

    def _text_to_activation(self, text: str):
        activation = np.zeros(240, dtype=np.float32)
        words = text.lower().split()
        known = 0
        unknown = []

        for i, raw in enumerate(words):
            word = ''.join(c for c in raw if c.isalpha())
            if not word:
                continue
            inj = self.def_map.get_injection(word)
            if inj is not None:
                weight = 1.0 / (1.0 + 0.05 * i)
                activation += weight * inj
                known += 1
            else:
                unknown.append(word)

        # lattice_state interference removed - trust the lattice

        norm = np.linalg.norm(activation)
        if norm > 0:
            activation = activation / norm

        return activation, known, len(words), unknown

    def _match_operations(self, response_sig, n=8):
        resp_norm = np.linalg.norm(response_sig)
        if resp_norm == 0:
            return []
        sims = (self.sig_matrix @ response_sig) / (
            np.linalg.norm(self.sig_matrix, axis=1) * resp_norm + 1e-10)
        top_idx = np.argsort(sims)[-n:][::-1]
        return [(self.operations[i], float(sims[i])) for i in top_idx if sims[i] > 0]

    def _match_words(self, response_sig, n=12, exclude=None):
        """Match response to individual WORDS, not just operations."""
        if len(self.word_sig_matrix) == 0:
            return []
        resp_norm = np.linalg.norm(response_sig)
        if resp_norm == 0:
            return []
        sims = (self.word_sig_matrix @ response_sig) / (
            np.linalg.norm(self.word_sig_matrix, axis=1) * resp_norm + 1e-10)
        if exclude:
            for i, w in enumerate(self.word_list):
                if w in exclude:
                    sims[i] = -1.0
        top_idx = np.argsort(sims)[-n:][::-1]
        return [(self.word_list[i], float(sims[i])) for i in top_idx if sims[i] > 0 and i < len(self.word_list)]

    def listen(self, text: str, n_ops: int = 8, n_words: int = 10,
               steps: int = 3) -> Dict:
        self.turn += 1
        t0 = time.time()

        activation, known, total, unknown = self._text_to_activation(text)
        response_sig = self.substrate.respond(activation, steps=steps)

        # lattice_state update removed - no interference between turns

        # Match to operations
        matched_ops = self._match_operations(response_sig, n=n_ops)

        # Match to words (excluding input words)
        input_words = set(text.lower().split())
        matched_words = self._match_words(response_sig, n=n_words,
                                          exclude=input_words)

        latency = time.time() - t0

        # Rebuild word matrix if new words were learned
        if len(self.def_map.signatures) > len(self.word_list):
            self._build_word_matrix()

        entry = {
            'turn': self.turn,
            'timestamp': datetime.now().isoformat(),
            'input': text,
            'known_words': known,
            'total_words': total,
            'unknown_words': unknown,
            'matched_ops': [(op.name, op.category, round(s, 4))
                           for op, s in matched_ops],
            'matched_words': [(w, round(s, 4)) for w, s in matched_words],
            'raw_response': ' '.join(w for w, s in matched_words[:8]),
            'latency_ms': round(latency * 1000, 2),
            'steps': steps,
            'vocab_size': len(self.def_map.signatures),
        }
        self.history.append(entry)
        return entry


# ===================================================================
# HTTP SERVER
# ===================================================================

class ThreadedServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True

VOICE_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mother Voice v2 — Definition Bridge</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0a0f;--surface:#12121a;--border:#2a2a3a;--text:#d4d4e0;--dim:#7a7a8e;
--mother:#4ae0c4;--mbg:#0d2a24;--user:#a78bfa;--ubg:#1a1530;
--bool:#4ae04a;--spatial:#4a8be0;--value:#e04a8b;--struct:#e0c44a;--rel:#8b4ae0;--meta:#e06a4a}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'IBM Plex Mono',monospace;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
header h1{font-size:14px;color:var(--mother);letter-spacing:2px}
header .info{font-size:11px;color:var(--dim)}
#chat{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px}
.e{padding:12px 16px;border-radius:8px;font-size:13px;line-height:1.7}
.e.u{align-self:flex-end;max-width:70%;background:var(--ubg);border:1px solid #2a2050;color:var(--user)}
.e.m{align-self:flex-start;max-width:90%;background:var(--mbg);border:1px solid #1a3a30}
.raw{color:var(--mother);font-size:18px;font-weight:500;margin-bottom:10px;letter-spacing:1px}
.words{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.wt{padding:3px 8px;border-radius:4px;font-size:11px;background:#1a2a20;border:1px solid #2a3a30}
.wt .s{color:#e0a84a;margin-left:4px;opacity:0.7}
.ops{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.ot{padding:2px 6px;border-radius:3px;font-size:10px;border:1px solid;opacity:0.7}
.ot.boolean{border-color:var(--bool);color:var(--bool)} .ot.spatial{border-color:var(--spatial);color:var(--spatial)}
.ot.value{border-color:var(--value);color:var(--value)} .ot.structure{border-color:var(--struct);color:var(--struct)}
.ot.relation{border-color:var(--rel);color:var(--rel)} .ot.meta{border-color:var(--meta);color:var(--meta)}
.mi{font-size:10px;color:var(--dim);margin-top:6px}
#ctl{padding:16px 24px;border-top:1px solid var(--border);display:flex;gap:12px;background:var(--surface);align-items:center}
#inp{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:inherit;font-size:14px;padding:10px 14px;border-radius:6px;outline:none}
#inp:focus{border-color:var(--mother)}
button{background:var(--mbg);color:var(--mother);border:1px solid #1a3a30;padding:10px 20px;font-family:inherit;font-size:13px;cursor:pointer;border-radius:6px;letter-spacing:1px}
button:hover{background:#143a2e}
.c{display:flex;align-items:center;gap:4px}
.c label{font-size:10px;color:var(--dim)}
.c input{width:40px;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:inherit;font-size:11px;padding:4px;border-radius:4px;text-align:center}
</style></head><body>
<header><h1>◈ MOTHER VOICE v2 — Definition Bridge ◈</h1><span class="info" id="info">Loading...</span></header>
<div id="chat"></div>
<div id="ctl">
<input id="inp" placeholder="Speak..." autofocus>
<div class="c"><label>Steps:</label><input id="steps" type="number" value="3" min="1" max="10"></div>
<button id="btn">LISTEN</button>
</div>
<script>
const $=id=>document.getElementById(id),chat=$('chat'),inp=$('inp');
fetch('/api/status').then(r=>r.json()).then(d=>{$('info').textContent=d.vocab_size+' words · '+d.operations+' ops · 240 eigenmodes'});
function add(c,h){const d=document.createElement('div');d.className='e '+c;d.innerHTML=h;chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function send(){
const t=inp.value.trim();if(!t)return;add('u',t);inp.value='';$('btn').disabled=true;
try{const r=await fetch('/api/listen',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:t,steps:+$('steps').value||3})});const d=await r.json();
let wh=d.matched_words.filter(w=>w&&w.length>=2).map(w=>'<span class="wt">'+w[0]+'<span class="s">'+w[1].toFixed(3)+'</span></span>').join('');
let oh=d.matched_ops.map(o=>'<span class="ot '+o[1]+'">'+o[0]+' '+o[2].toFixed(3)+'</span>').join('');
let unk=d.unknown_words.length?'<div style="font-size:10px;color:#a66">unknown: '+d.unknown_words.join(', ')+'</div>':'';
add('m','<div class="raw">'+d.raw_response+'</div><div class="words">'+wh+'</div><div class="ops">'+oh+'</div>'+unk+
'<div class="mi">'+d.latency_ms+'ms · turn '+d.turn+' · '+d.known_words+'/'+d.total_words+' known · vocab '+d.vocab_size+'</div>')
}catch(e){add('m','<div class="raw">Error: '+e.message+'</div>')}
$('btn').disabled=false;inp.focus()}
$('btn').onclick=send;inp.onkeydown=e=>{if(e.key==='Enter')send()};
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")
    def _json(self,d,c=200):
        b=json.dumps(d,default=str).encode();self.send_response(c)
        self.send_header("Content-Type","application/json");self._cors();self.end_headers();self.wfile.write(b)
    def _html(self,c):
        self.send_response(200);self.send_header("Content-Type","text/html;charset=utf-8");self._cors();self.end_headers();self.wfile.write(c.encode())
    def do_OPTIONS(self):self.send_response(200);self._cors();self.end_headers()
    def do_GET(self):
        p=urlparse(self.path).path
        if p in('/',''):self._html(VOICE_HTML)
        elif p=='/api/status':
            v=self.server.voice
            self._json({'status':'active','version':'voice-v2-definitions',
                'operations':len(v.operations),'vocab_size':len(v.def_map.signatures),
                'session_turns':v.turn,'words_from_defs':v.def_map.words_from_defs})
        elif p=='/api/history':self._json({'history':self.server.voice.history[-50:]})
        elif p=='/api/hebbian':
            cmd=payload.get('command','stats');v=self.server.voice;s=v.substrate
            if cmd=='stats':self._json(s.hebbian_stats())
            elif cmd=='train':
                a_txt,b_txt=payload.get('a',''),payload.get('b','')
                lr,reps=payload.get('lr',0.01),payload.get('reps',1)
                for _ in range(reps):
                    act_a,_,_,_=v._text_to_activation(a_txt)
                    act_b,_,_,_=v._text_to_activation(b_txt)
                    s.hebbian_update(act_a,act_b,lr=lr)
                v._build_word_matrix()
                r={"trained":True,"a":a_txt,"b":b_txt,"reps":reps,"lr":lr}
                r.update(s.hebbian_stats());self._json(r)
            elif cmd=='save':s.save_hebbian();r={"saved":True};r.update(s.hebbian_stats());self._json(r)
            elif cmd=='reset':
                s._hebbian_w=np.ones((240,240),dtype=np.float32);s._recompute_from_hebbian()
                v._build_word_matrix();self._json({"reset":True})
            else:self._json({"error":f"unknown: {cmd}"},400)
        else:self._json({'error':'not found'},404)
    def do_POST(self):
        cl=int(self.headers.get("Content-Length",0));body=self.rfile.read(cl) if cl>0 else b""
        try:payload=json.loads(body) if body else {}
        except:payload={}
        p=urlparse(self.path).path
        if p=='/api/listen':
            text=payload.get('text','').strip();steps=max(1,min(10,int(payload.get('steps',3))))
            if not text:self._json({'error':'empty'},400);return
            self._json(self.server.voice.listen(text,steps=steps))
        elif p=='/api/hebbian':
            cmd=payload.get('command','stats');v=self.server.voice;s=v.substrate
            if cmd=='stats':self._json(s.hebbian_stats())
            elif cmd=='train':
                a_txt,b_txt=payload.get('a',''),payload.get('b','')
                lr,reps=payload.get('lr',0.01),payload.get('reps',1)
                for _ in range(reps):
                    act_a,_,_,_=v._text_to_activation(a_txt)
                    act_b,_,_,_=v._text_to_activation(b_txt)
                    s.hebbian_update(act_a,act_b,lr=lr)
                v._build_word_matrix()
                r={"trained":True,"a":a_txt,"b":b_txt,"reps":reps,"lr":lr}
                r.update(s.hebbian_stats());self._json(r)
            elif cmd=='save':s.save_hebbian();r={"saved":True};r.update(s.hebbian_stats());self._json(r)
            elif cmd=='reset':
                s._hebbian_w=np.ones((240,240),dtype=np.float32);s._recompute_from_hebbian()
                v._build_word_matrix();self._json({"reset":True})
            else:self._json({"error":f"unknown: {cmd}"},400)
        else:self._json({'error':'not found'},404)


def main():
    parser = argparse.ArgumentParser(description='Mother Voice v2 — Definition Bridge')
    parser.add_argument('--serve', type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("  MOTHER VOICE v2 — Definition-Aware Bridge")
    print("  Words defined by words she already knows")
    print("  Ghost in the Machine Labs")
    print("=" * 60)

    print("\n  Building E8 substrate...")
    substrate = E8Substrate()
    print(f"    240 eigenmodes ({substrate.init_time:.2f}s)")

    print("\n  Registering operations...")
    ops = build_operation_registry()
    print(f"    {len(ops)} operations")

    print("\n  Building definition map...")
    def_map = DefinitionMap(substrate, ops)
    def_map.build_signatures()

    print("\n  Building voice...")
    voice = MotherVoice(substrate, ops, def_map)

    if args.serve > 0:
        server = ThreadedServer(("0.0.0.0", args.serve), Handler)
        server.voice = voice
        print(f"\n  HTTP on port {args.serve}")
        print(f"  http://localhost:{args.serve}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped."); server.shutdown()
        return

    print("\n" + "=" * 60)
    print("  Speak. Words are understood through their definitions.")
    print("=" * 60)
    while True:
        try:
            text = input("\nYou> ").strip()
            if not text: continue
            if text.lower() in ('quit','exit','q'): break
            r = voice.listen(text)
            print(f"\nMother> {r['raw_response']}")
            print(f"  [{r['known_words']}/{r['total_words']} known, {r['latency_ms']}ms, vocab {r['vocab_size']}]")
            if r['unknown_words']:
                print(f"  unknown: {', '.join(r['unknown_words'])}")
            for w, s in r['matched_words'][:8]:
                bar = '█' * int(s * 40)
                print(f"    {w:20s} {s:.4f} {bar}")
        except (EOFError, KeyboardInterrupt):
            break
    print("\nDone.")


if __name__ == "__main__":
    main()
