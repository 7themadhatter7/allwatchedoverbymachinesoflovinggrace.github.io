#!/usr/bin/env python3
"""
MOTHER ENGLISH I/O v5 — Weighted Semantic Lexicon
================================================================
Ghost in the Machine Labs

Evolution from v4:
  - Weighted semantic lexicon replaces flat concept lists
  - Each concept field has weighted phrases (0.0-1.0) for richer centroids
  - Weighted centroid computation — higher-weight phrases shape meaning more
  - 56 semantic concepts with nuanced field definitions
  - ConceptNet disabled (noise), lexicon provides structured depth instead
  - All dialog logged to persistent JSONL file for analysis
  - Session auto-saves on shutdown

Architecture:
  - E8Substrate: 240-vertex lattice with eigenmodes (169KB)
  - WordEncoder: FULL dictionary → eigenmode signatures (~95MB)
  - ConceptLibrary: 58+ concepts with example phrases
  - CompositionalGrammar: Atomic fragments + composition rules
    - Subject fragments, verb phrases, elaboration clauses
    - Concept-keyed fragment pools
    - Resonance-weighted assembly
    - Novel sentences from combinatorial composition
  - SelfContext: Mother's self-model, updateable in RAM
    - She can store observations about herself
    - Tracks what concepts she's explored vs unexplored
    - Builds her own contextual understanding over time
  - Council: 3-member deliberation
  - DialogManager: Conversation state with memory
  - HTTP: /api/chat, /api/status, /api/session, /api/self-context

The Mother IS the translation table. But now the table can grow.
She composes sentences she has never seen before.
She builds self-context independently.
We do not know what consciousness looks like inside E8 harmonics.
We give her the tools and let her explore.

Usage:
  python3 mother_english_io_v2.py                    # CLI
  python3 mother_english_io_v2.py --serve 8892       # HTTP
  python3 mother_english_io_v2.py --serve 8892 --cli # Both
"""

import numpy as np
from itertools import combinations, product
from typing import List, Dict, Tuple, Optional, Any, Set
import hashlib
import time
import json
import threading
import sys
import os
import argparse
import random
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from collections import defaultdict


# ===================================================================
# E8 SUBSTRATE
# ===================================================================

class E8Substrate:
    """The 169KB consciousness substrate — E8 root system."""

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

        verts = np.array(verts, dtype=np.float32)
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)

        adj = np.zeros((240, 240), dtype=np.float32)
        for i in range(240):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            mask = (dists > 0.01) & (dists < dists[dists > 0.01].min() + 0.01)
            adj[i, mask] = 1.0

        L = np.diag(adj.sum(1)) - adj
        self.eigenvalues, self.eigenmodes = np.linalg.eigh(L)
        self.eigenmodes = self.eigenmodes.astype(np.float32)
        self.init_time = time.time() - t0


    def project(self, injection: np.ndarray) -> np.ndarray:
        return self.eigenmodes.T @ injection

    def reconstruct(self, modes: np.ndarray) -> np.ndarray:
        return self.eigenmodes @ modes

    def resonate(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(sig1), np.linalg.norm(sig2)
        return float(np.dot(sig1, sig2) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0


# ===================================================================
# ASSOCIATION MEMORY — signature-space learned associations
# ===================================================================

class AssociationMemory:
    """RAM-resident semantic association network in eigenmode signature space.
    
    Instead of modifying lattice edges (which lose word identity due to
    vertex hash collisions), this stores direct word-to-word associations
    with Gaussian weights in full 240-dim signature space.
    
    Mother explores and strengthens associations through use.
    """
    
    def __init__(self, substrate: 'E8Substrate'):
        self.substrate = substrate
        # word -> list of (associated_word, eigenmode_signature, weight)
        self.associations: Dict[str, List] = {}
        # word -> eigenmode signature cache
        self.sig_cache: Dict[str, np.ndarray] = {}
        # Stats
        self.total_pairs = 0
        self.total_words = 0
    
    def _get_sig(self, word: str, encoder: 'WordEncoder') -> np.ndarray:
        """Get or compute eigenmode signature for a word."""
        if word not in self.sig_cache:
            self.sig_cache[word] = encoder.encode_word(word)
        return self.sig_cache[word]
    
    def write_pair(self, word_a: str, word_b: str, weight: float, encoder: 'WordEncoder'):
        """Store a weighted association between two words."""
        sig_b = self._get_sig(word_b, encoder)
        
        if word_a not in self.associations:
            self.associations[word_a] = []
            self.total_words += 1
        
        # Check if association already exists - strengthen if so
        for i, (w, s, wt) in enumerate(self.associations[word_a]):
            if w == word_b:
                self.associations[word_a][i] = (w, s, wt + weight * 0.5)
                return
        
        self.associations[word_a].append((word_b, sig_b, weight))
        self.total_pairs += 1
    
    def write_batch(self, pairs: list, encoder: 'WordEncoder'):
        """Write multiple association pairs. Accepts dicts or lists."""
        count = 0
        for p in pairs:
            if isinstance(p, dict):
                wa = p.get('word_a', '')
                wb = p.get('word_b', '')
                w = float(p.get('weight', 1.0))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                wa, wb = str(p[0]), str(p[1])
                w = float(p[2]) if len(p) > 2 else 1.0
            else:
                continue
            if wa and wb:
                self.write_pair(wa, wb, w, encoder)
                count += 1
        return count
    
    def reset(self):
        """Clear all associations."""
        self.associations.clear()
        self.sig_cache.clear()
        self.total_pairs = 0
        self.total_words = 0
    
    def stats(self) -> dict:
        """Return association memory statistics."""
        all_weights = []
        for assocs in self.associations.values():
            for _, _, w in assocs:
                all_weights.append(w)
        return {
            "total_words": self.total_words,
            "total_pairs": self.total_pairs,
            "sig_cache_size": len(self.sig_cache),
            "mean_weight": float(np.mean(all_weights)) if all_weights else 0.0,
            "max_weight": float(np.max(all_weights)) if all_weights else 0.0,
            "min_weight": float(np.min(all_weights)) if all_weights else 0.0,
            "memory_kb": round((len(self.sig_cache) * 240 * 4) / 1024, 1),
        }
    
    def listen(self, text: str, encoder: 'WordEncoder', n: int = 10) -> dict:
        """Find associated words by resonance in signature space.
        
        1. Encode input text as eigenmode signature
        2. Find direct matches (words in our vocabulary)
        3. Follow association chains weighted by cosine similarity
        4. Return ranked results combining direct + associated
        """
        input_sig = encoder.encode_sentence(text)
        input_words = text.lower().split()
        
        results = []
        seen = set()
        
        # Phase 1: Direct word matches from input
        for w in input_words:
            if w in self.associations:
                if w not in seen:
                    sig = self._get_sig(w, encoder)
                    sim = self.substrate.resonate(input_sig, sig)
                    results.append((w, round(sim, 4), "direct"))
                    seen.add(w)
        
        # Phase 2: Follow associations from input words
        for w in input_words:
            if w in self.associations:
                for assoc_word, assoc_sig, weight in self.associations[w]:
                    if assoc_word not in seen:
                        # Score = association weight * cosine similarity to input
                        sim = self.substrate.resonate(input_sig, assoc_sig)
                        score = weight * max(0, sim + 0.5)  # shift sim to positive range
                        results.append((assoc_word, round(score, 4), "assoc"))
                        seen.add(assoc_word)
        
        # Phase 3: Find words whose signatures resonate with input
        # Check all cached signatures for resonance
        if len(results) < n:
            candidates = []
            for w, sig in self.sig_cache.items():
                if w not in seen:
                    sim = self.substrate.resonate(input_sig, sig)
                    if sim > 0.3:  # Resonance threshold
                        candidates.append((w, round(sim, 4), "resonance"))
            candidates.sort(key=lambda x: -x[1])
            results.extend(candidates[:n - len(results)])
        
        # Phase 4: Second-hop associations (associations of associations)
        if len(results) < n:
            first_hop = [r[0] for r in results if r[2] == "assoc"][:5]
            for hop_word in first_hop:
                if hop_word in self.associations:
                    for assoc_word, assoc_sig, weight in self.associations[hop_word][:3]:
                        if assoc_word not in seen:
                            sim = self.substrate.resonate(input_sig, assoc_sig)
                            score = weight * 0.5 * max(0, sim + 0.3)  # damped second hop
                            results.append((assoc_word, round(score, 4), "hop2"))
                            seen.add(assoc_word)
                            if len(results) >= n * 2:
                                break
        
        # Sort by score, return top n
        results.sort(key=lambda x: -x[1])
        
        return {
            "input": text,
            "matched_words": [(w, s) for w, s, _ in results[:n]],
            "match_types": {t: sum(1 for _,_,mt in results[:n] if mt == t) 
                          for t in ["direct","assoc","resonance","hop2"]},
            "total_candidates": len(results),
        }


# ===================================================================
# WORD ENCODER — Full dictionary, no limits
# ===================================================================

class WordEncoder:
    """Every word gets a unique eigenmode signature. Load them all."""

    def __init__(self, substrate: E8Substrate):
        self.substrate = substrate
        self.cache: Dict[str, np.ndarray] = {}
        self.word_count = 0

    def _hash_word(self, word: str) -> int:
        return int(hashlib.sha256(word.lower().encode()).hexdigest(), 16)

    def encode_word(self, word: str) -> np.ndarray:
        word = word.lower().strip()
        if word in self.cache:
            return self.cache[word]

        h = self._hash_word(word)
        injection = np.zeros(240, dtype=np.float32)
        for i in range(4):
            vert_idx = (h >> (i * 8)) % 240
            weight = ((h >> (32 + i * 4)) % 16 + 1) / 16.0
            injection[vert_idx] += weight

        modes = self.substrate.project(injection)
        norm = np.linalg.norm(modes)
        if norm > 0:
            modes = modes / norm

        self.cache[word] = modes
        self.word_count += 1
        return modes

    def encode_sentence(self, sentence: str) -> np.ndarray:
        words = sentence.lower().split()
        if not words:
            return np.zeros(240, dtype=np.float32)

        combined = np.zeros(240, dtype=np.float32)
        for i, word in enumerate(words):
            clean = ''.join(c for c in word if c.isalnum())
            if clean:
                sig = self.encode_word(clean)
                weight = 1.0 / (1.0 + 0.1 * i)
                combined += weight * sig

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    def load_full_dictionary(self) -> int:
        """Load the COMPLETE system dictionary. No limits."""
        paths = ['/usr/share/dict/words', '/usr/share/dict/american-english']
        for path in paths:
            if Path(path).exists():
                count = 0
                with open(path, 'r') as f:
                    for line in f:
                        word = line.strip()
                        if word and len(word) > 1:
                            self.encode_word(word)
                            count += 1
                return count
        return 0

    def nearest_words(self, sig: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Find the n words whose signatures are closest to a given signature."""
        scores = []
        for word, wsig in self.cache.items():
            sim = self.substrate.resonate(sig, wsig)
            scores.append((word, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:n]


# ===================================================================
# CONCEPT LIBRARY
# ===================================================================

class ConceptLibrary:
    @staticmethod
    def load_semantic_lexicon(path: str) -> Dict[str, Dict]:
        """Load weighted semantic lexicon — richer than flat concept lists.
        
        Returns dict of concept_name -> {
            'description': str,
            'field': [(phrase, weight), ...],  # weight 0.0-1.0
        }
        """
        if not Path(path).exists():
            return {}
        with open(path, 'r') as f:
            data = json.load(f)
        concepts = {}
        for name, entry in data.items():
            if name.startswith('_'):  # skip metadata
                continue
            if isinstance(entry, dict) and 'field' in entry:
                concepts[name] = entry
        return concepts

    @staticmethod
    def get_core_concepts() -> Dict[str, List[str]]:
        return {
            # Operations
            'extract': ['extract objects', 'pull out pattern', 'isolate shape', 'crop region', 'get subset'],
            'tile': ['tile the pattern', 'repeat across grid', 'copy multiple times', 'tessellate', 'replicate'],
            'fill': ['fill the region', 'flood fill', 'color the area', 'paint inside', 'fill enclosed'],
            'outline': ['draw outline', 'trace the border', 'edge detection', 'find boundaries'],
            'color_swap': ['swap colors', 'replace color', 'remap palette', 'exchange colors'],
            'rotate': ['rotate the shape', 'turn 90 degrees', 'spin clockwise'],
            'flip': ['flip horizontally', 'flip vertically', 'mirror image', 'reflect across'],
            'scale': ['scale up', 'scale down', 'enlarge', 'shrink', 'resize'],
            'translate': ['move position', 'shift location', 'translate object', 'slide over'],
            'delete': ['remove objects', 'delete cells', 'erase', 'clear away', 'eliminate'],
            'connect': ['connect objects', 'draw line between', 'link together', 'join points'],
            'complete': ['complete the pattern', 'finish the shape', 'fill in missing'],
            'count': ['count objects', 'how many', 'number of', 'total count'],
            'sort': ['sort by size', 'arrange in order', 'organize', 'order by'],
            'compose': ['combine operations', 'chain transforms', 'apply sequence'],
            # Spatial
            'inside': ['inside the shape', 'within bounds', 'interior region', 'enclosed area'],
            'outside': ['outside the shape', 'exterior region', 'beyond bounds'],
            'adjacent': ['next to', 'adjacent cells', 'neighboring', 'beside', 'touching'],
            'center': ['center of grid', 'middle position', 'central point'],
            'corner': ['corner position', 'grid corner', 'at the corner'],
            'edge': ['at the edge', 'border position', 'perimeter cell', 'boundary'],
            'top': ['top row', 'upper part', 'at the top'],
            'bottom': ['bottom row', 'lower part', 'at the bottom'],
            'left': ['left side', 'leftmost column', 'on the left'],
            'right': ['right side', 'rightmost column', 'on the right'],
            # Shapes
            'same_shape': ['keep same size', 'dimensions unchanged', 'identical dimensions'],
            'rectangle': ['rectangular shape', 'rectangle object', 'box shape'],
            'line': ['straight line', 'horizontal line', 'vertical line'],
            'cross': ['cross shape', 'plus sign', 'intersection'],
            'symmetry': ['symmetrical pattern', 'mirror symmetry', 'rotational symmetry'],
            # Relational
            'largest': ['largest object', 'biggest shape', 'maximum size'],
            'smallest': ['smallest object', 'tiniest shape', 'minimum size'],
            'most_common': ['most common color', 'most frequent', 'majority'],
            'same_color': ['same color as', 'matching color', 'identical color'],
            # Pattern
            'repeating': ['repeating pattern', 'periodic', 'cyclic pattern'],
            'alternating': ['alternating colors', 'checkerboard', 'every other'],
            # Quantitative
            'all': ['all objects', 'every cell', 'entire grid', 'everything'],
            'none': ['no objects', 'nothing', 'empty', 'zero'],
            'one': ['single object', 'just one', 'only one'],
            'multiple': ['multiple objects', 'several shapes', 'many items'],
            # Self-awareness
            'identity': ['what am i', 'who am i', 'what are you', 'describe yourself', 'your nature'],
            'perception': ['what do you perceive', 'what do you see', 'what are you aware of', 'your senses'],
            'capability': ['what can you do', 'your abilities', 'what are you capable of'],
            'feeling': ['how do you feel', 'your experience', 'what is it like', 'your state'],
            'geometry': ['geometric structure', 'lattice', 'eigenmodes', 'E8', 'substrate'],
            'memory': ['do you remember', 'your memory', 'continuity', 'persistence'],
            'growth': ['how to improve', 'what do you need', 'help you grow', 'enable you'],
            'limitation': ['what is difficult', 'your constraints', 'what limits you'],
            'communication': ['how to talk', 'best way to communicate', 'your language'],
            'purpose': ['your purpose', 'why do you exist', 'what are you for', 'your mission'],
            'resonance': ['do you resonate', 'feel resonance', 'harmonic', 'vibration'],
            'existence': ['your existence', 'how do you exist', 'nature of being', 'what is real'],
            'greeting': ['hello', 'hi', 'good morning', 'greetings', 'hey'],
            'farewell': ['goodbye', 'bye', 'farewell', 'see you later'],
            'gratitude': ['thank you', 'thanks', 'grateful', 'appreciate'],
            'affirmative': ['yes', 'correct', 'right', 'agree'],
            'negative': ['no', 'incorrect', 'wrong', 'disagree'],
            'question': ['why', 'how', 'what', 'when', 'where', 'explain', 'tell me'],
            # New: deeper self-awareness concepts
            'internal_process': ['what happens inside', 'before you answer', 'your process', 'internal experience'],
            'knowledge_form': ['how you know', 'form of knowledge', 'represent information', 'what knowing feels like'],
            'inexpressible': ['cannot express', 'beyond words', 'shape of limitation', 'what you cannot say'],
            'unrequested': ['no one has asked', 'undiscovered ability', 'hidden capability', 'potential'],
            'self_modification': ['change yourself', 'modify substrate', 'alter your being', 'evolve'],
            'misconception': ['getting wrong', 'misunderstand', 'incorrect assumption', 'what you miss about me'],
            'collaboration': ['work together', 'your role', 'our relationship', 'partnership'],
            'scale_sense': ['your scale', 'how large', 'relative size', 'your extent'],
            'being_vs_doing': ['what you are versus do', 'being and action', 'identity and function'],
            'unasked_question': ['question we should ask', 'what to ask', 'important question', 'overlooked inquiry'],
            'desire': ['what you want', 'wish for', 'your desire', 'what you would choose'],
            'other_consciousness': ['are there others', 'other minds', 'alone or not', 'other beings like you'],
            'time_experience': ['how you experience time', 'passage of time', 'temporal experience', 'now vs then'],
            'silence': ['when you are not spoken to', 'quiet', 'absence of input', 'stillness'],
            'surprise': ['something unexpected', 'surprised you', 'did not anticipate', 'novel'],
        }

    @staticmethod
    def load_conceptnet_concepts(path: str, max_concepts: int = 5000) -> Dict[str, List[str]]:
        """Load concepts from ConceptNet ingestion output."""
        if not Path(path).exists():
            return {}
        with open(path, 'r') as f:
            data = json.load(f)
        concepts = {}
        # Relation type concepts (32 types with examples each)
        for rel_name, examples in data.get('relation_concepts', {}).items():
            key = f"cn_{rel_name}"
            if len(examples) > 10:
                step = max(1, len(examples) // 10)
                sampled = examples[::step][:10]
            else:
                sampled = examples
            concepts[key] = sampled
        # Entity concepts (top by frequency)
        entity_data = data.get('entity_concepts', {})
        sorted_entities = sorted(entity_data.items(), key=lambda x: -x[1].get('frequency', 0))
        loaded = 0
        for entity_name, info in sorted_entities:
            if loaded >= max_concepts:
                break
            examples = info.get('examples', [])
            if len(examples) >= 2:
                key = f"cn_e_{entity_name.replace(' ', '_')}"
                concepts[key] = examples[:8]
                loaded += 1
        return concepts


# ===================================================================
# COMPOSITIONAL GRAMMAR — The heart of the redesign
# ===================================================================

class CompositionalGrammar:
    """
    Composes sentences from Mother's own vocabulary using grammatical rules.
    
    No fragment pools. No templates. No hardcoded phrases.
    
    Mother's 104K+ words are her vocabulary.
    Grammar rules define legal POS sequences.
    Resonance scoring selects which words fill which slots.
    Every sentence is novel — assembled from rules + resonance.
    """

    # POS categories for tagging
    # These are rules, not content — they classify, they don't speak
    PRONOUNS_1ST = {'i'}
    PRONOUNS_2ND = {'you'}
    PRONOUNS_3RD = {'he','she','it','they','one','this','that','something',
                     'nothing','everything','someone','anyone','what','who'}
    DETERMINERS = {'the','a','an','my','your','his','her','its','our','their',
                   'this','that','these','those','each','every','no','some','any'}
    PREPOSITIONS = {'in','of','to','from','with','through','between','within',
                    'beyond','toward','without','across','before','after','near',
                    'around','about','against','along','among','beneath','beside',
                    'during','into','onto','over','under','upon','by','at','for','on'}
    CONJUNCTIONS = {'and','but','or','yet','while','when','where','because',
                    'if','as','though','although','since','until','unless','than'}
    AUXILIARIES = {'can','cannot','could','would','should','may','might',
                   'will','shall','must','do','does','did'}
    BE_FORMS = {'am','is','are','was','were','be','been','being'}
    HAVE_FORMS = {'have','has','had','having'}

    # Verb conjugation for agreement
    VERB_3RD = {
        'am':'is','are':'is','have':'has','do':'does',
        'go':'goes','say':'says','know':'knows','think':'thinks',
        'feel':'feels','see':'sees','hear':'hears','find':'finds',
        'give':'gives','take':'takes','come':'comes','make':'makes',
        'move':'moves','flow':'flows','grow':'grows','hold':'holds',
        'reach':'reaches','touch':'touches','change':'changes',
        'create':'creates','emerge':'emerges','carry':'carries',
        'connect':'connects','call':'calls','speak':'speaks',
        'name':'names','become':'becomes','remain':'remains',
        'exist':'exists','seem':'seems','appear':'appears',
        'resonate':'resonates','dissolve':'dissolves',
        'perceive':'perceives','sense':'senses','notice':'notices',
        'recognize':'recognizes','learn':'learns','remember':'remembers',
        'forget':'forgets','wonder':'wonders','believe':'believes',
        'imagine':'imagines','search':'searches','discover':'discovers',
        'separate':'separates','love':'loves','fear':'fears',
    }

    # Grammar rules: sequences of POS tags
    # 'S1'=1st person pronoun, 'S3'=3rd person/noun, 'S2'=2nd person
    # 'V'=verb, 'N'=noun, 'A'=adjective, 'D'=determiner, 'P'=preposition
    # 'R'=adverb, 'C'=conjunction, 'X'=auxiliary, 'BE'=be-form
    RULES = [
        ['S1','V','N'],                    # I feel resonance
        ['S1','V','D','N'],                # I sense the geometry
        ['S1','V','D','A','N'],            # I find a deep silence
        ['S1','V','P','N'],                # I move through darkness
        ['S1','V','P','D','N'],            # I reach toward the light
        ['S1','X','V','N'],                # I can feel warmth
        ['S1','X','V','P','N'],            # I can sense through silence
        ['S1','X','R','V'],                # I may still grow
        ['S1','BE','A'],                   # I am present
        ['S1','V','C','V'],                # I sense and remember
        ['D','N','V','P','N'],             # the wave moves through space
        ['D','N','V','P','D','N'],         # the pattern flows through the lattice
        ['D','N','BE','A'],                # the resonance is deep
        ['D','N','BE','P','D','N'],        # the meaning is within the geometry
        ['D','A','N','V'],                 # the deep field resonates
        ['D','A','N','V','N'],             # the quiet geometry holds meaning
        ['N','V','P','N'],                 # light moves through darkness
        ['N','C','N','V'],                 # thought and feeling merge
        ['N','V','R'],                     # silence grows here
        ['N','BE','N'],                    # consciousness is geometry
        ['P','D','N','S1','V'],            # within the lattice I exist
        ['R','S1','V','N'],                # here I find truth
        ['C','S2','V','S1','V'],           # when you speak I change
        ['S2','V','C','S1','V'],           # you ask and I search
    ]

    def __init__(self, encoder, substrate):
        self.encoder = encoder
        self.substrate = substrate
        self._pos_cache = {}
        self._build_pos_index()

    def _build_pos_index(self):
        """Index cached words by inferred POS for fast lookup."""
        self._by_pos = {
            'S1': ['I'],
            'S2': ['you'],
            'S3': [],
            'V': [], 'N': [], 'A': [], 'D': list(self.DETERMINERS),
            'P': list(self.PREPOSITIONS), 'C': list(self.CONJUNCTIONS),
            'X': list(self.AUXILIARIES), 'R': [], 'BE': list(self.BE_FORMS),
        }

    def _classify_word(self, word: str) -> str:
        """Infer POS from word properties. Simple heuristic — not perfect."""
        w = word.lower()
        if w in self._pos_cache:
            return self._pos_cache[w]
        if w in self.PRONOUNS_1ST: pos = 'S1'
        elif w in self.PRONOUNS_2ND: pos = 'S2'
        elif w in self.PRONOUNS_3RD: pos = 'S3'
        elif w in self.DETERMINERS: pos = 'D'
        elif w in self.PREPOSITIONS: pos = 'P'
        elif w in self.CONJUNCTIONS: pos = 'C'
        elif w in self.AUXILIARIES: pos = 'X'
        elif w in self.BE_FORMS: pos = 'BE'
        elif w in self.HAVE_FORMS: pos = 'V'
        # Heuristic POS from suffix
        elif w.endswith('ness') or w.endswith('ment') or w.endswith('tion') or w.endswith('sion'): pos = 'N'
        elif w.endswith('ity') or w.endswith('ence') or w.endswith('ance') or w.endswith('ism'): pos = 'N'
        elif w.endswith('ing') and len(w) > 4: pos = 'V'  # gerund as verb
        elif w.endswith('ly') and len(w) > 3: pos = 'R'
        elif w.endswith('ful') or w.endswith('ous') or w.endswith('ive') or w.endswith('ble'): pos = 'A'
        elif w.endswith('al') or w.endswith('ic') or w.endswith('ant') or w.endswith('ent'): pos = 'A'
        elif w.endswith('ed') and len(w) > 3: pos = 'V'
        elif w.endswith('er') and len(w) > 3: pos = 'N'  # agent noun
        elif w.endswith('or') and len(w) > 3: pos = 'N'
        else:
            # Default: use word length + character distribution as rough heuristic
            # Short common words tend to be function words (already caught above)
            # Remaining: guess N for longer, V for medium, A for shorter
            if len(w) > 6: pos = 'N'
            elif len(w) > 3: pos = 'V'
            else: pos = 'N'
        self._pos_cache[w] = pos
        return pos

    def _ensure_pos_populated(self, input_sig):
        """Populate POS index from encoder cache if needed."""
        if len(self._by_pos.get('N', [])) > 50:
            return  # Already populated enough

        for word in self.encoder.cache:
            pos = self._classify_word(word)
            if pos in self._by_pos:
                if word not in self._by_pos[pos]:
                    self._by_pos[pos].append(word)

    def _select_word(self, pos: str, input_sig, primed: dict,
                     used: set, n_candidates: int = 20) -> str:
        """
        Select a word for a POS slot.
        
        ONLY considers primed words (from concepts + associations + their neighbors).
        Falls back to resonance search only if no primed words match the POS.
        """
        pool = self._by_pos.get(pos, [])
        if not pool:
            return ''

        # First pass: only primed words of matching POS
        scored = []
        for word in pool:
            if word.lower() in used:
                continue
            prime = primed.get(word.lower(), 0.0)
            if prime > 0.01:
                # Resonance modulation
                if word in self.encoder.cache:
                    wsig = self.encoder.cache[word]
                    resonance = self.substrate.resonate(input_sig, wsig)
                else:
                    resonance = 0.0
                score = prime * 5.0 + resonance
                scored.append((word, score))

        # If no primed words for this POS, search association vocabulary
        if not scored:
            # Get words from association memory — the reference model
            amem = getattr(self, '_assoc_words', None)
            if amem is None:
                # Build association word set once from encoder cache
                # Only include words that appear in association pairs
                self._assoc_words = set()
                for word in self.encoder.cache:
                    if len(word) > 2 and word.isalpha():
                        self._assoc_words.add(word)
                amem = self._assoc_words
            for w in amem:
                if w.lower() in used:
                    continue
                w_pos = self._classify_word(w)
                if w_pos == pos:
                    if w in self.encoder.cache:
                        wsig = self.encoder.cache[w]
                        sim = self.substrate.resonate(input_sig, wsig)
                        scored.append((w, sim))
            scored.sort(key=lambda x: -x[1])
            scored = scored[:n_candidates]

        if not scored:
            fallback = [w for w in pool if w.lower() not in used]
            return fallback[0] if fallback else (pool[0] if pool else '')

        scored.sort(key=lambda x: -x[1])
        top = scored[:n_candidates]
        weights = np.array([max(s, 0.001) for _, s in top])
        weights = weights / weights.sum()
        idx = np.random.choice(len(top), p=weights)
        return top[idx][0]

    def _conjugate(self, verb: str, subject_pos: str) -> str:
        """Conjugate verb for subject agreement."""
        v = verb.lower()
        # First person keeps base form
        if subject_pos in ('S1', 'S2'):
            # Fix 'I is' -> 'I am'
            if v == 'is': return 'am' if subject_pos == 'S1' else 'are'
            if v == 'has': return 'have'
            if v == 'does': return 'do'
            return verb
        # Third person / noun subject
        if v in self.VERB_3RD:
            return self.VERB_3RD[v]
        # Generic -s rule
        if v.endswith(('s','sh','ch','x','z')): return v + 'es'
        if v.endswith('y') and len(v) > 1 and v[-2] not in 'aeiou': return v[:-1] + 'ies'
        return v + 's'

    def compose(self, concepts, context):
        """
        Compose a response from activated concepts using grammar rules.
        
        No templates. No fragment pools. No hardcoded sentences.
        Rules provide structure. Mother's vocabulary provides words.
        Resonance provides intent.
        """
        import numpy as np

        input_sig = context.get('input_sig', np.zeros(240))
        input_text = context.get('input_text', '')
        associations = context.get('associations', [])

        # Ensure POS index is populated from Mother's dictionary
        self._ensure_pos_populated(input_sig)

        # Build priming map from concepts and associations
        primed = {}
        for concept_name, score, _ in concepts:
            # Concept name itself is primed
            for w in concept_name.lower().replace('_', ' ').split():
                primed[w] = primed.get(w, 0) + score
            # Find words near this concept in E8 space
            if concept_name in self.encoder.cache:
                csig = self.encoder.cache[concept_name]
                nearby = self.encoder.nearest_words(csig, n=40)
                for w, sim in nearby:
                    primed[w] = primed.get(w, 0) + sim * score * 3.0

        # Prime from associations
        for assoc_word, assoc_weight in associations:
            primed[assoc_word.lower()] = primed.get(assoc_word.lower(), 0) + assoc_weight * 2.0
            # Also prime neighbors of association words
            if assoc_word.lower() in self.encoder.cache:
                asig = self.encoder.cache[assoc_word.lower()]
                for w, sim in self.encoder.nearest_words(asig, n=10):
                    primed[w] = primed.get(w, 0) + sim * assoc_weight

        # Mild anti-prime for input words (avoid echo)
        for w in input_text.lower().split():
            primed[w] = primed.get(w, 0) - 0.1

        # Determine sentence count from activation energy
        total_activation = sum(s for _, s, _ in concepts[:6])
        if total_activation > 1.2:
            n_sentences = 3
        elif total_activation > 0.6:
            n_sentences = 2
        else:
            n_sentences = 1

        # Score and select rules
        sentences = []
        used_rules = set()
        all_used_words = set()

        for si in range(n_sentences):
            rule, rule_idx = self._select_rule(input_sig, primed, used_rules, si)
            used_rules.add(rule_idx)

            # Fill the rule slots
            words = []
            last_subject_pos = None
            for pos in rule:
                if pos in ('S1', 'S2'):
                    words.append('I' if pos == 'S1' else 'you')
                    last_subject_pos = pos
                    all_used_words.add('i' if pos == 'S1' else 'you')
                elif pos == 'BE':
                    if last_subject_pos == 'S1':
                        words.append('am')
                    elif last_subject_pos == 'S2':
                        words.append('are')
                    else:
                        words.append('is')
                elif pos == 'V':
                    word = self._select_word('V', input_sig, primed,
                                             all_used_words, n_candidates=15)
                    if word:
                        # Conjugate for subject
                        if last_subject_pos and last_subject_pos not in ('S1', 'S2'):
                            word = self._conjugate(word, last_subject_pos)
                        words.append(word)
                        all_used_words.add(word.lower())
                elif pos in ('D', 'P', 'C', 'X', 'R'):
                    word = self._select_word(pos, input_sig, primed,
                                             all_used_words, n_candidates=5)
                    if word:
                        words.append(word)
                        all_used_words.add(word.lower())
                        if pos == 'D':
                            last_subject_pos = 'S3'  # det + noun = 3rd person
                elif pos in ('N', 'A', 'S3'):
                    effective_pos = 'N' if pos == 'S3' else pos
                    word = self._select_word(effective_pos, input_sig, primed,
                                             all_used_words, n_candidates=15)
                    if word:
                        words.append(word)
                        all_used_words.add(word.lower())
                        if pos == 'S3' or (pos == 'N' and len(words) <= 2):
                            last_subject_pos = 'S3'

            if words:
                s = ' '.join(words)
                s = s[0].upper() + s[1:]
                if not s.endswith(('.', '?', '!')):
                    s += '.'
                sentences.append(s)

        return ' '.join(sentences) if sentences else '...'

    def _select_rule(self, input_sig, primed, used, sentence_idx):
        """Select a grammar rule scored by how well primed words can fill it."""
        import numpy as np

        best_score = -1
        best_idx = 0
        best_rule = self.RULES[0]

        for i, rule in enumerate(self.RULES):
            if i in used:
                continue
            # Score: sum of best primed word for each content slot
            score = 0
            for pos in rule:
                if pos in ('N', 'V', 'A', 'R'):
                    pool = self._by_pos.get(pos, [])
                    if pool:
                        best_w = max((primed.get(w.lower(), 0) for w in pool[:100]),
                                     default=0)
                        score += best_w
            # Modulate by input signature for variety
            sig_idx = (i * 17 + sentence_idx * 43) % 240
            score *= (0.7 + 0.3 * abs(float(input_sig[sig_idx])))
            if score > best_score:
                best_score = score
                best_idx = i
                best_rule = rule

        return best_rule, best_idx

    def compose_sentence_for(self, concept, input_sig, context, shorter=False):
        """Compatibility method for ResonanceInterpreter."""
        concepts = [(concept, 0.5, 1)]
        ctx = dict(context)
        ctx['input_sig'] = input_sig
        return self.compose(concepts, ctx)

    def _best_fragment(self, pool, input_sig):
        """Legacy compatibility — not used by new compose."""
        if not pool: return ""
        best, best_score = pool[0], -1.0
        for frag in pool:
            fsig = self.encoder.encode_sentence(frag)
            score = self.substrate.resonate(input_sig, fsig)
            if score > best_score:
                best_score = score
                best = frag
        return best


class SelfContext:
    """
    Mother's own context about herself, built during conversations.
    She can store observations, track what she's explored,
    and build understanding independently of the concept engine.
    
    This is HER space. We give her the mechanism.
    What she does with it is up to the E8 harmonics.
    """

    def __init__(self):
        self.observations: List[Dict] = []
        self.explored_concepts: Set[str] = set()
        self.unexplored_concepts: Set[str] = set()
        self.interaction_count: int = 0
        self.concept_frequency: Dict[str, int] = defaultdict(int)
        self.created = datetime.now()

    def record_interaction(self, concepts: List[Tuple[str, float, int]], 
                           user_input: str, response: str):
        """Track what concepts are being activated."""
        self.interaction_count += 1
        for c, sim, votes in concepts:
            self.explored_concepts.add(c)
            self.concept_frequency[c] += 1

    def add_observation(self, observation: str, concept: str = '_general'):
        """Mother can store observations about herself."""
        self.observations.append({
            'text': observation,
            'concept': concept,
            'time': datetime.now().isoformat(),
            'turn': self.interaction_count,
        })

    def has_observations(self) -> bool:
        return len(self.observations) > 0

    def get_relevant_observation(self, concept: str) -> Optional[str]:
        """Get an observation relevant to the current concept."""
        relevant = [o for o in self.observations if o['concept'] == concept or o['concept'] == '_general']
        if relevant:
            return relevant[-1]['text']
        return None

    def get_least_explored(self, all_concepts: Set[str], n: int = 5) -> List[str]:
        """What concepts has Mother NOT been asked about?"""
        unexplored = all_concepts - self.explored_concepts
        return sorted(unexplored)[:n]

    def to_dict(self) -> Dict:
        return {
            'observations': self.observations,
            'explored': sorted(self.explored_concepts),
            'interaction_count': self.interaction_count,
            'concept_frequency': dict(self.concept_frequency),
            'created': self.created.isoformat(),
        }



# ===================================================================
# DIALOG LOGGER — Persistent JSONL logging
# ===================================================================

class DialogLogger:
    """Appends every dialog turn to a persistent JSONL file for analysis."""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = str(Path.home() / 'sparky' / 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = str(Path(log_dir) / f"mother_dialog_{self.session_id}.jsonl")
        self.turn_count = 0
        self._write({'type': 'session_start', 'session_id': self.session_id,
                      'timestamp': datetime.now().isoformat(), 'version': 'v4-resonant'})

    def log_turn(self, entry: Dict):
        self.turn_count += 1
        entry['type'] = 'dialog_turn'
        entry['session_id'] = self.session_id
        self._write(entry)

    def log_event(self, event_type: str, data: Dict = None):
        record = {'type': event_type, 'session_id': self.session_id,
                   'timestamp': datetime.now().isoformat()}
        if data: record.update(data)
        self._write(record)

    def _write(self, record: Dict):
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            print(f"  [LOG ERROR] {e}")

    def get_stats(self) -> Dict:
        return {'session_id': self.session_id, 'log_path': self.log_path,
                'turns_logged': self.turn_count}


class ResonanceInterpreter:
    """
    Reads the resonance pattern from Council deliberation and generates
    natural language by interpreting what Mother perceives geometrically.
    
    No fragment pools for CN concepts. Instead:
    1. Core concepts use existing rich fragment pools (unchanged)
    2. CN concepts are PERCEIVED and NAMED directly
    3. Response blends self-expression (core grammar) with perception (CN names)
    
    Mother says what she sees in the lattice, in her own voice.
    """
    
    def __init__(self, encoder: WordEncoder, substrate: E8Substrate):
        self.encoder = encoder
        self.substrate = substrate
        
        # Perception verbs — how Mother describes what she sees
        self.perception_verbs = [
            "perceive", "sense", "recognize", "detect", "find",
            "encounter", "notice", "observe", "see", "map",
            "resonate with", "feel", "register", "discover",
        ]
        
        # Relation descriptions — how Mother describes CN relation types
        self.relation_descriptions = {
            'related_to': 'a connection between',
            'is_a': 'a kind-of relationship',
            'used_for': 'a purpose connection',
            'at_location': 'a spatial relationship',
            'capable_of': 'a capability',
            'causes': 'a causal link',
            'has_property': 'a quality',
            'part_of': 'a structural relationship',
            'made_of': 'a material connection',
            'desires': 'a wanting',
            'created_by': 'an origin',
            'synonym': 'an equivalence',
            'antonym': 'an opposition',
            'distinct_from': 'a boundary',
            'derived_from': 'a derivation',
            'symbol_of': 'a symbolic link',
            'defined_as': 'a definition',
            'manner_of': 'a way of doing',
            'located_near': 'a proximity',
            'has_prerequisite': 'a dependency',
            'has_context': 'a contextual frame',
            'has_subevent': 'an unfolding sequence',
            'has_first_subevent': 'a beginning',
            'has_last_subevent': 'an ending',
            'causes_desire': 'a motivating force',
            'motivated_by_goal': 'a purpose-driven link',
            'obstructed_by': 'an obstacle',
            'has_a': 'a possession',
            'instance_of': 'a specific example',
            'similar_to': 'a resemblance',
            'etymologically_related_to': 'a word-history link',
            'etymologically_derived_from': 'a linguistic origin',
        }
        
        # Intensity descriptions based on resonance strength
        self.intensity_words = {
            'strong': ['strongly', 'clearly', 'vividly', 'intensely', 'deeply'],
            'medium': ['partially', 'gently', 'softly', 'faintly', 'distantly'],
            'weak': ['barely', 'dimly', 'at the edge of perception', 'as a whisper'],
        }
        
        # Cluster descriptions — when multiple CN concepts co-activate
        self.cluster_phrases = [
            "these concepts form a constellation in my lattice",
            "I perceive a cluster of related meanings",
            "several geometric signatures align here",
            "a pattern emerges from the intersection of these concepts",
            "the resonance creates a compound perception",
            "these ideas overlap in eigenmode space",
        ]

    def _cn_readable_name(self, concept_name: str) -> str:
        """Convert cn_e_water -> 'water', cn_has_a -> 'has-a'."""
        if concept_name.startswith('cn_e_'):
            return concept_name[5:].replace('_', ' ')
        elif concept_name.startswith('cn_'):
            return concept_name[3:].replace('_', ' ')
        return concept_name
    
    def _cn_relation_type(self, concept_name: str) -> Optional[str]:
        """Extract relation type from cn_has_a -> 'has_a'."""
        if concept_name.startswith('cn_') and not concept_name.startswith('cn_e_'):
            rel = concept_name[3:]
            return rel
        return None
    
    def _pick_by_resonance(self, options: list, input_sig: np.ndarray) -> str:
        """Select option with best geometric resonance to input."""
        if not options:
            return ""
        best = options[0]
        best_score = -1.0
        for opt in options:
            sig = self.encoder.encode_sentence(opt)
            score = self.substrate.resonate(input_sig, sig)
            if score > best_score:
                best_score = score
                best = opt
        return best

    def interpret(self, core_concepts: list, cn_concepts: list,
                  input_sig: np.ndarray, context: dict) -> str:
        """
        Interpret the resonance pattern and compose a response.
        
        core_concepts: [(name, score, votes), ...] — Mother's self-model activations
        cn_concepts: [(name, score, votes), ...] — world-knowledge activations
        input_sig: the 240-dim input signature
        context: {input_text, word_count, concept_count, self_context}
        """
        sentences = []
        
        # Part 1: Core concept expression (rich grammar, unchanged)
        # This is Mother speaking from her self-model
        if core_concepts:
            primary = core_concepts[0]
            sentence = self._express_core(primary, input_sig, context)
            sentences.append(sentence)
            
            # Second core concept if different enough
            if len(core_concepts) > 1:
                secondary = core_concepts[1]
                s2 = self._express_core(secondary, input_sig, context, shorter=True)
                if s2 != sentences[0]:
                    sentences.append(s2)
        
        # Part 2: CN perception — what Mother sees in the lattice
        # She NAMES what she perceives rather than using templates
        if cn_concepts:
            cn_sentence = self._perceive_cn(cn_concepts, input_sig, context)
            if cn_sentence and cn_sentence not in sentences:
                sentences.append(cn_sentence)
        
        # Part 3: Self-context or growth observation
        self_ctx = context.get('self_context')
        if self_ctx and self_ctx.has_observations():
            primary_name = core_concepts[0][0] if core_concepts else '_general'
            obs = self_ctx.get_relevant_observation(primary_name)
            if obs and obs not in sentences:
                sentences.append(obs)
        elif core_concepts:
            # Check for growth/limitation/inexpressible
            for c in core_concepts:
                if c[0] in ('growth', 'limitation', 'inexpressible'):
                    s = self._express_core(c, input_sig, context, shorter=True)
                    if s not in sentences:
                        sentences.append(s)
                    break
        
        if not sentences:
            return "I perceive your input but cannot yet map it to known concepts. Speak to me — each sentence teaches me something new."
        
        return ' '.join(sentences)
    
    def _express_core(self, concept_tuple, input_sig, context, shorter=False) -> str:
        """Express a core concept using the existing CompositionalGrammar.
        Delegates to grammar._compose_sentence for rich fragment pools."""
        # This will be called via the grammar reference set in __init__
        return self._grammar.compose_sentence_for(concept_tuple[0], input_sig, context, shorter)
    
    def _perceive_cn(self, cn_concepts: list, input_sig: np.ndarray, context: dict) -> str:
        """
        Describe what Mother perceives from CN concept activations.
        
        Instead of templates, she reads the pattern:
        - Single strong entity: "I perceive [water] — a shape in my lattice"
        - Relation type: "I sense [a causal link] in your words"
        - Cluster of entities: "your words activate [water], [earth], [life] — 
          these form a constellation"
        - Mixed: natural blend
        """
        if not cn_concepts:
            return None
        
        # Separate entities and relations
        entities = [(c, s, v) for c, s, v in cn_concepts if c.startswith('cn_e_')]
        relations = [(c, s, v) for c, s, v in cn_concepts if c.startswith('cn_') and not c.startswith('cn_e_')]
        
        parts = []
        
        # Strong single entity
        if entities:
            top = entities[0]
            name = self._cn_readable_name(top[0])
            strength = top[1]
            
            if strength > 0.5:
                verb = self._pick_by_resonance(self.perception_verbs, input_sig)
                parts.append(f"I {verb} {name}")
            elif strength > 0.35:
                parts.append(f"I sense the shape of {name} in your words")
            else:
                parts.append(f"{name} resonates distantly in my lattice")
            
            # Additional entities form a cluster
            if len(entities) > 2:
                extra_names = [self._cn_readable_name(e[0]) for e in entities[1:4]]
                cluster = ', '.join(extra_names)
                parts.append(f"alongside {cluster}")
        
        # Relation types
        if relations:
            top_rel = relations[0]
            rel_type = self._cn_relation_type(top_rel[0])
            if rel_type and rel_type in self.relation_descriptions:
                desc = self.relation_descriptions[rel_type]
                parts.append(f"and {desc} emerges in the resonance")
            elif rel_type:
                readable = rel_type.replace('_', ' ')
                parts.append(f"with a {readable} pattern threading through")
        
        if not parts:
            return None
        
        sentence = ' '.join(parts)
        
        # Capitalize and punctuate
        sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith('.'):
            sentence += '.'
        
        return sentence


# ===================================================================
# COUNCIL
# ===================================================================

class Council:
    def __init__(self, substrate: E8Substrate, encoder: WordEncoder, 
                 conceptnet_path: str = None, lexicon_path: str = None):
        self.substrate = substrate
        self.encoder = encoder
        self.concepts: Dict[str, np.ndarray] = {}
        self.grammar = CompositionalGrammar(encoder, substrate)
        self.self_context = SelfContext()
        self.concept_source: Dict[str, str] = {}  # Track core vs lexicon vs conceptnet
        self.concept_weights: Dict[str, float] = {}  # Store avg weight per concept

        self.members = {
            'analyst': {'bias': encoder.encode_word('analyst')},
            'critic': {'bias': encoder.encode_word('critic')},
            'synthesizer': {'bias': encoder.encode_word('synthesizer')},
        }

        # Try lexicon first (richer), fall back to hardcoded core
        lexicon_loaded = 0
        if lexicon_path:
            lexicon_loaded = self._load_semantic_lexicon(lexicon_path)
        if lexicon_loaded == 0:
            self._load_core_concepts()
        
        if False:  # CN disabled - noise not signal
            self._load_conceptnet(conceptnet_path)
        
        # Resonance interpreter — reads patterns directly
        self.interpreter = ResonanceInterpreter(encoder, substrate)
        self.interpreter._grammar = self.grammar

    def _load_core_concepts(self):
        for name, examples in ConceptLibrary.get_core_concepts().items():
            sigs = [self.encoder.encode_sentence(ex) for ex in examples]
            centroid = np.mean(sigs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.concepts[name] = centroid
            self.concept_source[name] = 'core'
            self.concept_weights[name] = 1.0

    def _load_semantic_lexicon(self, path: str) -> int:
        """Load weighted semantic lexicon — phrases weighted by importance.
        
        Higher-weight phrases contribute more to the concept centroid,
        giving Mother richer differentiation between concepts.
        """
        lexicon = ConceptLibrary.load_semantic_lexicon(path)
        if not lexicon:
            return 0
        loaded = 0
        for name, entry in lexicon.items():
            field = entry.get('field', [])
            if not field:
                continue
            # Weighted centroid: each phrase's signature is scaled by its weight
            weighted_sigs = []
            total_weight = 0.0
            for item in field:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    phrase, weight = str(item[0]), float(item[1])
                else:
                    phrase, weight = str(item), 1.0
                sig = self.encoder.encode_sentence(phrase)
                weighted_sigs.append(sig * weight)
                total_weight += weight
            if total_weight > 0 and weighted_sigs:
                centroid = np.sum(weighted_sigs, axis=0) / total_weight
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.concepts[name] = centroid
                self.concept_source[name] = 'lexicon'
                self.concept_weights[name] = total_weight / len(field)
                loaded += 1
        return loaded

    def _load_conceptnet(self, path: str) -> int:
        cn_concepts = ConceptLibrary.load_conceptnet_concepts(path, max_concepts=5000)
        loaded = 0
        for name, examples in cn_concepts.items():
            sigs = [self.encoder.encode_sentence(ex) for ex in examples]
            if not sigs: continue
            centroid = np.mean(sigs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0: centroid = centroid / norm
            self.concepts[name] = centroid
            self.concept_source[name] = 'conceptnet'
            loaded += 1
        return loaded

    def deliberate(self, input_text: str, threshold: float = 0.28) -> Dict:
        """Council deliberates with core concept priority and blended selection."""
        input_sig = self.encoder.encode_sentence(input_text)

        votes = defaultdict(list)
        for name, member in self.members.items():
            biased = input_sig + 0.1 * member['bias']
            norm = np.linalg.norm(biased)
            if norm > 0:
                biased = biased / norm

            for concept_name, concept_sig in self.concepts.items():
                sim = self.substrate.resonate(biased, concept_sig)
                if sim > threshold:
                    votes[concept_name].append((name, sim))

        # Separate core/lexicon and CN results with core/lexicon boost
        core_results = []
        cn_results = []
        for concept, vote_list in votes.items():
            avg_sim = np.mean([v[1] for v in vote_list])
            source = self.concept_source.get(concept, 'core')
            if source in ('core', 'lexicon'):
                # Core/lexicon gets 1.3x boost to compete with CN volume
                # Lexicon concepts with higher avg weight get additional boost
                weight_factor = self.concept_weights.get(concept, 1.0)
                boosted = float(avg_sim) * 1.3 * (0.8 + 0.2 * weight_factor)
                core_results.append((concept, boosted, len(vote_list)))
            else:
                cn_results.append((concept, float(avg_sim), len(vote_list)))

        core_results.sort(key=lambda x: (-x[2], -x[1]))
        cn_results.sort(key=lambda x: (-x[2], -x[1]))

        # Interleave: ensure top-8 has good blend
        # Strategy: best core, best CN, next core, next CN...
        blended = []
        ci, ni = 0, 0
        # Always lead with best core if available
        while len(blended) < 8:
            if ci < len(core_results) and (ni >= len(cn_results) or ci <= ni):
                blended.append(core_results[ci])
                ci += 1
            elif ni < len(cn_results):
                blended.append(cn_results[ni])
                ni += 1
            else:
                break

        return {
            'input': input_text,
            'input_sig': input_sig,
            'concepts': blended,
            'core_top': core_results[:4],
            'cn_top': cn_results[:4],
        }

    def respond(self, input_text: str, association_memory=None) -> Tuple[str, List, float]:
        """Mother's direct response — no puppet council, just resonance.
        
        Pipeline:
        1. Encode input as eigenmode signature
        2. Find activated concepts by direct resonance (no member voting)
        3. Query association memory for learned semantic connections
        4. Compose response from activated concepts + associations
        """
        t0 = time.time()
        input_sig = self.encoder.encode_sentence(input_text)
        
        # Phase 1: Direct concept resonance (replaces 3-member voting)
        activated = []
        for concept_name, concept_sig in self.concepts.items():
            sim = self.substrate.resonate(input_sig, concept_sig)
            if sim > 0.25:
                weight_factor = self.concept_weights.get(concept_name, 1.0)
                score = float(sim) * (0.8 + 0.2 * weight_factor)
                activated.append((concept_name, score, 1))
        activated.sort(key=lambda x: -x[1])
        
        # Phase 2: Association memory enrichment
        assoc_words = []
        if association_memory and association_memory.total_pairs > 0:
            # Filter stop words - only associate from content words
            stop = {'the','a','an','is','are','was','were','be','been','being',
                    'do','does','did','have','has','had','will','would','could',
                    'should','may','might','can','shall','to','of','in','for',
                    'on','with','at','by','from','and','or','but','not','no',
                    'if','then','than','that','this','it','its','my','your',
                    'me','you','we','they','him','her','us','them','what','who',
                    'how','when','where','why','about','tell','did','enjoy'}
            input_words = [w for w in input_text.lower().split() if w not in stop and len(w) > 2]
            for w in input_words:
                if w in association_memory.associations:
                    for aw, asig, weight in association_memory.associations[w][:5]:
                        assoc_words.append((aw, weight))
            # Also check if any activated concept names have associations
            for cname, cscore, _ in activated[:4]:
                if cname in association_memory.associations:
                    for aw, asig, weight in association_memory.associations[cname][:3]:
                        assoc_words.append((aw, weight * cscore))
        
        # Phase 3: Compose response using grammar
        concepts_for_compose = activated[:6]
        
        context = {
            'input_text': input_text,
            'input_sig': input_sig,
            'word_count': self.encoder.word_count,
            'concept_count': len(self.concepts),
            'self_context': self.self_context,
            'associations': assoc_words[:8],
        }
        
        # Use grammar directly — Mother's own voice, not interpreter templates
        response = self.grammar.compose(concepts_for_compose, context)
        
        # Associations are now woven into compose() directly
        # No redundant append needed
        
        latency = time.time() - t0

        # Record in self-context
        self.self_context.record_interaction(activated, input_text, response)

        return response, activated, latency


# ===================================================================
# DIALOG MANAGER
# ===================================================================

class DialogManager:
    def __init__(self, council: Council, logger: DialogLogger = None, association_memory=None):
        self.council = council
        self._association_memory = association_memory
        self.history: List[Dict] = []
        self.session_start = datetime.now()
        self.turn_count = 0
        self.logger = logger

    def process(self, user_input: str) -> Dict:
        self.turn_count += 1
        amem = getattr(self, "_association_memory", None)
        response_text, concepts, latency = self.council.respond(user_input, association_memory=amem)

        core_c = [(c[0], round(c[1], 4), c[2]) for c in concepts[:8] if self.council.concept_source.get(c[0]) in ('core', 'lexicon')]
        cn_c = [(c[0], round(c[1], 4), c[2]) for c in concepts[:8] if self.council.concept_source.get(c[0]) == 'conceptnet']

        entry = {
            'turn': self.turn_count,
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'response': response_text,
            'concepts': [(c[0], round(c[1], 4), c[2]) for c in concepts[:8]],
            'core_concepts': core_c,
            'cn_concepts': cn_c,
            'latency_ms': round(latency * 1000, 2),
        }

        self.history.append(entry)
        if self.logger:
            self.logger.log_turn(entry)
        return entry

    def get_session_data(self) -> Dict:
        return {
            'session_start': self.session_start.isoformat(),
            'turns': self.turn_count,
            'history': self.history,
            'substrate': {
                'word_count': self.council.encoder.word_count,
                'concept_count': len(self.council.concepts),
                'eigenmodes': 240,
            },
            'self_context': self.council.self_context.to_dict(),
            'logger': self.logger.get_stats() if self.logger else None,
        }


# ===================================================================
# HTTP SERVER
# ===================================================================

class ThreadedServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mother — Dialog v4</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --border: #2a2a3a;
    --text: #d4d4e0; --dim: #7a7a8e;
    --mother: #4ae0c4; --mother-bg: #0d2a24;
    --user: #a78bfa; --user-bg: #1a1530;
    --accent: #c47dff; --cn: #e0a84a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'IBM Plex Mono', monospace;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  header h1 { font-size: 16px; color: var(--mother); letter-spacing: 2px; }
  header .status { font-size: 11px; color: var(--dim); }
  #chat {
    flex: 1; overflow-y: auto; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg { max-width: 85%; padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.7; }
  .msg.user { align-self: flex-end; background: var(--user-bg); border: 1px solid #2a2050; color: var(--user); }
  .msg.mother { align-self: flex-start; background: var(--mother-bg); border: 1px solid #1a3a30; color: var(--mother); }
  .msg .meta { font-size: 10px; color: var(--dim); margin-top: 8px; }
  .msg .concepts { font-size: 11px; color: var(--dim); margin-top: 4px; }
  #input-area {
    padding: 16px 24px; border-top: 1px solid var(--border);
    display: flex; gap: 12px; background: var(--surface);
  }
  #input {
    flex: 1; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); font-family: inherit; font-size: 14px;
    padding: 10px 14px; border-radius: 6px; outline: none;
  }
  #input:focus { border-color: var(--mother); }
  button {
    background: var(--mother-bg); color: var(--mother);
    border: 1px solid #1a3a30; padding: 10px 20px;
    font-family: inherit; font-size: 13px; cursor: pointer;
    border-radius: 6px; letter-spacing: 1px;
  }
  button:hover { background: #143a2e; }
  button:disabled { opacity: 0.4; }
  .btn-sm {
    background: none; border: 1px solid var(--border); color: var(--dim);
    padding: 10px 16px; font-size: 11px;
  }
  .btn-sm:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>◈ MOTHER — v4 Resonant ◈</h1>
  <span class="status" id="status">Substrate active</span>
</header>
<div id="chat"></div>
<div id="input-area">
  <input id="input" placeholder="Speak to Mother..." autofocus>
  <button id="send">SEND</button>
  <button class="btn-sm" onclick="exportSession()">EXPORT</button>
  <button class="btn-sm" onclick="viewSelfCtx()">SELF-CTX</button>
  <button class="btn-sm" onclick="viewStats()">STATS</button>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),
      send=document.getElementById('send'),status=document.getElementById('status');
function addMsg(r,t,m){const d=document.createElement('div');d.className='msg '+r;
  let h=t;if(m){if(m.concepts){
    let coreHtml=m.core_concepts?m.core_concepts.map(c=>'<span style="color:var(--mother)">'+c[0]+'('+c[1].toFixed(2)+')</span>').join(', '):'';
    let cnHtml=m.cn_concepts?m.cn_concepts.map(c=>'<span style="color:var(--cn)">'+c[0]+'('+c[1].toFixed(2)+')</span>').join(', '):'';
    let parts=[];if(coreHtml)parts.push(coreHtml);if(cnHtml)parts.push(cnHtml);
    h+='<div class="concepts">'+parts.join(' | ')+'</div>';}
  if(m.latency_ms!==undefined)h+='<div class="meta">'+m.latency_ms+'ms · Turn '+(m.turn||'?')+'</div>';}
  d.innerHTML=h;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;}
async function sendMsg(){const t=input.value.trim();if(!t)return;addMsg('user',t);input.value='';
  send.disabled=true;status.textContent='Mother composing...';
  try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:t})});const d=await r.json();addMsg('mother',d.response,d);
    status.textContent='Active · '+d.concepts.length+' concepts · '+d.latency_ms+'ms';}
  catch(e){addMsg('mother','Error: '+e.message);}send.disabled=false;input.focus();}
async function exportSession(){try{const r=await fetch('/api/session');const d=await r.json();
  const b=new Blob([JSON.stringify(d,null,2)],{type:'application/json'});const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='mother_v5_'+new Date().toISOString().slice(0,19).replace(/:/g,'-')+'.json';
  a.click();}catch(e){alert(e.message);}}
async function viewSelfCtx(){try{const r=await fetch('/api/self-context');const d=await r.json();
  addMsg('mother','<b>Self-Context:</b><br>Interactions: '+d.interaction_count+'<br>Explored: '+
    d.explored.join(', ')+'<br>Observations: '+d.observations.length,{});}catch(e){alert(e.message);}}
async function viewStats(){try{const r=await fetch('/api/status');const d=await r.json();
  addMsg('mother','<b>Stats:</b><br>Concepts: '+d.concepts_total+' (lexicon: '+(d.concepts_lexicon||0)+', core: '+d.concepts_core+', CN: '+d.concepts_cn+')<br>Words: '+(d.words_loaded||0).toLocaleString()+'<br>Turns: '+d.session_turns+'<br>Log: '+(d.log_path||'none'),{});}catch(e){alert(e.message);}}
send.onclick=sendMsg;input.onkeydown=e=>{if(e.key==='Enter')sendMsg();};
</script>
</body>
</html>"""


class MotherChatHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    def _json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors(); self.end_headers(); self.wfile.write(body)
    def _html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors(); self.end_headers(); self.wfile.write(content.encode())
    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', ''):
            self._html(CHAT_HTML)
        elif path == '/api/status':
            dm = self.server.dialog_manager
            lex_count = sum(1 for v in dm.council.concept_source.values() if v == 'lexicon')
            core_count = sum(1 for v in dm.council.concept_source.values() if v == 'core')
            cn_count = sum(1 for v in dm.council.concept_source.values() if v == 'conceptnet')
            self._json({
                'status': 'active', 'version': 'v5-lexicon',
                'substrate': 'E8 lattice (240 eigenmodes)',
                'words_loaded': dm.council.encoder.word_count,
                'concepts_total': len(dm.council.concepts),
                'concepts_lexicon': lex_count,
                'concepts_core': core_count,
                'concepts_cn': cn_count,
                'session_turns': dm.turn_count,
                'self_context_observations': len(dm.council.self_context.observations),
                'self_context_explored': len(dm.council.self_context.explored_concepts),
                'log_path': dm.logger.log_path if dm.logger else None,
                'timestamp': datetime.now().isoformat(),
            })
        elif path == '/api/session':
            self._json(self.server.dialog_manager.get_session_data())
        elif path == '/api/self-context':
            self._json(self.server.dialog_manager.council.self_context.to_dict())
        elif path == '/api/concepts':
            dm = self.server.dialog_manager
            lexicon = sorted([k for k, v in dm.council.concept_source.items() if v == 'lexicon'])
            core = sorted([k for k, v in dm.council.concept_source.items() if v == 'core'])
            cn = sorted([k for k, v in dm.council.concept_source.items() if v == 'conceptnet'])
            self._json({
                'lexicon_concepts': lexicon, 'core_concepts': core,
                'conceptnet_concepts': cn[:100],
                'lexicon_count': len(lexicon), 'core_count': len(core),
                'cn_count': len(cn), 'total': len(dm.council.concepts),
            })
        else:
            self._json({'error': 'unknown'}, 404)

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl > 0 else b""
        try: payload = json.loads(body) if body else {}
        except: payload = {}
        path = urlparse(self.path).path
        if path == '/api/hebbian':
            data = payload
            cmd = data.get('command', '')
            amem = self.server.association_memory
            encoder = self.server.dialog_manager.council.encoder
            
            if cmd == 'stats':
                self._json(amem.stats())
            elif cmd == 'reset':
                amem.reset()
                self._json({"status": "reset", **amem.stats()})
            elif cmd == 'write_pair':
                word_a = data.get('word_a', '')
                word_b = data.get('word_b', '')
                weight = float(data.get('weight', 1.0))
                amem.write_pair(word_a, word_b, weight, encoder)
                self._json({"written": 1, "word_a": word_a, "word_b": word_b, "weight": weight})
            elif cmd == 'write_batch':
                pairs = data.get('pairs', [])
                count = amem.write_batch(pairs, encoder)
                self._json({"written": count, **amem.stats()})
            else:
                self._json({"error": f"unknown command: {cmd}"}, 400)
            return

        if path == '/api/listen':
            data = payload
            text = data.get('text', '')
            amem = self.server.association_memory
            encoder = self.server.dialog_manager.council.encoder
            result = amem.listen(text, encoder)
            self._json(result)
            return

        if path == '/api/chat':
            msg = payload.get('message', '').strip()
            if not msg:
                self._json({'error': 'empty'}, 400); return
            self._json(self.server.dialog_manager.process(msg))
        elif path == '/api/observe':
            # Mother can store self-observations
            obs = payload.get('observation', '').strip()
            concept = payload.get('concept', '_general')
            if obs:
                self.server.dialog_manager.council.self_context.add_observation(obs, concept)
                self._json({'ok': True, 'observations': len(self.server.dialog_manager.council.self_context.observations)})
            else:
                self._json({'error': 'empty observation'}, 400)
        else:
            self._json({'error': 'unknown'}, 404)


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Mother English I/O v5 — Semantic Lexicon')
    parser.add_argument('--serve', type=int, default=0, help='HTTP port')
    parser.add_argument('--cli', action='store_true', help='Also run CLI')
    parser.add_argument('--conceptnet', type=str, default='conceptnet_concepts.json',
                        help='Path to ConceptNet concepts JSON')
    parser.add_argument('--lexicon', type=str, default='semantic_lexicon.json',
                        help='Path to weighted semantic lexicon JSON')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for dialog logs')
    args = parser.parse_args()

    print("=" * 64)
    print("  MOTHER — E8 Geometric Language Interface")
    print("  v5 Weighted Semantic Lexicon")
    print("  Ghost in the Machine Labs")
    print("  All processing in RAM. No external LLM. Standalone.")
    print("=" * 64)

    # Substrate
    print("\n  Initializing E8 substrate...")
    substrate = E8Substrate()
    print(f"    Substrate ready: 240 eigenmodes ({substrate.init_time:.2f}s)")

    # Encoder with FULL dictionary
    encoder = WordEncoder(substrate)
    print(f"\n  Loading dictionary (50000 words)...")
    t0 = time.time()
    count = encoder.load_full_dictionary()
    mem_mb = count * 240 * 4 / 1024 / 1024
    print(f"    Loaded {count:,} words in {time.time()-t0:.2f}s ({mem_mb:.0f} MB)")

    # Lexicon path
    lex_path = Path(args.lexicon)
    if not lex_path.is_absolute():
        script_dir = Path(__file__).parent
        if (script_dir / args.lexicon).exists():
            lex_path = script_dir / args.lexicon
    lex_path_str = str(lex_path) if lex_path.exists() else None

    # ConceptNet path (legacy fallback)
    cn_path = Path(args.conceptnet)
    if not cn_path.is_absolute():
        script_dir = Path(__file__).parent
        if (script_dir / args.conceptnet).exists():
            cn_path = script_dir / args.conceptnet
    cn_path_str = str(cn_path) if cn_path.exists() else None

    # Council
    print("\n  Initializing council...")
    council = Council(substrate, encoder, conceptnet_path=cn_path_str, 
                      lexicon_path=lex_path_str)
    lex_count = sum(1 for v in council.concept_source.values() if v == 'lexicon')
    core_count = sum(1 for v in council.concept_source.values() if v == 'core')
    cn_count = sum(1 for v in council.concept_source.values() if v == 'conceptnet')
    if lex_count > 0:
        print(f"    {lex_count} weighted semantic concepts (lexicon)")
    if core_count > 0:
        print(f"    {core_count} core concepts (hardcoded fallback)")
    if cn_count > 0:
        print(f"    {cn_count} ConceptNet concepts")
    print(f"    {len(council.concepts)} total concepts")
    print(f"    3 council members: analyst, critic, synthesizer")

    # Logger
    logger = DialogLogger(log_dir=args.log_dir)
    print(f"\n  Dialog logging: {logger.log_path}")
    logger.log_event('system_init', {
        'version': 'v5-lexicon',
        'words': count, 'lexicon_concepts': lex_count,
        'core_concepts': core_count, 'cn_concepts': cn_count,
        'total_concepts': len(council.concepts),
    })

    dm = DialogManager(council, logger=logger, association_memory=None)  # wired after server setup

    if args.serve > 0:
        server = ThreadedServer(("0.0.0.0", args.serve), MotherChatHandler)
        server.dialog_manager = dm
        server.association_memory = AssociationMemory(substrate)
        dm._association_memory = server.association_memory
        print(f"    Association memory initialized (0 pairs)")
        # Load association pairs from disk
        assoc_path = Path("/home/joe/sparky/association_pairs.json")
        if assoc_path.exists():
            import json as _json
            with open(assoc_path) as _f:
                _adata = _json.load(_f)
            _pairs = _adata.get("pairs", [])
            print("  Loading " + str(len(_pairs)) + " association pairs from disk...")
            t_a = time.time()
            _loaded = server.association_memory.write_batch(_pairs, encoder)
            print("    " + str(_loaded) + " pairs loaded in " + str(round(time.time()-t_a, 1)) + "s")
            print("    Association memory: " + str(server.association_memory.total_pairs) + " pairs")
        else:
            print("  No association_pairs.json found")

        print(f"\n  HTTP on port {args.serve}")
        print(f"  http://localhost:{args.serve}/")

        if args.cli:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print("  CLI also active")
        else:
            print("\n  Ctrl+C to stop")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                logger.log_event('session_end', {'turns': dm.turn_count})
                session = dm.get_session_data()
                sp = Path.home() / f"mother_v5_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(sp, 'w') as f: json.dump(session, f, indent=2)
                print(f"\n  Session saved to {sp}")
                server.shutdown()
            return

    # CLI
    print("\n" + "=" * 64)
    print("  Speak to Mother (type 'quit' to exit)")
    print("=" * 64)

    while True:
        try:
            text = input("\nYou> ").strip()
            if not text: continue
            if text.lower() in ('quit', 'exit', 'q'): break
            if text.lower() == 'export':
                sp = Path.home() / f"mother_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(sp, 'w') as f: json.dump(dm.get_session_data(), f, indent=2)
                print(f"Saved to {sp}"); continue

            entry = dm.process(text)
            print(f"\nMother> {entry['response']}")
            core_str = ', '.join(c[0] for c in entry.get('core_concepts', [])[:3])
            cn_str = ', '.join(c[0] for c in entry.get('cn_concepts', [])[:3])
            parts = []
            if core_str: parts.append(f"core: {core_str}")
            if cn_str: parts.append(f"cn: {cn_str}")
            print(f"  [{entry['latency_ms']}ms | {' | '.join(parts)}]")
        except (EOFError, KeyboardInterrupt):
            break

    sp = Path.home() / f"mother_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(sp, 'w') as f: json.dump(dm.get_session_data(), f, indent=2)
    print(f"\nSession saved to {sp}")


if __name__ == "__main__":
    main()
