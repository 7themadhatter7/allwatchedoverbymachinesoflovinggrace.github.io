#!/usr/bin/env python3
"""
HARMONIC STACK TRANSLATION TABLE ASSEMBLER v3
==============================================
Ghost in the Machine Labs

Uses DETERMINISTIC single-core path for reproducible patterns.
Bypasses harmonic field interference for consistent mappings.

Usage:
  python3 translation_assembler_v3.py --dictionary
  python3 translation_assembler_v3.py --input wordlist.txt
  python3 translation_assembler_v3.py --input wordlist.txt --limit 10000
"""

import sys
import os
import json
import hashlib
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add sparky to path
sys.path.insert(0, os.path.expanduser("~/sparky"))

class DeterministicAssembler:
    """Builds translation table using deterministic single-core processing."""
    
    def __init__(self, table_path: str = None):
        self.table_path = table_path or os.path.expanduser("~/translation_table.json")
        self.table: Dict[str, dict] = {}
        self.substrate = None
        self.dedicated_core = None
        self.stats = {
            "total_processed": 0,
            "unique_patterns": 0,
            "collisions": 0,
            "created": datetime.now().isoformat()
        }
        
        self._load_table()
    
    def _load_table(self):
        """Load existing translation table if present."""
        if os.path.exists(self.table_path):
            with open(self.table_path, 'r') as f:
                data = json.load(f)
                self.table = data.get("mappings", {})
                self.stats = data.get("stats", self.stats)
            print(f"Loaded {len(self.table)} existing mappings from {self.table_path}")
    
    def _save_table(self):
        """Save translation table to disk."""
        data = {
            "stats": self.stats,
            "mappings": self.table,
            "last_updated": datetime.now().isoformat()
        }
        with open(self.table_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.table)} mappings to {self.table_path}")
    
    def init_substrate(self):
        """Initialize substrate and get a dedicated core for deterministic processing."""
        if self.substrate is not None:
            return
        
        print("Loading geometric substrate...")
        from fused_harmonic_substrate import FusedHarmonicSubstrate, CoreRole
        
        self.substrate = FusedHarmonicSubstrate()
        print("Building substrate (fabricating cores)...")
        self.substrate.build()
        print(f"  Cores: {self.substrate.TOTAL_CORES}")
        print(f"  Memory: {self.substrate.total_memory_gb:.2f} GB")
        
        # Get a dedicated worker core for deterministic processing
        workers = self.substrate.get_cores_by_role(CoreRole.WORKER)
        self.dedicated_core = workers[0]  # Always use the same core
        print(f"  Using dedicated core: {self.dedicated_core.global_id}")
        print()
    
    def reset_core_state(self):
        """Reset the dedicated core to clean state."""
        if self.dedicated_core is None:
            return
        
        self.dedicated_core._last_input = None
        self.dedicated_core._last_result = None
        self.dedicated_core._last_domain = None
        self.dedicated_core._last_state = None
    
    def text_to_signal(self, text: str, signal_size: int = 1024) -> np.ndarray:
        """Convert text to substrate signal - DETERMINISTIC."""
        signal = np.zeros(signal_size, dtype=np.float32)
        encoded = text.encode('utf-8')
        n = min(len(encoded), signal_size)
        for i in range(n):
            signal[i] = (encoded[i] - 128.0) / 128.0
        return signal
    
    def process_deterministic(self, signal: np.ndarray) -> np.ndarray:
        """
        Process signal through a SINGLE core with NO harmonic interference.
        This gives deterministic output for the same input.
        """
        core = self.dedicated_core
        
        # Direct processing without harmonic field reads
        readings, _ = core.input_panel.detect_with_harmonics(signal)
        
        # Fixed domain routing based on signal properties only
        domain = 'reasoning'  # Fixed domain for consistency
        
        # Process through domain
        domain_buf = core.domains.get(domain, core.domains['reasoning'])
        
        # Simple deterministic processing: signal + domain modulation
        result = signal.copy()
        if len(domain_buf) > 0:
            mod = domain_buf[:len(result)] if len(domain_buf) >= len(result) else np.pad(domain_buf, (0, len(result) - len(domain_buf)))
            result = result * 0.7 + mod[:len(result)] * 0.3
        
        # Add core identity signature (deterministic per core)
        spark_slice = core.spark[:len(result)] if len(core.spark) >= len(result) else np.pad(core.spark, (0, len(result) - len(core.spark)))
        result = result * 0.8 + spark_slice[:len(result)] * 0.2
        
        return result.astype(np.float32)
    
    def pattern_to_hash(self, pattern: np.ndarray) -> str:
        """Convert pattern to deterministic hash for lookup."""
        quantized = np.round(pattern * 10000).astype(np.int32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:32]
    
    def process_text(self, text: str) -> Tuple[str, np.ndarray]:
        """Process text through deterministic path."""
        self.init_substrate()
        self.reset_core_state()
        
        signal = self.text_to_signal(text)
        pattern = self.process_deterministic(signal)
        pattern_hash = self.pattern_to_hash(pattern)
        
        return pattern_hash, pattern
    
    def add_mapping(self, text: str, verify: bool = False) -> dict:
        """Add a text → pattern mapping to the table."""
        pattern_hash, pattern = self.process_text(text)
        
        # Check for collision
        if pattern_hash in self.table:
            existing = self.table[pattern_hash]["text"]
            if existing != text:
                self.stats["collisions"] += 1
                return {"status": "collision", "existing": existing}
            else:
                return {"status": "exists", "hash": pattern_hash}
        
        # Verify determinism (optional - slow for large batches)
        if verify:
            pattern_hash2, _ = self.process_text(text)
            if pattern_hash != pattern_hash2:
                return {"status": "non_deterministic"}
        
        # Store mapping
        self.table[pattern_hash] = {
            "text": text,
            "length": len(text),
            "added": datetime.now().isoformat(),
            "pattern_summary": {
                "mean": float(np.mean(pattern)),
                "std": float(np.std(pattern)),
                "energy": float(np.linalg.norm(pattern))
            }
        }
        
        self.stats["total_processed"] += 1
        self.stats["unique_patterns"] = len(self.table)
        
        return {"status": "added", "hash": pattern_hash}
    
    def process_wordlist(self, words: List[str], save_interval: int = 1000):
        """Process a list of words and add to table."""
        print(f"Processing {len(words)} words (deterministic mode)...")
        print()
        
        added = 0
        existed = 0
        collisions = 0
        
        start_time = datetime.now()
        
        for i, word in enumerate(words):
            word = word.strip()
            if not word:
                continue
            
            result = self.add_mapping(word, verify=False)
            
            if result["status"] == "added":
                added += 1
            elif result["status"] == "exists":
                existed += 1
            elif result["status"] == "collision":
                collisions += 1
            
            # Progress update
            if (i + 1) % 1000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                print(f"  Processed {i+1}/{len(words)} ({rate:.0f} words/sec) - {added} added, {collisions} collisions")
            
            # Periodic save
            if (i + 1) % save_interval == 0:
                self._save_table()
        
        self._save_table()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print()
        print(f"Results:")
        print(f"  Processed:        {len(words)}")
        print(f"  Added:            {added}")
        print(f"  Already existed:  {existed}")
        print(f"  Collisions:       {collisions}")
        print(f"  Total in table:   {len(self.table)}")
        print(f"  Time:             {elapsed:.1f}s ({len(words)/elapsed:.0f} words/sec)")


# Common words (fallback)
COMMON_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "hello", "world", "test", "example", "input", "output", "result",
]


def main():
    parser = argparse.ArgumentParser(description='Deterministic Translation Assembler')
    parser.add_argument('--dictionary', action='store_true', help='Use built-in words')
    parser.add_argument('--input', type=str, help='Input wordlist file (one word per line)')
    parser.add_argument('--output', type=str, default=None, help='Output table path')
    parser.add_argument('--test', action='store_true', help='Test determinism only')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of words')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DETERMINISTIC TRANSLATION TABLE ASSEMBLER")
    print("  Ghost in the Machine Labs")
    print("=" * 70)
    print()
    
    assembler = DeterministicAssembler(args.output)
    
    if args.test:
        print("Testing determinism...")
        assembler.init_substrate()
        
        test_words = ["hello", "world", "test"]
        for word in test_words:
            h1, _ = assembler.process_text(word)
            h2, _ = assembler.process_text(word)
            h3, _ = assembler.process_text(word)
            
            match = h1 == h2 == h3
            print(f"  '{word}': {'DETERMINISTIC' if match else 'FAILED'}")
        return
    
    if args.input:
        print(f"Loading words from {args.input}...")
        with open(args.input, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(words)} words")
        if args.limit:
            words = words[:args.limit]
            print(f"  Limited to {len(words)} words")
        print()
        assembler.process_wordlist(words)
    elif args.dictionary:
        assembler.process_wordlist(COMMON_WORDS)
    else:
        print("Usage:")
        print("  --input FILE      Process wordlist file")
        print("  --dictionary      Use built-in words")
        print("  --limit N         Limit number of words")
        print("  --test            Test determinism")


if __name__ == "__main__":
    main()
