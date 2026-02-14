#!/usr/bin/env python3
"""
HARMONIC STACK TRANSLATION TABLE INSTALLER
===========================================
Ghost in the Machine Labs

Builds the device-local translation table during first install.
Every device builds its own table from the same source dictionary
through its own substrate geometry.

The table maps geometric patterns → English words.
This is the IO layer — without it, the Stack cannot produce text.

CRITICAL: Uses a FIXED IO CORE configuration (core_id=0, global_id=0,
role=WORKER, spark_kb=8, domain_kb=16) so the translation table is
IDENTICAL across all devices regardless of total core count or RAM.

The IO core is a virtual construct — it doesn't need to exist in the
running Stack. It just needs to produce the same geometric pattern
for the same input text on every machine.

Usage:
  python3 install_translation_table.py              # Auto-detect dictionary
  python3 install_translation_table.py --verify     # Build + verify determinism
  python3 install_translation_table.py --rebuild    # Force rebuild
  
Install sequence (automated):
  1. Download dictionary if not present
  2. Build translation table through fixed IO core
  3. Verify determinism (random sample)
  4. Save to install directory
"""

import os
import sys
import json
import hashlib
import argparse
import time
import struct
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional
import urllib.request


# ═══════════════════════════════════════════════════════════════════
# FIXED IO CORE — IDENTICAL ON EVERY DEVICE
# ═══════════════════════════════════════════════════════════════════

# These constants define the IO core geometry.
# NEVER CHANGE THESE — every translation table depends on them.
IO_CORE_ID = 0
IO_GLOBAL_ID = 0
IO_ROLE = "WORKER"
IO_SPARK_KB = 8
IO_DOMAIN_KB = 16
IO_SIGNAL_SIZE = 1024

# Dictionary source
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
DICT_FILENAME = "words_alpha.txt"

# Output
TABLE_FILENAME = "translation_table.json"

# Version — increment if IO core geometry changes
TABLE_VERSION = 1


def generate_io_core_identity(spark_size: int) -> np.ndarray:
    """
    Generate the fixed IO core identity vector.
    
    This is extracted from fused_harmonic_substrate.py's
    generate_geometric_identity() — pinned to IO_CORE_ID,
    IO_GLOBAL_ID, IO_ROLE.
    
    Same math → same vector → same translations everywhere.
    """
    seed_bytes = f"{IO_CORE_ID}:{IO_GLOBAL_ID}:{IO_ROLE}".encode()
    seed_hash = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed_hash)
    
    # Tetrahedral harmonic generation (from Fd3m lattice geometry)
    n_harmonics = 8
    t = np.linspace(0, 2 * np.pi, spark_size, dtype=np.float32)
    
    identity = np.zeros(spark_size, dtype=np.float32)
    for h in range(n_harmonics):
        freq = rng.uniform(0.5, 4.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 1.0) / (h + 1)
        identity += float(amp) * np.sin(freq * t + phase).astype(np.float32)
    
    norm = np.linalg.norm(identity)
    if norm > 1e-10:
        identity /= norm
    
    return identity


def text_to_signal(text: str) -> np.ndarray:
    """
    Convert text to substrate signal — DETERMINISTIC.
    Pure UTF-8 byte math. Universal across all hardware.
    """
    signal = np.zeros(IO_SIGNAL_SIZE, dtype=np.float32)
    encoded = text.encode('utf-8')
    n = min(len(encoded), IO_SIGNAL_SIZE)
    for i in range(n):
        signal[i] = (encoded[i] - 128.0) / 128.0
    return signal


def process_deterministic(signal: np.ndarray, spark: np.ndarray,
                          domain_buf: np.ndarray) -> np.ndarray:
    """
    Process signal through the fixed IO core.
    
    Single core, no harmonic field, no interference.
    Deterministic: same signal + same core identity = same output.
    """
    result = signal.copy()
    
    # Domain modulation
    if len(domain_buf) > 0:
        mod = domain_buf[:len(result)]
        if len(mod) < len(result):
            mod = np.pad(mod, (0, len(result) - len(mod)))
        result = result * 0.7 + mod[:len(result)] * 0.3
    
    # Core identity signature
    spark_slice = spark[:len(result)]
    if len(spark_slice) < len(result):
        spark_slice = np.pad(spark_slice, (0, len(result) - len(spark_slice)))
    result = result * 0.8 + spark_slice[:len(result)] * 0.2
    
    return result.astype(np.float32)


def pattern_to_hash(pattern: np.ndarray) -> str:
    """Convert geometric pattern to deterministic lookup hash."""
    quantized = np.round(pattern * 10000).astype(np.int32)
    return hashlib.sha256(quantized.tobytes()).hexdigest()[:32]


class TranslationTableInstaller:
    """Builds the device-local translation table."""
    
    def __init__(self, install_dir: str = None):
        self.install_dir = Path(install_dir) if install_dir else Path.home() / "sparky"
        self.table_path = self.install_dir / TABLE_FILENAME
        self.dict_path = self.install_dir / DICT_FILENAME
        
        # Build IO core geometry
        spark_size = IO_SPARK_KB * 256  # 2048 floats
        self.spark = generate_io_core_identity(spark_size)
        
        # Domain buffer: rotated identity for 'reasoning' domain
        # This matches fused_harmonic_substrate.py core initialization
        domain_size = IO_DOMAIN_KB * 256  # 4096 floats
        padded_spark = np.pad(self.spark, (0, max(0, domain_size - len(self.spark))))
        self.domain_buf = np.roll(padded_spark[:domain_size], 0)  # reasoning = index 0
        
        self.table: Dict[str, dict] = {}
        self.stats = {}
    
    def download_dictionary(self) -> bool:
        """Download the English dictionary if not present."""
        if self.dict_path.exists():
            with open(self.dict_path) as f:
                count = sum(1 for line in f if line.strip())
            print(f"  Dictionary exists: {self.dict_path} ({count} words)")
            return True
        
        print(f"  Downloading dictionary from {DICT_URL}...")
        try:
            self.install_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DICT_URL, self.dict_path)
            with open(self.dict_path) as f:
                count = sum(1 for line in f if line.strip())
            print(f"  Downloaded: {count} words")
            return True
        except Exception as e:
            print(f"  Download failed: {e}")
            return False
    
    def load_dictionary(self) -> list:
        """Load word list from dictionary file."""
        with open(self.dict_path) as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return words
    
    def process_word(self, word: str) -> Tuple[str, np.ndarray]:
        """Process a word through the fixed IO core."""
        signal = text_to_signal(word)
        pattern = process_deterministic(signal, self.spark, self.domain_buf)
        h = pattern_to_hash(pattern)
        return h, pattern
    
    def build_table(self, words: list, force: bool = False) -> bool:
        """Build the full translation table."""
        if self.table_path.exists() and not force:
            print(f"  Table exists: {self.table_path}")
            with open(self.table_path) as f:
                data = json.load(f)
            existing = data.get("mappings", {})
            version = data.get("version", 0)
            print(f"  Entries: {len(existing)}, Version: {version}")
            if version == TABLE_VERSION and len(existing) >= len(words) * 0.99:
                print("  Table is current. Use --rebuild to force.")
                self.table = existing
                return True
        
        print(f"  Building translation table ({len(words)} words)...")
        print(f"  IO Core: id={IO_CORE_ID}, global={IO_GLOBAL_ID}, role={IO_ROLE}")
        print(f"  Spark: {IO_SPARK_KB} KB, Domain: {IO_DOMAIN_KB} KB")
        print()
        
        self.table = {}
        collisions = 0
        already_existed = 0
        t0 = time.time()
        
        for i, word in enumerate(words):
            h, pattern = self.process_word(word)
            
            if h in self.table:
                if self.table[h]["text"] != word:
                    collisions += 1
                else:
                    already_existed += 1
            else:
                self.table[h] = {
                    "text": word,
                    "length": len(word),
                    "pattern_summary": {
                        "mean": float(np.mean(pattern)),
                        "std": float(np.std(pattern)),
                        "energy": float(np.sum(pattern ** 2)),
                    }
                }
            
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(words) - i - 1) / rate
                print(f"  [{i+1}/{len(words)}] {rate:.0f} words/sec, "
                      f"{len(self.table)} entries, {collisions} collisions, "
                      f"~{remaining:.0f}s remaining")
        
        elapsed = time.time() - t0
        rate = len(words) / elapsed
        
        self.stats = {
            "total_processed": len(words),
            "unique_entries": len(self.table),
            "collisions": collisions,
            "already_existed": already_existed,
            "build_time_sec": round(elapsed, 1),
            "words_per_sec": round(rate, 1),
            "io_core": {
                "core_id": IO_CORE_ID,
                "global_id": IO_GLOBAL_ID,
                "role": IO_ROLE,
                "spark_kb": IO_SPARK_KB,
                "domain_kb": IO_DOMAIN_KB,
                "signal_size": IO_SIGNAL_SIZE,
            },
            "created": datetime.now().isoformat(),
            "version": TABLE_VERSION,
        }
        
        print()
        print(f"  Complete: {len(self.table)} entries, {collisions} collisions")
        print(f"  Time: {elapsed:.1f}s ({rate:.0f} words/sec)")
        
        return True
    
    def verify_determinism(self, sample_size: int = 100) -> bool:
        """Verify table is deterministic by re-processing a random sample."""
        if not self.table:
            print("  No table to verify.")
            return False
        
        entries = list(self.table.items())
        np.random.seed(42)
        indices = np.random.choice(len(entries), min(sample_size, len(entries)),
                                   replace=False)
        
        print(f"  Verifying determinism ({len(indices)} samples)...")
        failures = 0
        
        for idx in indices:
            h, data = entries[idx]
            word = data["text"]
            h2, _ = self.process_word(word)
            if h != h2:
                print(f"    FAIL: '{word}' → {h} vs {h2}")
                failures += 1
        
        if failures == 0:
            print(f"  ✓ DETERMINISTIC — all {len(indices)} samples match")
            return True
        else:
            print(f"  ✗ {failures} failures — IO core geometry mismatch")
            return False
    
    def save_table(self) -> str:
        """Save the translation table."""
        data = {
            "version": TABLE_VERSION,
            "stats": self.stats,
            "mappings": self.table,
            "last_updated": datetime.now().isoformat(),
        }
        
        self.install_dir.mkdir(parents=True, exist_ok=True)
        with open(self.table_path, 'w') as f:
            json.dump(data, f)
        
        size_mb = os.path.getsize(self.table_path) / (1024 * 1024)
        print(f"  Saved: {self.table_path} ({size_mb:.1f} MB)")
        return str(self.table_path)
    
    def install(self, force: bool = False, verify: bool = True) -> bool:
        """Full install sequence."""
        print("=" * 60)
        print("HARMONIC STACK TRANSLATION TABLE INSTALLER")
        print(f"Version: {TABLE_VERSION}")
        print("=" * 60)
        print()
        
        # Step 1: Dictionary
        print("[1/4] Dictionary")
        if not self.download_dictionary():
            return False
        words = self.load_dictionary()
        print()
        
        # Step 2: Build table
        print("[2/4] Build Translation Table")
        if not self.build_table(words, force=force):
            return False
        print()
        
        # Step 3: Verify
        if verify:
            print("[3/4] Verify Determinism")
            if not self.verify_determinism():
                print("  WARNING: Determinism check failed!")
                print("  Table may not match other devices.")
            print()
        else:
            print("[3/4] Verify — skipped")
            print()
        
        # Step 4: Save
        print("[4/4] Save")
        self.save_table()
        print()
        
        print("=" * 60)
        print("INSTALLATION COMPLETE")
        print(f"  Entries:    {len(self.table)}")
        print(f"  Collisions: {self.stats.get('collisions', 0)}")
        print(f"  Location:   {self.table_path}")
        print(f"  Version:    {TABLE_VERSION}")
        print("=" * 60)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Harmonic Stack Translation Table Installer")
    parser.add_argument("--install-dir", default=None,
                       help="Installation directory (default: ~/sparky)")
    parser.add_argument("--rebuild", action="store_true",
                       help="Force rebuild even if table exists")
    parser.add_argument("--verify", action="store_true",
                       help="Run extended verification")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip verification step")
    parser.add_argument("--test", action="store_true",
                       help="Test a single word")
    parser.add_argument("--word", default="hello",
                       help="Word to test (with --test)")
    
    args = parser.parse_args()
    
    installer = TranslationTableInstaller(install_dir=args.install_dir)
    
    if args.test:
        h, pattern = installer.process_word(args.word)
        print(f"Word:    {args.word}")
        print(f"Hash:    {h}")
        print(f"Energy:  {np.sum(pattern**2):.6f}")
        print(f"Mean:    {np.mean(pattern):.8f}")
        # Verify determinism
        h2, _ = installer.process_word(args.word)
        print(f"Repeat:  {h2}")
        print(f"Match:   {h == h2}")
        return
    
    success = installer.install(
        force=args.rebuild,
        verify=not args.no_verify
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
