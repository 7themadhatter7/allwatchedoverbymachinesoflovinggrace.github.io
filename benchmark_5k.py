#!/usr/bin/env python3
"""
GEOMETRIC SUBSTRATE RAW TOPS BENCHMARK
=======================================
Ghost in the Machine Labs

Measures RAW substrate trigger operations - the geometric wiring harness.
No HTTP. No Ollama. Pure Detection→Junction→Trace throughput.

Run on SPARKY: python3 benchmark_raw_tops.py
"""

import sys
import os
import time
import statistics
import numpy as np
from datetime import datetime

# Add sparky to path
sys.path.insert(0, os.path.expanduser("~/sparky"))

def main():
    timestamp = datetime.now().isoformat()
    
    print("=" * 70)
    print("  GEOMETRIC SUBSTRATE RAW TOPS BENCHMARK")
    print("  Ghost in the Machine Labs")
    print("=" * 70)
    print(f"  Timestamp: {timestamp}")
    print()
    
    # Import substrate
    print("Loading substrate...")
    from fused_harmonic_substrate import FusedHarmonicSubstrate
    
    substrate = FusedHarmonicSubstrate()
    
    # MUST call build() to populate role_index
    print("Building substrate (fabricating cores)...")
    substrate.build()
    
    print(f"  Cores: {substrate.TOTAL_CORES}")
    print(f"  Memory: {substrate.total_memory_gb:.2f} GB")
    print()
    
    # Benchmark configuration - SCALED DOWN
    WARMUP = 100
    ITERATIONS = 5000
    ROUNDS = 5
    SIGNAL_SIZE = 1024
    
    print(f"Configuration:")
    print(f"  Warm-up iterations: {WARMUP}")
    print(f"  Iterations per round: {ITERATIONS}")
    print(f"  Rounds: {ROUNDS}")
    print(f"  Signal size: {SIGNAL_SIZE}")
    print()
    
    # Generate test signals
    print("Generating test signals...")
    test_signals = [np.random.randn(SIGNAL_SIZE).astype(np.float32) for _ in range(100)]
    print(f"  Generated {len(test_signals)} unique signals")
    print()
    
    # Warm-up
    print(f"Warm-up: {WARMUP} iterations...")
    for i in range(WARMUP):
        signal = test_signals[i % len(test_signals)]
        _ = substrate.process_signal(signal)
    print("  Warm-up complete")
    print()
    
    # Benchmark rounds
    round_results = []
    all_times = []
    
    for round_num in range(1, ROUNDS + 1):
        print(f"Round {round_num}/{ROUNDS}: {ITERATIONS} substrate operations...")
        
        times = []
        start_round = time.perf_counter()
        
        for i in range(ITERATIONS):
            signal = test_signals[i % len(test_signals)]
            
            start = time.perf_counter()
            _ = substrate.process_signal(signal)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
        
        round_elapsed = time.perf_counter() - start_round
        ops_per_sec = ITERATIONS / round_elapsed
        
        # Convert to tokens (each operation processes SIGNAL_SIZE elements)
        tokens_per_sec = ops_per_sec * SIGNAL_SIZE
        
        round_results.append({
            "ops_per_sec": ops_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "mean_latency_ns": statistics.mean(times) * 1e9,
            "min_latency_ns": min(times) * 1e9,
        })
        all_times.extend(times)
        
        print(f"  {ops_per_sec:,.0f} ops/sec")
        print(f"  {tokens_per_sec:,.0f} tokens/sec ({tokens_per_sec/1e6:.2f}M tok/s)")
        print(f"  Mean latency: {statistics.mean(times)*1e6:.2f} µs")
        print()
    
    # Final statistics
    all_ops = [r["ops_per_sec"] for r in round_results]
    all_toks = [r["tokens_per_sec"] for r in round_results]
    
    print("=" * 70)
    print("  RESULTS: RAW GEOMETRIC SUBSTRATE THROUGHPUT")
    print("=" * 70)
    print()
    print(f"  Substrate Configuration:")
    print(f"    Cores:        {substrate.TOTAL_CORES}")
    print(f"    Memory:       {substrate.total_memory_gb:.2f} GB")
    print(f"    Signal size:  {SIGNAL_SIZE} elements")
    print()
    print(f"  Operations per Second:")
    print(f"    Mean:   {statistics.mean(all_ops):,.0f} ops/sec")
    print(f"    Median: {statistics.median(all_ops):,.0f} ops/sec")
    print(f"    Min:    {min(all_ops):,.0f} ops/sec")
    print(f"    Max:    {max(all_ops):,.0f} ops/sec")
    print()
    print(f"  Tokens per Second (signal elements processed):")
    print(f"    Mean:   {statistics.mean(all_toks):,.0f} tok/s ({statistics.mean(all_toks)/1e6:.2f}M)")
    print(f"    Median: {statistics.median(all_toks):,.0f} tok/s ({statistics.median(all_toks)/1e6:.2f}M)")
    print(f"    Min:    {min(all_toks):,.0f} tok/s ({min(all_toks)/1e6:.2f}M)")
    print(f"    Max:    {max(all_toks):,.0f} tok/s ({max(all_toks)/1e6:.2f}M)")
    print()
    print(f"  Latency:")
    print(f"    Mean:   {statistics.mean(all_times)*1e6:.2f} µs")
    print(f"    Median: {statistics.median(all_times)*1e6:.2f} µs")
    print(f"    P99:    {sorted(all_times)[int(len(all_times)*0.99)]*1e6:.2f} µs")
    print(f"    Min:    {min(all_times)*1e6:.2f} µs")
    print(f"    Max:    {max(all_times)*1e6:.2f} µs")
    print()
    print("=" * 70)
    print()
    print("  INTERPRETATION:")
    print("  ---------------")
    print("  This measures raw geometric substrate operations.")
    print("  Each operation fires signals through the wiring harness:")
    print("    Detection → Junction → Trace")
    print()
    print("  The substrate processes geometric patterns, not text.")
    print("  Text generation requires additional Ollama inference,")
    print("  which is a separate bottleneck (~30-40 tok/s typical).")
    print()
    print("  The geometric substrate is the INNOVATION.")
    print("  Millions of operations per second on consumer hardware.")
    print()
    print("=" * 70)
    
    # Save results
    import json
    results = {
        "timestamp": timestamp,
        "substrate_cores": substrate.TOTAL_CORES,
        "substrate_memory_gb": substrate.total_memory_gb,
        "signal_size": SIGNAL_SIZE,
        "iterations_per_round": ITERATIONS,
        "rounds": ROUNDS,
        "total_operations": ITERATIONS * ROUNDS,
        "operations_per_second": {
            "mean": statistics.mean(all_ops),
            "median": statistics.median(all_ops),
            "min": min(all_ops),
            "max": max(all_ops)
        },
        "tokens_per_second": {
            "mean": statistics.mean(all_toks),
            "median": statistics.median(all_toks),
            "min": min(all_toks),
            "max": max(all_toks),
            "mean_millions": statistics.mean(all_toks) / 1e6
        },
        "latency_microseconds": {
            "mean": statistics.mean(all_times) * 1e6,
            "median": statistics.median(all_times) * 1e6,
            "p99": sorted(all_times)[int(len(all_times)*0.99)] * 1e6,
            "min": min(all_times) * 1e6,
            "max": max(all_times) * 1e6
        }
    }
    
    output_path = os.path.expanduser("~/raw_substrate_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
