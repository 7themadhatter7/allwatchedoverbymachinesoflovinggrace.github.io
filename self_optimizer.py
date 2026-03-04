#!/usr/bin/env python3
"""
HARMONIC SUBSTRATE SELF-OPTIMIZER
=================================
Ghost in the Machine Labs

Finds the core count that achieves 90% of peak throughput.
Measures actual performance, not theoretical RAM capacity.

The throughput curve rises then plateaus due to sqrt(N) cluster
sampling and per-core overhead. The optimizer finds that plateau.
"""

import sys
import os
import time
import math
import json
import numpy as np

sys.path.insert(0, os.path.expanduser('~/sparky'))
sys.path.insert(0, os.path.expanduser('~/harmonic-stack'))

from fused_harmonic_substrate import FusedHarmonicSubstrate, CoreRole


def measure(core_count: int, iters: int = 300) -> dict:
    """Build substrate at N cores, measure throughput."""
    s = FusedHarmonicSubstrate()
    s.TOTAL_CORES = core_count
    per_core_mb = 0.305
    budget_mb = core_count * per_core_mb * 1.25
    s._detect_capacity = classmethod(
        lambda cls, reserve_pct=0.2, _b=budget_mb, _n=core_count: {
            'total_ram_mb': _b * 1.25, 'available_ram_mb': _b,
            'reserve_mb': 0, 'budget_mb': _b, 'per_core_mb': per_core_mb,
            'max_cores': _n, 'target_cores': _n
        }
    ).__get__(type(s))
    s._autoscale_buffers = staticmethod(
        lambda bm, _n=core_count, **kw: {
            'cores': _n, 'spark_kb': 8, 'domain_kb': 16,
            'per_core_kb': 312, 'per_core_mb': 0.305,
            'total_mb': _n * 0.305, 'budget_mb': bm,
            'target_pct': 0.9, 'utilization': 90.0
        }
    )
    s.ALLOCATION = s._scale_allocation(core_count)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    t_fab = time.perf_counter()
    try:
        s.build()
    finally:
        sys.stdout = old_stdout
    fab_s = time.perf_counter() - t_fab

    signal = np.random.randn(1024).astype(np.float32)

    # Warmup
    for _ in range(min(20, iters // 5)):
        s.process_signal(signal)

    t0 = time.perf_counter()
    for _ in range(iters):
        s.process_signal(signal)
    elapsed = time.perf_counter() - t0

    ops = iters / elapsed
    workers = s.get_cores_by_role(CoreRole.WORKER)
    cluster = max(1, min(int(math.sqrt(len(workers))), 256))
    routers = min(16, len(s.get_cores_by_role(CoreRole.ROUTER)))
    fires_per_call = routers + cluster
    mem_gb = getattr(s, 'total_memory_gb', 0)

    return {
        'cores': core_count,
        'cluster': cluster,
        'fires_call': fires_per_call,
        'ops_s': round(ops, 1),
        'fires_s': round(fires_per_call * ops),
        'latency_ms': round(elapsed / iters * 1000, 2),
        'mem_gb': round(mem_gb, 3),
        'fab_s': round(fab_s, 1),
    }


def optimize(target_pct: float = 0.90):
    """
    Find optimal core count.

    1. Sample throughput at exponential intervals
    2. Find peak
    3. Refine around peak
    4. Report core count at target_pct of peak
    """
    cap = FusedHarmonicSubstrate._detect_capacity()
    avail_gb = FusedHarmonicSubstrate._available_ram_mb() / 1024

    print('=' * 60)
    print('  HARMONIC SUBSTRATE SELF-OPTIMIZER')
    print('  Ghost in the Machine Labs')
    print('=' * 60)
    print(f'  RAM: {cap["total_ram_mb"]/1024:.1f} GB total, {avail_gb:.1f} GB available')
    print(f'  Target: {target_pct*100:.0f}% of peak throughput')
    print()

    # Phase 1: Coarse sweep
    # sqrt(N) means diminishing returns fast. Sample 50 to 1000.
    samples = [50, 100, 150, 200, 300, 400, 500, 650, 800, 1000]
    print('  Phase 1: Coarse sweep')
    print(f'  {"Cores":>6} {"Cluster":>7} {"Ops/s":>8} {"Fires/s":>9} '
          f'{"Lat ms":>7} {"Mem GB":>7} {"Fab s":>5}')
    print('  ' + '-' * 52)

    results = []
    for n in samples:
        r = measure(n)
        results.append(r)
        print(f'  {r["cores"]:>6} {r["cluster"]:>7} {r["ops_s"]:>8.0f} '
              f'{r["fires_s"]:>9,.0f} {r["latency_ms"]:>7.2f} '
              f'{r["mem_gb"]:>7.3f} {r["fab_s"]:>5.1f}')

    # Find peak ops/s (not fires/s — ops/s is what matters for prompt latency)
    peak = max(results, key=lambda r: r['ops_s'])
    peak_ops = peak['ops_s']
    threshold = peak_ops * target_pct

    print(f'\n  Peak: {peak_ops:.0f} ops/s at {peak["cores"]} cores')
    print(f'  Target ({target_pct*100:.0f}%): {threshold:.0f} ops/s')

    # Find all points above threshold
    above = [r for r in results if r['ops_s'] >= threshold]
    if not above:
        above = [peak]

    # The optimum is the HIGHEST core count still above threshold
    # (more cores = more geometric diversity = richer field)
    optimum_coarse = max(above, key=lambda r: r['cores'])

    # Phase 2: Refine around the optimum
    # Search between the optimum and the next sample above it that drops below threshold
    low = optimum_coarse['cores']
    # Find next sample above that drops below
    higher = [r for r in results if r['cores'] > low and r['ops_s'] < threshold]
    if higher:
        high = min(higher, key=lambda r: r['cores'])['cores']
    else:
        high = min(low * 2, 1500)

    if high - low > 50:
        print(f'\n  Phase 2: Refine [{low} - {high}]')
        step = max(25, (high - low) // 6)
        refine_points = list(range(low, high + 1, step))
        # Avoid re-measuring existing points
        existing_cores = {r['cores'] for r in results}
        refine_points = [n for n in refine_points if n not in existing_cores]

        for n in refine_points:
            r = measure(n)
            results.append(r)
            print(f'  {r["cores"]:>6} {r["cluster"]:>7} {r["ops_s"]:>8.0f} '
                  f'{r["fires_s"]:>9,.0f} {r["latency_ms"]:>7.2f} '
                  f'{r["mem_gb"]:>7.3f} {r["fab_s"]:>5.1f}')

    # Final determination
    above = [r for r in results if r['ops_s'] >= threshold]
    optimum = max(above, key=lambda r: r['cores'])

    # Memory check: does optimum fit in available RAM?
    if optimum['mem_gb'] > avail_gb * 0.90:
        print(f'\n  WARNING: {optimum["cores"]} cores needs {optimum["mem_gb"]:.1f} GB '
              f'but only {avail_gb:.1f} GB available')
        # Fall back to largest that fits
        fits = [r for r in above if r['mem_gb'] < avail_gb * 0.90]
        if fits:
            optimum = max(fits, key=lambda r: r['cores'])

    print()
    print('=' * 60)
    print('  RESULT')
    print('=' * 60)
    print(f'  Optimal cores:    {optimum["cores"]}')
    print(f'  Cluster size:     {optimum["cluster"]} (sqrt sampling)')
    print(f'  Throughput:       {optimum["ops_s"]:.0f} ops/s')
    print(f'  Total fires:      {optimum["fires_s"]:,.0f} fires/s')
    print(f'  Latency:          {optimum["latency_ms"]:.2f} ms')
    print(f'  Memory:           {optimum["mem_gb"]:.3f} GB')
    print(f'  Fabrication:      {optimum["fab_s"]:.1f}s')
    print(f'  Peak efficiency:  {optimum["ops_s"]/peak_ops*100:.1f}% of peak')
    print()
    print(f'  Usage:')
    print(f'    python3 harmonic_v1.py --cores {optimum["cores"]}')
    print('=' * 60)

    # Save results
    output = {
        'optimum': optimum,
        'peak': {'ops_s': peak_ops, 'cores': peak['cores']},
        'target_pct': target_pct,
        'threshold_ops': threshold,
        'system': {
            'total_ram_gb': round(cap['total_ram_mb'] / 1024, 1),
            'available_ram_gb': round(avail_gb, 1),
        },
        'all_samples': sorted(results, key=lambda r: r['cores']),
    }
    out_path = os.path.expanduser('~/optimizer_result.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Full data: {out_path}')

    return output


if __name__ == '__main__':
    pct = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
    optimize(target_pct=pct)
