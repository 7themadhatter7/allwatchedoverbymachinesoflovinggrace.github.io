#!/usr/bin/env python3
"""
E8 Bootstrap V3 — Python Control Flow Suite
============================================
Ghost in the Machine Labs

Extends V2 (18/18 single-step ops) with:
  Phase 1: Control flow operations (for, while, if/else, comprehensions)
  Phase 2: Composition chains (2-3 step pipelines)
  Phase 3: Self-improvement loop (decoded programs → new grammar patterns)

Architecture:
  Same as V2 — present I/O pairs, E8 finds field, decoder reads field as program.
  But now the programs are multi-step, conditional, iterative.

Key insight from V2: RM handles position-local ops natively.
Global-dependent ops need the "RM as programmer" paradigm.
This suite tests whether RM can learn the PATTERNS of control flow
as geometric transformations, not the control flow itself.

Depends on: e8_arc_engine.py (E8 engine), e8_bootstrap_v2.py (FieldDecoder)
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, '/home/joe/sparky/e8_arc_agent')
from e8_arc_engine import solve_task, apply_field, N_COLORS
from e8_bootstrap_v2 import FieldDecoder, test_operation

random.seed(42)
np.random.seed(42)

LOG = Path('/home/joe/sparky/e8_arc_agent/logs/bootstrap_v3.log')
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


# =================================================================
# PHASE 1: CONTROL FLOW OPERATIONS
# =================================================================
# Each operation is a Python function that transforms a 1D list.
# The E8 engine must learn the transformation from examples.
# The decoder must recognize the pattern and emit Python.

def phase1_ops():
    """Control flow operations — testing if RM can learn iterative patterns."""
    ops = {}

    # --- FOR LOOP patterns ---

    # Cumulative sum (for i in range: accumulate)
    def cumsum(lst):
        result = []
        s = 0
        for v in lst:
            s = (s + v) % 10
            result.append(s)
        return result
    ops['cumsum'] = cumsum

    # Running max (for loop with state tracking)
    def running_max(lst):
        result = []
        mx = 0
        for v in lst:
            mx = max(mx, v)
            result.append(mx)
        return result
    ops['running_max'] = running_max

    # Pairwise diff (adjacent element relationship)
    def pairwise_diff(lst):
        return [abs(lst[i] - lst[i-1]) if i > 0 else lst[0] for i in range(len(lst))]
    ops['pairwise_diff'] = pairwise_diff

    # Left shift with wrap (circular rotation via iteration)
    def left_shift_wrap(lst):
        return lst[1:] + [lst[0]]
    ops['left_shift_wrap'] = left_shift_wrap

    # --- CONDITIONAL patterns ---

    # Threshold replace (if value > 5 then 9 else 0)
    def threshold(lst):
        return [9 if v > 5 else 0 for v in lst]
    ops['threshold'] = threshold

    # Even/odd transform (if v%2==0: v//2 else v*2%10)
    def even_odd(lst):
        return [v // 2 if v % 2 == 0 else (v * 2) % 10 for v in lst]
    ops['even_odd'] = even_odd

    # Conditional neighbor (if left > right: keep, else: swap with next)
    def cond_neighbor(lst):
        result = list(lst)
        for i in range(len(lst) - 1):
            if lst[i] < lst[i+1]:
                result[i] = lst[i+1]
        return result
    ops['cond_neighbor'] = cond_neighbor

    # Replace minority color with majority
    def majority_replace(lst):
        counts = Counter(lst)
        if len(counts) < 2:
            return list(lst)
        majority = counts.most_common(1)[0][0]
        minority = counts.most_common()[-1][0]
        return [majority if v == minority else v for v in lst]
    ops['majority_replace'] = majority_replace

    # --- WHILE / CONVERGENCE patterns ---

    # Bubble sort single pass (partial sort — one sweep)
    def bubble_pass(lst):
        result = list(lst)
        for i in range(len(result) - 1):
            if result[i] > result[i+1]:
                result[i], result[i+1] = result[i+1], result[i]
        return result
    ops['bubble_pass'] = bubble_pass

    # Gravity down (nonzero values sink to right, zeros float left)
    def gravity_right(lst):
        nonzero = [v for v in lst if v != 0]
        return [0] * (len(lst) - len(nonzero)) + nonzero
    ops['gravity_right'] = gravity_right

    # --- COMPREHENSION patterns ---

    # Map with modular arithmetic
    def mod_transform(lst):
        return [(v * 3 + 1) % 10 for v in lst]
    ops['mod_transform'] = mod_transform

    # Filter and mark (comprehension-like: mark positions where v > median)
    def mark_above_median(lst):
        if not lst:
            return lst
        med = sorted(lst)[len(lst) // 2]
        return [1 if v > med else 0 for v in lst]
    ops['mark_above_median'] = mark_above_median

    # Enumerate-based (position-dependent transform)
    def pos_multiply(lst):
        return [(v * i) % 10 for i, v in enumerate(lst)]
    ops['pos_multiply'] = pos_multiply

    # --- COMPOSITION patterns (2-step) ---

    # Reverse then increment
    def reverse_increment(lst):
        rev = lst[::-1]
        return [(v + 1) % 10 for v in rev]
    ops['reverse_increment'] = reverse_increment

    # Sort then diff
    def sort_then_diff(lst):
        s = sorted(lst)
        return [abs(s[i] - s[i-1]) if i > 0 else s[0] for i in range(len(s))]
    ops['sort_then_diff'] = sort_then_diff

    # Threshold then count
    def threshold_count(lst):
        binary = [1 if v > 4 else 0 for v in lst]
        total = sum(binary)
        return [total] * len(lst)
    ops['threshold_count'] = threshold_count

    # --- 3-STEP COMPOSITION ---

    # Find max, create mask, apply mask
    def max_mask_apply(lst):
        mx = max(lst)
        mask = [1 if v == mx else 0 for v in lst]
        return [v * m for v, m in zip(lst, mask)]
    ops['max_mask_apply'] = max_mask_apply

    # Count colors, recolor minority, shift
    def recolor_shift(lst):
        counts = Counter(lst)
        if len(counts) < 2:
            return lst[1:] + [lst[0]]
        minority = counts.most_common()[-1][0]
        recolored = [0 if v == minority else v for v in lst]
        return recolored[1:] + [recolored[0]]
    ops['recolor_shift'] = recolor_shift

    return ops


# =================================================================
# PHASE 2: 2D GRID OPERATIONS (extending to real ARC dimensionality)
# =================================================================

def phase2_2d_ops():
    """2D grid operations — the real ARC domain."""
    ops = {}

    # Row-wise reverse
    def reverse_rows(grid):
        return [row[::-1] for row in grid]
    ops['reverse_rows'] = reverse_rows

    # Column-wise max propagation (fill each column with its max)
    def col_max_fill(grid):
        h, w = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for c in range(w):
            mx = max(grid[r][c] for r in range(h))
            for r in range(h):
                result[r][c] = mx
        return result
    ops['col_max_fill'] = col_max_fill

    # Diagonal mirror (transpose)
    def transpose(grid):
        h, w = len(grid), len(grid[0])
        return [[grid[r][c] for r in range(h)] for c in range(w)]
    ops['transpose'] = transpose

    # Horizontal symmetry completion (mirror top to bottom)
    def h_symmetry(grid):
        h = len(grid)
        result = [row[:] for row in grid]
        for r in range(h // 2):
            result[h - 1 - r] = result[r][:]
        return result
    ops['h_symmetry'] = h_symmetry

    # Conditional cell transform (if cell > 5 and neighbor > 5: set to 9)
    def neighbor_threshold(grid):
        h, w = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(h):
            for c in range(w):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(grid[nr][nc])
                if grid[r][c] > 5 and any(n > 5 for n in neighbors):
                    result[r][c] = 9
        return result
    ops['neighbor_threshold'] = neighbor_threshold

    # Color count per row (replace each cell with count of its color in that row)
    def row_color_count(grid):
        result = []
        for row in grid:
            counts = Counter(row)
            result.append([counts[v] for v in row])
        return result
    ops['row_color_count'] = row_color_count

    # Gravity down (nonzero cells fall to bottom of column)
    def gravity_down(grid):
        h, w = len(grid), len(grid[0])
        result = [[0]*w for _ in range(h)]
        for c in range(w):
            col_vals = [grid[r][c] for r in range(h) if grid[r][c] != 0]
            for i, v in enumerate(col_vals):
                result[h - len(col_vals) + i][c] = v
        return result
    ops['gravity_down'] = gravity_down

    # Border detection (mark cells adjacent to 0 as 1, others unchanged)
    def border_detect(grid):
        h, w = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                            result[r][c] = 1
                            break
        return result
    ops['border_detect'] = border_detect

    return ops


# =================================================================
# TEST HARNESS for 2D grid ops
# =================================================================

def test_2d_operation(name, func, height=4, width=4, n_train=80, n_test=5):
    """Like test_operation but for 2D grids."""
    random.seed(42)

    seen = set()
    train = []
    while len(train) < n_train:
        grid = [[random.randint(0, 9) for _ in range(width)] for _ in range(height)]
        key = tuple(tuple(r) for r in grid)
        if key in seen:
            continue
        seen.add(key)
        out = func(grid)
        # Clamp to 0-9
        out = [[max(0, min(9, v)) for v in row] for row in out]
        train.append({"input": grid, "output": out})

    tests = []
    while len(tests) < n_test:
        grid = [[random.randint(0, 9) for _ in range(width)] for _ in range(height)]
        key = tuple(tuple(r) for r in grid)
        if key not in seen:
            out = func(grid)
            out = [[max(0, min(9, v)) for v in row] for row in out]
            tests.append({"input": grid, "expected": out})
            seen.add(key)

    task = {
        "train": train,
        "test": [{"input": tests[0]["input"]}]
    }

    result = solve_task(task)
    if result is None:
        return {"name": name, "status": "no_field", "correct": 0, "total": n_test}

    field, meta = result

    # Test via direct field application (not decoder — for 2D we validate the field works)
    correct = 0
    for t in tests:
        try:
            applied = apply_field(field, t["input"], meta)
            if applied is not None:
                applied = [[max(0, min(9, int(round(v)))) for v in row] for row in applied]
                if applied == t["expected"]:
                    correct += 1
        except Exception:
            pass

    return {
        "name": name,
        "status": "pass" if correct == n_test else ("partial" if correct > 0 else "fail"),
        "correct": correct,
        "total": n_test,
    }


# =================================================================
# PHASE 3: SELF-IMPROVEMENT — decoded programs become grammar
# =================================================================

def extract_grammar_from_decoded(results):
    """
    Take successful decoded programs and extract grammar patterns
    for injection into mother_complete.py.
    
    Each decoded program is a composition pattern that Mother learned
    through geometric field formation — not by being told.
    """
    patterns = []
    for r in results:
        if r.get('status') != 'pass' or 'code' not in r:
            continue
        code = r['code']
        name = r['name']

        # Classify the pattern
        has_loop = 'for ' in code or 'while ' in code
        has_cond = 'if ' in code
        has_comp = '[' in code and 'for' in code
        has_sort = 'sort' in code
        has_counter = 'Counter' in code or 'count' in code

        ops_used = []
        if has_loop: ops_used.append('for_range')
        if has_cond: ops_used.append('if_else')
        if has_comp: ops_used.append('list_comp')
        if has_sort: ops_used.append('sorted_by')
        if has_counter: ops_used.append('Counter')

        if ops_used:
            patterns.append({
                'name': f'decoded_{name}',
                'category': 'learned',
                'ops_used': ops_used,
                'source_code': code,
                'origin': 'e8_field_decode',
            })

    return patterns


# =================================================================
# MAIN RUNNER
# =================================================================

def run_phase1():
    log("=" * 64)
    log("E8 BOOTSTRAP V3 — PYTHON CONTROL FLOW SUITE")
    log("=" * 64)

    ops = phase1_ops()
    results = []
    passed = 0
    total = len(ops)

    for name, func in ops.items():
        log(f"\nTesting: {name}")
        t0 = time.time()
        r = test_operation(name, func, width=5, n_train=80, n_test=5)
        dt = time.time() - t0
        r['time'] = dt

        status = r['status']
        correct = r.get('correct', 0)
        n = r.get('total', 5)

        if status == 'pass':
            passed += 1
            log(f"  PASS  {correct}/{n}  ({dt:.2f}s)")
            if 'code' in r:
                log(f"  Code: {r['code'][:120]}")
        elif status == 'partial':
            log(f"  PARTIAL  {correct}/{n}  ({dt:.2f}s)")
        elif status == 'no_field':
            log(f"  NO FIELD  ({dt:.2f}s)")
        else:
            log(f"  FAIL  ({dt:.2f}s)")

        results.append(r)

    log(f"\n{'=' * 64}")
    log(f"PHASE 1 RESULTS: {passed}/{total}")
    log(f"{'=' * 64}")

    return results


def run_phase2():
    log(f"\n{'=' * 64}")
    log("PHASE 2: 2D GRID OPERATIONS")
    log(f"{'=' * 64}")

    ops = phase2_2d_ops()
    results = []
    passed = 0

    for name, func in ops.items():
        log(f"\nTesting 2D: {name}")
        t0 = time.time()
        r = test_2d_operation(name, func, height=4, width=4, n_train=80, n_test=5)
        dt = time.time() - t0
        r['time'] = dt

        if r['status'] == 'pass':
            passed += 1
            log(f"  PASS  {r['correct']}/{r['total']}  ({dt:.2f}s)")
        else:
            log(f"  {r['status'].upper()}  {r['correct']}/{r['total']}  ({dt:.2f}s)")

        results.append(r)

    log(f"\nPHASE 2 RESULTS: {passed}/{len(ops)}")
    return results


def run_phase3(p1_results):
    log(f"\n{'=' * 64}")
    log("PHASE 3: SELF-IMPROVEMENT — GRAMMAR EXTRACTION")
    log(f"{'=' * 64}")

    patterns = extract_grammar_from_decoded(p1_results)
    log(f"Extracted {len(patterns)} grammar patterns from decoded programs")

    for p in patterns:
        log(f"  {p['name']}: ops={p['ops_used']}")

    # Save for injection into mother_complete.py
    if patterns:
        out = Path('/home/joe/sparky/e8_arc_agent/state/decoded_grammar.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(patterns, f, indent=2)
        log(f"Saved to {out}")

    return patterns


def run_all():
    t0 = time.time()

    p1 = run_phase1()
    p2 = run_phase2()
    p3 = run_phase3(p1)

    dt = time.time() - t0

    log(f"\n{'=' * 64}")
    log(f"COMPLETE — {dt:.1f}s total")
    p1_pass = sum(1 for r in p1 if r['status'] == 'pass')
    p2_pass = sum(1 for r in p2 if r['status'] == 'pass')
    log(f"Phase 1 (1D control flow): {p1_pass}/{len(p1)}")
    log(f"Phase 2 (2D grid ops):     {p2_pass}/{len(p2)}")
    log(f"Phase 3 (grammar extract): {len(p3)} patterns")
    log(f"{'=' * 64}")

    # Save full results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'phase1': p1,
        'phase2': p2,
        'phase3': p3,
        'summary': {
            'p1_pass': p1_pass, 'p1_total': len(p1),
            'p2_pass': p2_pass, 'p2_total': len(p2),
            'grammar_extracted': len(p3),
            'total_time': dt,
        }
    }
    out = Path('/home/joe/sparky/e8_arc_agent/state/bootstrap_v3_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Full results: {out}")


if __name__ == '__main__':
    run_all()
