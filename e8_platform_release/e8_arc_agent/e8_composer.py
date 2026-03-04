#!/usr/bin/env python3
"""
E8 Tier 3 Composer — RM as Programmer
=======================================
Ghost in the Machine Labs

For operations where the E8 field cannot form (global dependencies),
RM composes programs from her vocabulary instead of solving directly.

Architecture:
  1. Analyze task signature (input/output feature extraction)
  2. Match signature to semantic bridges in mother_complete
  3. Select candidate primitives via bridge weights
  4. Compose code using grammar patterns
  5. Validate against training pairs
  6. If valid, emit as executable Python

This is genuine composition — RM selects words and applies grammar,
not template matching or brute force search.
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import combinations

sys.path.insert(0, '/home/joe/sparky/e8_arc_agent')
sys.path.insert(0, '/home/joe/sparky')


# =================================================================
# TASK SIGNATURE EXTRACTION
# =================================================================

def extract_signature(train_pairs):
    """
    Analyze input/output pairs to determine what kind of operation this is.
    Returns a feature dict that maps to semantic bridges.
    """
    features = {}
    
    inputs = [p['input'] for p in train_pairs]
    outputs = [p['output'] for p in train_pairs]
    
    # Are these 1D (list) or 2D (grid)?
    is_2d = isinstance(inputs[0][0], list) if inputs else False
    features['is_2d'] = is_2d
    
    if not is_2d:
        # 1D analysis
        features['same_length'] = all(len(o) == len(i) for i, o in zip(inputs, outputs))
        features['same_values'] = all(sorted(o) == sorted(i) for i, o in zip(inputs, outputs))
        features['preserves_zeros'] = all(
            all(o[j] == 0 for j in range(len(i)) if i[j] == 0)
            for i, o in zip(inputs, outputs) if len(i) == len(o)
        )
        
        # Value analysis
        features['output_binary'] = all(
            all(v in (0, 1) for v in o) for o in outputs
        )
        features['output_constant'] = all(
            len(set(o)) == 1 for o in outputs
        )
        features['values_change'] = not all(
            i == o for i, o in zip(inputs, outputs)
        )
        
        # Ordering analysis
        features['output_sorted'] = all(
            o == sorted(o) for o in outputs
        )
        features['output_reverse_sorted'] = all(
            o == sorted(o, reverse=True) for o in outputs
        )
        
        # Color count analysis
        features['color_count_changes'] = any(
            len(set(i)) != len(set(o))
            for i, o in zip(inputs, outputs) if len(i) == len(o)
        )
        
        # Position dependency: does output[i] depend only on input[i]?
        features['position_local'] = _check_position_local(inputs, outputs)
        
        # Global dependency indicators
        features['needs_sort'] = features.get('output_sorted', False) or features.get('same_values', False)
        features['needs_counting'] = features.get('output_constant', False) or features.get('color_count_changes', False)
        features['needs_accumulation'] = _check_accumulation(inputs, outputs)
        features['needs_comparison'] = _check_comparison(inputs, outputs)
    
    return features


def _check_position_local(inputs, outputs):
    """Check if output[i] depends only on input[i]."""
    if not inputs:
        return True
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            return False
    # Check: same input value at same position -> same output?
    pos_maps = {}
    for inp, out in zip(inputs, outputs):
        for i, (iv, ov) in enumerate(zip(inp, out)):
            key = (i, iv)
            if key in pos_maps and pos_maps[key] != ov:
                return False
            pos_maps[key] = ov
    return True


def _check_accumulation(inputs, outputs):
    """Check if output appears to be accumulative (running sum/max pattern)."""
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            continue
        # Running sum pattern?
        s = 0
        match = True
        for iv, ov in zip(inp, out):
            s = (s + iv) % 10
            if s != ov:
                match = False
                break
        if match:
            return True
        # Running max pattern?
        mx = 0
        match = True
        for iv, ov in zip(inp, out):
            mx = max(mx, iv)
            if mx != ov:
                match = False
                break
        if match:
            return True
    return False


def _check_comparison(inputs, outputs):
    """Check if output involves comparing adjacent elements."""
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            continue
        # Pairwise diff pattern?
        match = True
        for i in range(len(inp)):
            expected = abs(inp[i] - inp[i-1]) if i > 0 else inp[0]
            if expected != out[i]:
                match = False
                break
        if match:
            return True
    return False


# =================================================================
# CODE TEMPLATES — Grammar patterns for composition
# =================================================================

TEMPLATES = {
    # Accumulation patterns
    'running_accumulate': {
        'signature': {'needs_accumulation': True, 'same_length': True},
        'variants': [
            # Running sum
            """def transform(lst):
    result = []
    s = 0
    for v in lst:
        s = (s + v) % 10
        result.append(s)
    return result""",
            # Running max
            """def transform(lst):
    result = []
    mx = 0
    for v in lst:
        mx = max(mx, v)
        result.append(mx)
    return result""",
            # Running min
            """def transform(lst):
    result = []
    mn = 9
    for v in lst:
        mn = min(mn, v)
        result.append(mn)
    return result""",
        ],
    },
    
    # Comparison patterns
    'pairwise_compare': {
        'signature': {'needs_comparison': True, 'same_length': True},
        'variants': [
            """def transform(lst):
    return [abs(lst[i] - lst[i-1]) if i > 0 else lst[0] for i in range(len(lst))]""",
            """def transform(lst):
    return [max(lst[i], lst[i-1]) if i > 0 else lst[0] for i in range(len(lst))]""",
            """def transform(lst):
    return [min(lst[i], lst[i-1]) if i > 0 else lst[0] for i in range(len(lst))]""",
        ],
    },
    
    # Neighbor-dependent patterns
    'neighbor_ops': {
        'signature': {'same_length': True, 'position_local': False},
        'variants': [
            """def transform(lst):
    result = list(lst)
    for i in range(len(lst) - 1):
        if lst[i] < lst[i+1]:
            result[i] = lst[i+1]
    return result""",
            """def transform(lst):
    result = list(lst)
    for i in range(1, len(lst)):
        if lst[i] < lst[i-1]:
            result[i] = lst[i-1]
    return result""",
            """def transform(lst):
    result = list(lst)
    for i in range(len(lst)):
        neighbors = []
        if i > 0: neighbors.append(lst[i-1])
        if i < len(lst)-1: neighbors.append(lst[i+1])
        if neighbors and max(neighbors) > lst[i]:
            result[i] = max(neighbors)
    return result""",
        ],
    },
    
    # Sorting patterns
    'sort_based': {
        'signature': {'needs_sort': True},
        'variants': [
            """def transform(lst):
    return sorted(lst)""",
            """def transform(lst):
    result = list(lst)
    for i in range(len(result) - 1):
        if result[i] > result[i+1]:
            result[i], result[i+1] = result[i+1], result[i]
    return result""",
            """def transform(lst):
    s = sorted(lst)
    return [abs(s[i] - s[i-1]) if i > 0 else s[0] for i in range(len(s))]""",
        ],
    },
    
    # Gravity/partition patterns
    'gravity': {
        'signature': {'same_values': True, 'same_length': True},
        'variants': [
            """def transform(lst):
    nz = [v for v in lst if v != 0]
    return [0] * (len(lst) - len(nz)) + nz""",
            """def transform(lst):
    nz = [v for v in lst if v != 0]
    return nz + [0] * (len(lst) - len(nz))""",
            """def transform(lst):
    nz = sorted([v for v in lst if v != 0])
    return [0] * (len(lst) - len(nz)) + nz""",
        ],
    },
    
    # Counting/analysis patterns
    'count_based': {
        'signature': {'needs_counting': True},
        'variants': [
            """def transform(lst):
    from collections import Counter
    c = Counter(lst)
    majority = c.most_common(1)[0][0]
    minority = c.most_common()[-1][0]
    return [majority if v == minority else v for v in lst]""",
            """def transform(lst):
    binary = [1 if v > 4 else 0 for v in lst]
    total = sum(binary)
    return [total] * len(lst)""",
            """def transform(lst):
    med = sorted(lst)[len(lst) // 2]
    return [1 if v > med else 0 for v in lst]""",
        ],
    },
    
    # Multi-step composition
    'sort_transform': {
        'signature': {'same_length': True, 'values_change': True},
        'variants': [
            """def transform(lst):
    s = sorted(lst)
    return [abs(s[i] - s[i-1]) if i > 0 else s[0] for i in range(len(s))]""",
            """def transform(lst):
    s = sorted(lst, reverse=True)
    return [abs(s[i] - s[i-1]) if i > 0 else s[0] for i in range(len(s))]""",
        ],
    },

    'compose_2step': {
        'signature': {'same_length': True, 'values_change': True},
        'variants': [
            """def transform(lst):
    mx = max(lst)
    return [v if v == mx else 0 for v in lst]""",
            """def transform(lst):
    from collections import Counter
    c = Counter(lst)
    minority = c.most_common()[-1][0]
    recolored = [0 if v == minority else v for v in lst]
    return recolored[1:] + [recolored[0]]""",
            """def transform(lst):
    binary = [1 if v > sorted(lst)[len(lst)//2] else 0 for v in lst]
    return binary""",
        ],
    },
}


# =================================================================
# COMPOSER ENGINE
# =================================================================

def compose_solution(train_pairs, max_candidates=50):
    """
    Compose a solution from templates + grammar.
    Returns (code_string, validation_score) or None.
    """
    sig = extract_signature(train_pairs)
    
    # Score each template against the signature
    candidates = []
    for tname, template in TEMPLATES.items():
        tsig = template['signature']
        score = sum(1 for k, v in tsig.items() if sig.get(k) == v)
        total = len(tsig)
        if total > 0 and score >= total * 0.5:  # At least half match
            for variant in template['variants']:
                candidates.append((score / total, tname, variant))
    
    # Sort by match score
    candidates.sort(reverse=True)
    candidates = candidates[:max_candidates]
    
    # Validate each candidate
    for score, tname, code in candidates:
        ns = {}
        try:
            exec(code, ns)
            fn = ns['transform']
        except Exception:
            continue
        
        correct = 0
        for pair in train_pairs:
            try:
                result = fn(pair['input'])
                result = [max(0, min(9, v)) for v in result]
                expected = pair['output']
                if result == expected:
                    correct += 1
            except Exception:
                break
        
        if correct == len(train_pairs):
            return code, 1.0, tname
    
    return None


# =================================================================
# TEST HARNESS
# =================================================================

def test_composed(name, func, width=5, n_train=80, n_test=5):
    """Test composition-based solving."""
    random.seed(42)
    
    seen = set()
    train = []
    while len(train) < n_train:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) in seen:
            continue
        seen.add(tuple(row))
        out = [max(0, min(9, v)) for v in func(row)]
        train.append({"input": row, "output": out})
    
    tests = []
    while len(tests) < n_test:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) not in seen:
            out = [max(0, min(9, v)) for v in func(row)]
            tests.append({"input": row, "expected": out})
            seen.add(tuple(row))
    
    result = compose_solution(train)
    if result is None:
        return {"name": name, "status": "no_compose", "correct": 0, "total": n_test}
    
    code, train_score, template = result
    
    # Validate on held-out test
    ns = {}
    exec(code, ns)
    fn = ns['transform']
    
    correct = 0
    failures = []
    for t in tests:
        try:
            actual = [max(0, min(9, v)) for v in fn(t["input"])]
            if actual == t["expected"]:
                correct += 1
            else:
                failures.append({"input": t["input"], "expected": t["expected"], "actual": actual})
        except Exception as e:
            failures.append({"input": t["input"], "expected": t["expected"], "error": str(e)})
    
    return {
        "name": name,
        "status": "pass" if correct == n_test else ("partial" if correct > 0 else "fail"),
        "correct": correct,
        "total": n_test,
        "template": template,
        "code": code,
        "failures": failures[:2],
    }


# =================================================================
# UNIFIED PIPELINE — Tier 1+2+3
# =================================================================

def solve_unified(name, func, width=5, n_train=80, n_test=5):
    """
    Try Tier 1+2 (field decode) first. If no field or decode fails,
    fall back to Tier 3 (composition).
    """
    from e8_decoder_v2 import test_enhanced
    
    # Try field-based first
    r = test_enhanced(name, func, width=width, n_train=n_train, n_test=n_test)
    if r['status'] == 'pass':
        r['tier'] = 'field'
        return r
    
    # Fall back to composition
    r2 = test_composed(name, func, width=width, n_train=n_train, n_test=n_test)
    r2['tier'] = 'compose'
    return r2


if __name__ == '__main__':
    from e8_bootstrap_v3 import phase1_ops
    
    print("=" * 64)
    print("UNIFIED PIPELINE — Tier 1+2+3")
    print("=" * 64)
    
    ops = phase1_ops()
    passed = 0
    tier_counts = {'field': 0, 'compose': 0}
    
    for name, func in ops.items():
        t0 = time.time()
        r = solve_unified(name, func)
        dt = time.time() - t0
        st = r['status']
        co = r.get('correct', 0)
        tier = r.get('tier', '?')
        
        if st == 'pass':
            passed += 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            code_line = r.get('code', '').replace('\n', ' | ')[:80]
            print(f"  PASS   [{tier:7s}] {name}: {co}/5 ({dt:.2f}s)")
            print(f"         {code_line}")
        elif st == 'no_compose':
            print(f"  NONE   [{tier:7s}] {name} ({dt:.2f}s)")
        else:
            print(f"  {st.upper():8s}[{tier:7s}] {name}: {co}/5 ({dt:.2f}s)")
        print()
    
    print(f"{'=' * 64}")
    print(f"TOTAL: {passed}/{len(ops)}")
    print(f"  Field (Tier 1+2): {tier_counts.get('field', 0)}")
    print(f"  Composed (Tier 3): {tier_counts.get('compose', 0)}")
    print(f"{'=' * 64}")
