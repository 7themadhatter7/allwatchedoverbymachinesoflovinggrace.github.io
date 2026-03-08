#!/usr/bin/env python3
"""
E8 Bootstrap Language V2 — Field Decoder
==========================================
Ghost in the Machine Labs

Architecture shift: the solved field IS the program.
RM finds the field via geometric pattern matching.
The decoder reads the field's structure and translates
it into executable Python primitives.

Flow:
  1. Present input/output pairs to E8 engine
  2. Engine finds field via pseudoinverse
  3. Decoder analyzes field structure:
     - Permutation signature → positional ops
     - Value mapping signature → transform ops
     - Combined signatures → composite programs
  4. Decoder emits executable Python
  5. Emitted code runs on CPU for novel inputs

The field doesn't need to output program tokens.
The field's matrix IS the program in geometric form.
We just need to read it.
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/joe/sparky/e8_arc_agent')
from e8_arc_engine import (
    solve_task, apply_field, build_and_validate,
    grid_to_state, state_to_grid, N_COLORS
)

random.seed(42)
np.random.seed(42)


# =================================================================
# FIELD DECODER: Read field structure as program
# =================================================================

class FieldDecoder:
    """
    Reads a solved E8 field matrix and extracts the program it encodes.
    
    A field is a matrix F where: output_state = F @ input_state
    
    For a 1xW grid with 10 colors, F is (W*10) x (W*10).
    Each 10x10 block F[i*10:(i+1)*10, j*10:(j+1)*10] describes
    how input position j's color maps to output position i's color.
    
    We analyze these blocks to extract:
    - Which input position feeds each output position (permutation)
    - How colors are remapped at each position (value transform)
    """
    
    def __init__(self, field, width, meta=None):
        self.field = field
        self.width = width
        self.meta = meta or {}
        self.blocks = self._extract_blocks()
        self.permutation = None
        self.color_maps = None
        self.program = None
    
    def _extract_blocks(self):
        """Extract the W×W grid of 10×10 color-mapping blocks."""
        w = self.width
        blocks = np.zeros((w, w, N_COLORS, N_COLORS), dtype=np.float32)
        for oi in range(w):
            for ii in range(w):
                block = self.field[
                    oi * N_COLORS:(oi + 1) * N_COLORS,
                    ii * N_COLORS:(ii + 1) * N_COLORS
                ]
                blocks[oi, ii] = block
        return blocks
    
    def decode(self):
        """Full decode pipeline."""
        self.permutation = self._decode_permutation()
        self.color_maps = self._decode_color_maps()
        self.program = self._synthesize_program()
        return self.program
    
    def _decode_permutation(self):
        """
        Determine which input position drives each output position.
        
        For each output position, find which input position's block
        has the strongest signal (highest Frobenius norm).
        """
        w = self.width
        perm = []
        strengths = []
        
        for oi in range(w):
            norms = []
            for ii in range(w):
                norms.append(np.linalg.norm(self.blocks[oi, ii], 'fro'))
            source = int(np.argmax(norms))
            strength = norms[source]
            perm.append(source)
            strengths.append(strength)
        
        return {
            'mapping': perm,
            'strengths': strengths,
            'is_identity': perm == list(range(w)),
            'is_reverse': perm == list(range(w - 1, -1, -1)),
            'is_permutation': len(set(perm)) == w,
        }
    
    def _decode_color_maps(self):
        """
        For each output position, extract the color remapping
        from its dominant input block.
        
        The 10x10 block acts as: out_color = block @ one_hot(in_color)
        So argmax of each column gives the color mapping.
        """
        w = self.width
        maps = []
        
        for oi in range(w):
            source = self.permutation['mapping'][oi]
            block = self.blocks[oi, source]
            
            # For each input color, which output color does it produce?
            color_map = {}
            for c_in in range(N_COLORS):
                c_out = int(np.argmax(block[:, c_in]))
                color_map[c_in] = c_out
            
            maps.append(color_map)
        
        return maps
    
    def _analyze_color_map(self, cmap):
        """Classify a single position's color mapping."""
        # Identity?
        if all(cmap[c] == c for c in range(N_COLORS)):
            return {'type': 'identity'}
        
        # Complement (9 - c)?
        if all(cmap[c] == 9 - c for c in range(N_COLORS)):
            return {'type': 'complement'}
        
        # Constant offset (c + k) % 10 or clamped?
        for k in range(1, N_COLORS):
            if all(cmap[c] == min(c + k, 9) for c in range(N_COLORS)):
                return {'type': 'add', 'param': k}
            if all(cmap[c] == max(c - k, 0) for c in range(N_COLORS)):
                return {'type': 'subtract', 'param': k}
        
        # Modulo?
        for m in range(2, N_COLORS):
            if all(cmap[c] == c % m for c in range(N_COLORS)):
                return {'type': 'mod', 'param': m}
        
        # Threshold?
        for t in range(1, N_COLORS):
            if all(cmap[c] == (9 if c >= t else 0) for c in range(N_COLORS)):
                return {'type': 'threshold', 'param': t}
        
        # Constant output?
        vals = set(cmap.values())
        if len(vals) == 1:
            return {'type': 'constant', 'value': vals.pop()}
        
        # Replace specific value?
        changes = {c: cmap[c] for c in range(N_COLORS) if cmap[c] != c}
        if len(changes) <= 2:
            return {'type': 'replace', 'changes': dict(changes)}
        
        return {'type': 'unknown', 'map': cmap}
    
    def _synthesize_program(self):
        """
        Combine permutation and color map analysis into
        a human-readable and executable program description.
        """
        perm = self.permutation
        
        # --- Identify positional operation ---
        if perm['is_identity']:
            pos_op = 'identity'
            pos_code = 'lst'
        elif perm['is_reverse']:
            pos_op = 'reverse'
            pos_code = 'lst[::-1]'
        else:
            # Check for rotation
            mapping = perm['mapping']
            w = self.width
            
            # Rotate left by k: output[i] = input[(i+k) % w]
            rot_left = None
            for k in range(1, w):
                if all(mapping[i] == (i + k) % w for i in range(w)):
                    rot_left = k
                    break
            
            # Rotate right by k: output[i] = input[(i-k) % w]
            rot_right = None
            for k in range(1, w):
                if all(mapping[i] == (i - k) % w for i in range(w)):
                    rot_right = k
                    break
            
            if rot_left:
                pos_op = f'rotate_left_{rot_left}'
                pos_code = f'lst[{rot_left}:] + lst[:{rot_left}]'
            elif rot_right:
                pos_op = f'rotate_right_{rot_right}'
                pos_code = f'lst[-{rot_right}:] + lst[:-{rot_right}]'
            elif perm['is_permutation']:
                pos_op = f'permute_{mapping}'
                pos_code = f'[lst[i] for i in {mapping}]'
            else:
                pos_op = f'gather_{mapping}'
                pos_code = f'[lst[i] for i in {mapping}]'
        
        # --- Identify value operation ---
        color_analyses = [self._analyze_color_map(cm) for cm in self.color_maps]
        
        # Check if all positions have the same color map
        uniform = all(
            color_analyses[i] == color_analyses[0]
            for i in range(1, len(color_analyses))
        )
        
        if uniform:
            ca = color_analyses[0]
            val_op = ca['type']
            if ca['type'] == 'identity':
                val_code = None  # No value transform needed
            elif ca['type'] == 'complement':
                val_code = '[9 - v for v in {input}]'
            elif ca['type'] == 'add':
                val_code = f'[min(v + {ca["param"]}, 9) for v in {{input}}]'
            elif ca['type'] == 'subtract':
                val_code = f'[max(v - {ca["param"]}, 0) for v in {{input}}]'
            elif ca['type'] == 'mod':
                val_code = f'[v % {ca["param"]} for v in {{input}}]'
            elif ca['type'] == 'threshold':
                val_code = f'[9 if v >= {ca["param"]} else 0 for v in {{input}}]'
            elif ca['type'] == 'replace':
                changes = ca['changes']
                pairs = ', '.join(str(k) + ': ' + str(v) for k, v in changes.items())
                val_code = '[{' + pairs + '}.get(v, v) for v in {input}]'
            elif ca['type'] == 'constant':
                cval = ca['value']
                val_code = f'[{cval}] * len({{input}})'
            else:
                val_code = None  # Fall back to field application
        else:
            val_op = 'per_position'
            # Build per-position lookup tables as executable code.
            # Each position i gets its own color map dict.
            # Generates: [maps[i].get(v, v) for i, v in enumerate(input)]
            maps_repr = repr([dict(cm) for cm in self.color_maps])
            val_code = f'[__maps[__i].get(__v, __v) for __i, __v in enumerate({{input}})]'
            val_setup = f'    __maps = {maps_repr}\n'
        
        # --- Compose final program ---
        if val_code is None:
            # Pure positional (or unrecognized value transform - use field directly)
            final_code = f'def transform(lst):\n    return {pos_code}'
        elif pos_op == 'identity':
            # Pure value transform
            if val_op == 'per_position':
                final_code = (
                    f'def transform(lst):\n'
                    f'{val_setup}'
                    f'    return {val_code.replace("{input}", "lst")}'
                )
            else:
                final_code = f'def transform(lst):\n    return {val_code.replace("{input}", "lst")}'
        else:
            # Composed: position first, then value
            if val_op == 'per_position':
                final_code = (
                    f'def transform(lst):\n'
                    f'    step1 = {pos_code}\n'
                    f'{val_setup}'
                    f'    return {val_code.replace("{input}", "step1")}'
                )
            else:
                final_code = (
                    f'def transform(lst):\n'
                    f'    step1 = {pos_code}\n'
                    f'    return {val_code.replace("{input}", "step1")}'
                )
        
        return {
            'positional': {
                'operation': pos_op,
                'code': pos_code,
                'permutation': perm['mapping'],
            },
            'value': {
                'operation': val_op,
                'analyses': color_analyses,
                'code': val_code,
                'uniform': uniform,
            },
            'executable': final_code,
        }
    
    def execute(self, input_list):
        """Execute the decoded program on a new input."""
        if self.program is None:
            self.decode()
        
        exec_env = {}
        exec(self.program['executable'], exec_env)
        return exec_env['transform'](input_list)
    
    def report(self):
        """Human-readable report of what the field encodes."""
        if self.program is None:
            self.decode()
        
        p = self.program
        lines = [
            f"FIELD DECODE REPORT",
            f"  Width: {self.width}",
            f"  Positional: {p['positional']['operation']}",
            f"    Permutation: {p['positional']['permutation']}",
            f"  Value: {p['value']['operation']}",
            f"    Uniform: {p['value']['uniform']}",
            f"  Executable:",
        ]
        for code_line in p['executable'].split('\n'):
            lines.append(f"    {code_line}")
        
        return '\n'.join(lines)


# =================================================================
# TEST SUITE
# =================================================================

def test_operation(name, func, width=5, n_train=60, n_test=5):
    """
    End-to-end test:
    1. Generate training pairs from func
    2. Solve with E8 engine (find field)
    3. Decode field into program
    4. Execute decoded program on novel inputs
    5. Compare to func
    """
    random.seed(42)
    
    # Generate training data
    seen = set()
    train = []
    while len(train) < n_train:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) in seen:
            continue
        seen.add(tuple(row))
        out = func(row)
        out = [max(0, min(9, v)) for v in out]
        train.append({"input": [row], "output": [out]})
    
    # Generate test data
    tests = []
    while len(tests) < n_test:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) not in seen:
            out = func(row)
            out = [max(0, min(9, v)) for v in out]
            tests.append({"input": row, "expected": out})
            seen.add(tuple(row))
    
    # Solve
    task = {
        "train": train,
        "test": [{"input": [tests[0]["input"]]}]
    }
    
    result = solve_task(task)
    if result is None:
        return {"name": name, "status": "no_field"}
    
    field, meta = result
    ih, iw, oh, ow = meta['dims']
    
    # Decode
    decoder = FieldDecoder(field, width, meta)
    program = decoder.decode()
    
    # Test decoded program
    correct = 0
    failures = []
    for t in tests:
        try:
            actual = decoder.execute(t["input"])
            actual = [max(0, min(9, v)) for v in actual]
            if actual == t["expected"]:
                correct += 1
            else:
                failures.append({
                    "input": t["input"],
                    "expected": t["expected"],
                    "actual": actual,
                })
        except Exception as e:
            failures.append({
                "input": t["input"],
                "expected": t["expected"],
                "error": str(e),
            })
    
    return {
        "name": name,
        "status": "pass" if correct == n_test else "partial",
        "correct": correct,
        "total": n_test,
        "report": decoder.report(),
        "code": program['executable'],
        "failures": failures[:2],  # First 2 failures for debug
    }


def run_full_suite():
    WIDTH = 5
    N_TRAIN = 60
    
    print("=" * 64)
    print("E8 BOOTSTRAP V2 — FIELD DECODER")
    print("Ghost in the Machine Labs")
    print("The solved field IS the program. Decoder reads it.")
    print("=" * 64)
    
    operations = {
        "BIN1_POSITIONAL": [
            ("reverse",       lambda lst: lst[::-1]),
            ("rotate_left_1", lambda lst: lst[1:] + lst[:1]),
            ("rotate_left_2", lambda lst: lst[2:] + lst[:2]),
            ("rotate_right_1",lambda lst: lst[-1:] + lst[:-1]),
            ("swap_ends",     lambda lst: [lst[-1]] + lst[1:-1] + [lst[0]]),
            ("mirror_half",   lambda lst: lst[:len(lst)//2] + [lst[len(lst)//2]] + lst[:len(lst)//2][::-1]),
        ],
        "BIN2_VALUE_TRANSFORM": [
            ("increment",     lambda lst: [min(v + 1, 9) for v in lst]),
            ("decrement",     lambda lst: [max(v - 1, 0) for v in lst]),
            ("add_3",         lambda lst: [min(v + 3, 9) for v in lst]),
            ("complement",    lambda lst: [9 - v for v in lst]),
            ("mod3",          lambda lst: [v % 3 for v in lst]),
            ("threshold_5",   lambda lst: [9 if v >= 5 else 0 for v in lst]),
            ("threshold_3",   lambda lst: [9 if v >= 3 else 0 for v in lst]),
            ("replace_0w5",   lambda lst: [5 if v == 0 else v for v in lst]),
        ],
        "BIN3_COMPOSED": [
            ("rev_then_inc",  lambda lst: [min(v + 1, 9) for v in lst[::-1]]),
            ("rotL_then_comp",lambda lst: [9 - v for v in (lst[1:] + lst[:1])]),
            ("rev_then_mod3", lambda lst: [v % 3 for v in lst[::-1]]),
            ("rev_then_thresh",lambda lst: [9 if v >= 5 else 0 for v in lst[::-1]]),
        ],
    }
    
    total_ops = 0
    total_pass = 0
    all_results = {}
    t0 = time.time()
    
    for bin_name, ops in operations.items():
        print(f"\n{'─' * 64}")
        print(f"  {bin_name}")
        print(f"{'─' * 64}")
        
        bin_results = []
        
        for name, func in ops:
            total_ops += 1
            result = test_operation(name, func, WIDTH, N_TRAIN)
            bin_results.append(result)
            
            if result['status'] == 'pass':
                total_pass += 1
                print(f"  {name:20s} — PASS")
                # Show decoded program
                for line in result['code'].split('\n'):
                    print(f"    {line}")
            elif result['status'] == 'partial':
                print(f"  {name:20s} — PARTIAL {result['correct']}/{result['total']}")
                for line in result['code'].split('\n'):
                    print(f"    {line}")
                if result['failures']:
                    f = result['failures'][0]
                    print(f"    FAIL: input={f.get('input')}")
                    print(f"          expect={f.get('expected')}")
                    print(f"          actual={f.get('actual', f.get('error'))}")
            else:
                print(f"  {name:20s} — NO FIELD")
        
        all_results[bin_name] = bin_results
    
    elapsed = time.time() - t0
    
    # Summary
    print(f"\n{'=' * 64}")
    print(f"  RESULTS")
    print(f"{'=' * 64}")
    print(f"  Operations decoded: {total_pass}/{total_ops}")
    print(f"  Time: {elapsed:.2f}s")
    print()
    
    for bin_name, results in all_results.items():
        passed = sum(1 for r in results if r['status'] == 'pass')
        print(f"  {bin_name:25s}  {passed}/{len(results)}")
    
    print(f"\n{'=' * 64}")
    print(f"  DECODED PROGRAMS SUMMARY")
    print(f"{'=' * 64}")
    for bin_name, results in all_results.items():
        for r in results:
            if r['status'] in ('pass', 'partial') and 'code' in r:
                print(f"\n  [{r['name']}]")
                for line in r['code'].split('\n'):
                    print(f"    {line}")
    
    print(f"\n{'=' * 64}")
    
    # Save
    output_path = Path('/home/joe/sparky/e8_arc_agent/state/bootstrap_v2_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize results (remove lambdas)
    save_results = {}
    for bin_name, results in all_results.items():
        save_results[bin_name] = []
        for r in results:
            save_r = {k: v for k, v in r.items() if k != 'report'}
            save_results[bin_name].append(save_r)
    
    report = {
        "engine": "E8 Bootstrap V2 — Field Decoder",
        "timestamp": datetime.now().isoformat(),
        "width": WIDTH,
        "n_train": N_TRAIN,
        "elapsed": round(elapsed, 2),
        "passed": total_pass,
        "total": total_ops,
        "results": save_results,
    }
    json.dump(report, open(output_path, 'w'), indent=2, default=str)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run_full_suite()

