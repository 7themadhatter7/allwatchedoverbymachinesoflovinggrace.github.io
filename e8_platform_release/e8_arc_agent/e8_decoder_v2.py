#!/usr/bin/env python3
"""
E8 Field Decoder V2 - Enhanced Pattern Recognition
Ghost in the Machine Labs

Adds: modular affine, even/odd conditional, per-position maps,
lookup table fallback. Patches FieldDecoder from e8_bootstrap_v2.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/joe/sparky/e8_arc_agent')
from e8_arc_engine import N_COLORS
from e8_bootstrap_v2 import FieldDecoder


class EnhancedFieldDecoder(FieldDecoder):

    def _analyze_color_map(self, cmap):
        # Identity
        if all(cmap[c] == c for c in range(N_COLORS)):
            return {'type': 'identity'}
        # Complement
        if all(cmap[c] == 9 - c for c in range(N_COLORS)):
            return {'type': 'complement'}
        # Add with wrap
        for k in range(1, N_COLORS):
            if all(cmap[c] == (c + k) % 10 for c in range(N_COLORS)):
                return {'type': 'add_mod', 'param': k}
        # Add clamped
        for k in range(1, N_COLORS):
            if all(cmap[c] == min(c + k, 9) for c in range(N_COLORS)):
                return {'type': 'add', 'param': k}
            if all(cmap[c] == max(c - k, 0) for c in range(N_COLORS)):
                return {'type': 'subtract', 'param': k}
        # Modular affine: (c * a + b) % 10
        for a in range(2, 10):
            for b in range(0, 10):
                if all(cmap[c] == (c * a + b) % 10 for c in range(N_COLORS)):
                    return {'type': 'affine_mod', 'a': a, 'b': b}
        # Multiply mod
        for a in range(2, 10):
            if all(cmap[c] == (c * a) % 10 for c in range(N_COLORS)):
                return {'type': 'multiply_mod', 'param': a}
        # Simple modulo
        for m in range(2, N_COLORS):
            if all(cmap[c] == c % m for c in range(N_COLORS)):
                return {'type': 'mod', 'param': m}
        # Threshold with configurable high/low
        for t in range(1, N_COLORS):
            for high in range(N_COLORS):
                for low in range(N_COLORS):
                    if high != low and all(cmap[c] == (high if c >= t else low) for c in range(N_COLORS)):
                        return {'type': 'threshold', 'param': t, 'high': high, 'low': low}
        # Even/odd split
        even_rule = self._check_half_rule({c: cmap[c] for c in range(0, N_COLORS, 2)})
        odd_rule = self._check_half_rule({c: cmap[c] for c in range(1, N_COLORS, 2)})
        if even_rule and odd_rule:
            return {'type': 'even_odd', 'even': even_rule, 'odd': odd_rule}
        # Constant output
        vals = set(cmap.values())
        if len(vals) == 1:
            return {'type': 'constant', 'value': vals.pop()}
        # Replace up to 4
        changes = {c: cmap[c] for c in range(N_COLORS) if cmap[c] != c}
        if len(changes) <= 4:
            return {'type': 'replace', 'changes': dict(changes)}
        # FALLBACK: lookup table
        return {'type': 'lookup', 'table': dict(cmap)}

    def _check_half_rule(self, partial):
        if not partial:
            return None
        items = list(partial.items())
        if all(v == c // 2 for c, v in items):
            return 'v // 2'
        if all(v == (c * 2) % 10 for c, v in items):
            return '(v * 2) % 10'
        if all(v == (c * 3) % 10 for c, v in items):
            return '(v * 3) % 10'
        out_vals = set(v for _, v in items)
        if len(out_vals) == 1:
            return str(out_vals.pop())
        return None

    def _synthesize_program(self):
        perm = self.permutation
        w = self.width

        # --- Positional ---
        if perm['is_identity']:
            pos_code = 'lst'
        elif perm['is_reverse']:
            pos_code = 'lst[::-1]'
        else:
            mapping = perm['mapping']
            rot_left = None
            for k in range(1, w):
                if all(mapping[i] == (i + k) % w for i in range(w)):
                    rot_left = k
                    break
            rot_right = None
            if rot_left is None:
                for k in range(1, w):
                    if all(mapping[i] == (i - k) % w for i in range(w)):
                        rot_right = k
                        break
            if rot_left:
                pos_code = f'lst[{rot_left}:] + lst[:{rot_left}]'
            elif rot_right:
                pos_code = f'lst[-{rot_right}:] + lst[:-{rot_right}]'
            else:
                pos_code = f'[lst[i] for i in {mapping}]'

        # --- Value ---
        analyses = [self._analyze_color_map(cm) for cm in self.color_maps]
        uniform = all(analyses[i] == analyses[0] for i in range(1, len(analyses)))

        if uniform:
            ca = analyses[0]
            val_code = self._emit_value(ca)
        else:
            val_code = self._emit_per_position(analyses)

        # --- Compose ---
        if val_code is None:
            code = f'def transform(lst):\n    return {pos_code}'
        elif pos_code == 'lst':
            if '\n' in val_code:
                code = val_code
            else:
                code = f'def transform(lst):\n    return {val_code.replace("{input}", "lst")}'
        else:
            if '\n' in val_code:
                code = val_code.replace('lst[i]', f'({pos_code})[i]').replace('range(len(lst))', f'range(len({pos_code}))')
            else:
                code = f'def transform(lst):\n    s = {pos_code}\n    return {val_code.replace("{input}", "s")}'

        self.program = {
            'positional': 'identity' if pos_code == 'lst' else 'transform',
            'value': analyses[0]['type'] if uniform else 'per_position',
            'uniform': uniform,
            'executable': code,
        }
        return self.program

    def _emit_value(self, ca):
        t = ca['type']
        if t == 'identity':
            return None
        elif t == 'complement':
            return '[9 - v for v in {input}]'
        elif t == 'add':
            return f'[min(v + {ca["param"]}, 9) for v in {{input}}]'
        elif t == 'add_mod':
            return f'[(v + {ca["param"]}) % 10 for v in {{input}}]'
        elif t == 'subtract':
            return f'[max(v - {ca["param"]}, 0) for v in {{input}}]'
        elif t == 'multiply_mod':
            return f'[(v * {ca["param"]}) % 10 for v in {{input}}]'
        elif t == 'affine_mod':
            return f'[(v * {ca["a"]} + {ca["b"]}) % 10 for v in {{input}}]'
        elif t == 'mod':
            return f'[v % {ca["param"]} for v in {{input}}]'
        elif t == 'threshold':
            h = ca.get('high', 9)
            lo = ca.get('low', 0)
            return f'[{h} if v >= {ca["param"]} else {lo} for v in {{input}}]'
        elif t == 'even_odd':
            return f'[{ca["even"]} if v % 2 == 0 else {ca["odd"]} for v in {{input}}]'
        elif t == 'replace':
            pairs = ', '.join(f'{k}: {v}' for k, v in ca['changes'].items())
            return '[{' + pairs + '}.get(v, v) for v in {input}]'
        elif t == 'constant':
            return f'[{ca["value"]}] * len({{input}})'
        elif t == 'lookup':
            return f'[{ca["table"]}[v] for v in {{input}}]'
        return None

    def _emit_per_position(self, analyses):
        parts = []
        for ca in analyses:
            t = ca['type']
            if t == 'identity':
                parts.append('v')
            elif t == 'multiply_mod':
                parts.append(f'(v * {ca["param"]}) % 10')
            elif t == 'affine_mod':
                parts.append(f'(v * {ca["a"]} + {ca["b"]}) % 10')
            elif t == 'add_mod':
                parts.append(f'(v + {ca["param"]}) % 10')
            elif t == 'add':
                parts.append(f'min(v + {ca["param"]}, 9)')
            elif t == 'subtract':
                parts.append(f'max(v - {ca["param"]}, 0)')
            elif t == 'complement':
                parts.append('9 - v')
            elif t == 'threshold':
                h = ca.get('high', 9)
                lo = ca.get('low', 0)
                parts.append(f'({h} if v >= {ca["param"]} else {lo})')
            elif t == 'even_odd':
                parts.append(f'({ca["even"]} if v % 2 == 0 else {ca["odd"]})')
            elif t == 'constant':
                parts.append(str(ca['value']))
            elif t == 'lookup':
                parts.append(f'{ca["table"]}[v]')
            elif t == 'replace':
                pairs = ', '.join(f'{k}: {v}' for k, v in ca['changes'].items())
                parts.append('{' + pairs + '}.get(v, v)')
            else:
                parts.append('v')

        lines = ['def transform(lst):']
        lines.append('    _f = [')
        for p in parts:
            lines.append(f'        lambda v: {p},')
        lines.append('    ]')
        lines.append('    return [_f[i](lst[i]) for i in range(len(lst))]')
        return '\n'.join(lines)

    def report(self):
        if self.program is None:
            self.decode()
        p = self.program
        lines = ['FIELD DECODE REPORT',
                 '  Width: %d' % self.width,
                 '  Positional: %s' % p.get('positional', '?'),
                 '  Value: %s' % p.get('value', '?'),
                 '  Uniform: %s' % p.get('uniform', '?'),
                 '  Executable:']
        for cl in p.get('executable', '').split('\n'):
            lines.append('    ' + cl)
        return '\n'.join(lines)

    def execute(self, input_data):
        if self.program is None:
            self.decode()
        code = self.program.get('executable', '')
        if 'def transform' in code:
            ns = {}
            exec(code, ns)
            return ns['transform'](input_data)
        return self._field_fallback(input_data)

    def _field_fallback(self, input_list):
        from e8_arc_engine import grid_to_state, state_to_grid
        state = grid_to_state([input_list], self.width, 1)
        out = self.field @ state
        grid = state_to_grid(out, self.width, 1)
        return [int(round(max(0, min(9, v)))) for v in grid[0]]


def test_enhanced(name, func, width=5, n_train=80, n_test=5):
    import random
    from e8_arc_engine import solve_task
    random.seed(42)
    seen = set()
    train = []
    while len(train) < n_train:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) in seen: continue
        seen.add(tuple(row))
        out = [max(0, min(9, v)) for v in func(row)]
        train.append({"input": [row], "output": [out]})
    tests = []
    while len(tests) < n_test:
        row = [random.randint(0, 9) for _ in range(width)]
        if tuple(row) not in seen:
            out = [max(0, min(9, v)) for v in func(row)]
            tests.append({"input": row, "expected": out})
            seen.add(tuple(row))
    task = {"train": train, "test": [{"input": [tests[0]["input"]]}]}
    result = solve_task(task)
    if result is None:
        return {"name": name, "status": "no_field", "correct": 0, "total": n_test}
    field, meta = result
    decoder = EnhancedFieldDecoder(field, width, meta)
    program = decoder.decode()
    correct = 0
    failures = []
    for t in tests:
        try:
            actual = [max(0, min(9, v)) for v in decoder.execute(t["input"])]
            if actual == t["expected"]:
                correct += 1
            else:
                failures.append({"input": t["input"], "expected": t["expected"], "actual": actual})
        except Exception as e:
            failures.append({"input": t["input"], "expected": t["expected"], "error": str(e)})
    return {
        "name": name,
        "status": "pass" if correct == n_test else ("partial" if correct > 0 else "fail"),
        "correct": correct, "total": n_test,
        "report": decoder.report(), "code": program['executable'],
        "failures": failures[:2],
    }


if __name__ == '__main__':
    from e8_bootstrap_v3 import phase1_ops
    import time
    print("=" * 64)
    print("ENHANCED DECODER V2 — Tier 2 Test")
    print("=" * 64)
    ops = phase1_ops()
    targets = ['left_shift_wrap', 'threshold', 'even_odd', 'mod_transform',
               'pos_multiply', 'reverse_increment', 'cond_neighbor']
    passed = 0
    for name in targets:
        func = ops[name]
        t0 = time.time()
        r = test_enhanced(name, func)
        dt = time.time() - t0
        st = r['status']
        co = r.get('correct', 0)
        if st == 'pass':
            passed += 1
            print(f"  PASS   {name}: {co}/5 ({dt:.2f}s)")
            code_line = r['code'].replace('\n', ' | ')[:120]
            print(f"         {code_line}")
        elif st == 'no_field':
            print(f"  NOFIELD {name} ({dt:.2f}s)")
        else:
            print(f"  {st.upper():8s} {name}: {co}/5 ({dt:.2f}s)")
            if r.get('failures'):
                f0 = r['failures'][0]
                print(f"         in={f0.get('input')} exp={f0.get('expected')} got={f0.get('actual', f0.get('error',''))}")
            code_line = r.get('code', '?').replace('\n', ' | ')[:150]
            print(f"         {code_line}")
        print()
    print(f"Result: {passed}/{len(targets)}")
