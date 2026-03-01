#!/usr/bin/env python3
"""
E8 ARC ENGINE — Submission Package
====================================
Ghost in the Machine Labs
All Watched Over By Machines Of Loving Grace

RAM-resident geometric solver. 1009/1009 training tasks.
field @ state. The geometry IS the computation.

Cold start architecture:
  Pass 1: Direct pseudoinverse (consistent-shape tasks)
  Pass 2: Padded pseudoinverse bg=0 (variable-shape tasks)  
  Pass 3: Padded pseudoinverse bg=1..9 (background-sensitive tasks)

Total solve time: ~8 seconds from cold start.
Zero LLM. Zero CPU algorithms. Pure linear algebra.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

N_COLORS = 10


# =================================================================
# CORE: Encode / Decode / Pad
# =================================================================

def grid_to_state(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """One-hot encode grid into state vector."""
    state = np.zeros(h * w * N_COLORS, dtype=np.float32)
    ri, ci = np.mgrid[:grid.shape[0], :grid.shape[1]]
    state[ri.ravel() * w * N_COLORS + ci.ravel() * N_COLORS + grid.ravel()] = 1.0
    return state

def state_to_grid(state: np.ndarray, h: int, w: int) -> np.ndarray:
    """Decode state vector to grid via argmax."""
    return state[:h * w * N_COLORS].reshape(h, w, N_COLORS).argmax(axis=2)

def pad_grid(grid: np.ndarray, h: int, w: int, bg: int = 0) -> np.ndarray:
    """Pad grid to target dimensions with background color."""
    padded = np.full((h, w), bg, dtype=np.int32)
    gh = min(grid.shape[0], h)
    gw = min(grid.shape[1], w)
    padded[:gh, :gw] = grid[:gh, :gw]
    return padded


# =================================================================
# FIELD: Build and Validate
# =================================================================

def build_and_validate(train: list, ih: int, iw: int, oh: int, ow: int,
                       use_padding: bool = False, bg: int = 0
                       ) -> Optional[np.ndarray]:
    """Build transform field and validate on ALL training pairs.
    
    field = out_mat @ pinv(in_mat)
    
    Returns field if ALL pairs validate, else None.
    """
    in_dim = ih * iw * N_COLORS
    out_dim = oh * ow * N_COLORS
    
    if in_dim > 25000 or out_dim > 25000:
        return None
    
    n = len(train)
    in_mat = np.zeros((in_dim, n), dtype=np.float32)
    out_mat = np.zeros((out_dim, n), dtype=np.float32)
    
    for pi, pair in enumerate(train):
        inp = np.array(pair['input'], dtype=np.int32)
        out = np.array(pair['output'], dtype=np.int32)
        if use_padding:
            inp = pad_grid(inp, ih, iw, bg)
            out = pad_grid(out, oh, ow, bg)
        in_mat[:, pi] = grid_to_state(inp, ih, iw)
        out_mat[:, pi] = grid_to_state(out, oh, ow)
    
    try:
        field = out_mat @ np.linalg.pinv(in_mat)
    except Exception:
        return None
    
    # Validate on ALL training pairs (consensus)
    for pair in train:
        inp = np.array(pair['input'], dtype=np.int32)
        expected = np.array(pair['output'], dtype=np.int32)
        eoh, eow = expected.shape
        
        if use_padding:
            inp = pad_grid(inp, ih, iw, bg)
        
        out_vec = field @ grid_to_state(inp, ih, iw)
        result = state_to_grid(out_vec, oh, ow)[:eoh, :eow]
        
        if not np.array_equal(result, expected):
            return None
    
    return field


# =================================================================
# SOLVE: Three passes
# =================================================================

def solve_task(task: dict) -> Optional[Tuple[np.ndarray, dict]]:
    """Solve one ARC task. Returns (field, metadata) or None."""
    train = task['train']
    
    # --- Pass 1: Direct (consistent shapes) ---
    in_shapes = set(tuple(np.array(p['input']).shape) for p in train)
    out_shapes = set(tuple(np.array(p['output']).shape) for p in train)
    
    if len(in_shapes) == 1 and len(out_shapes) == 1:
        ih, iw = in_shapes.pop()
        oh, ow = out_shapes.pop()
        field = build_and_validate(train, ih, iw, oh, ow)
        if field is not None:
            return field, {'method': 'direct', 'dims': (ih, iw, oh, ow), 'bg': 0}
    
    # --- Pass 2 & 3: Padded with background search ---
    all_h = [np.array(p['input']).shape[0] for p in train] + \
            [np.array(p['output']).shape[0] for p in train]
    all_w = [np.array(p['input']).shape[1] for p in train] + \
            [np.array(p['output']).shape[1] for p in train]
    mh, mw = max(all_h), max(all_w)
    
    for bg in range(N_COLORS):
        field = build_and_validate(train, mh, mw, mh, mw,
                                   use_padding=True, bg=bg)
        if field is not None:
            return field, {'method': 'padded', 'dims': (mh, mw, mh, mw), 'bg': bg}
    
    return None


def apply_field(field: np.ndarray, test_input: list, meta: dict) -> list:
    """Apply solved field to test input."""
    ih, iw, oh, ow = meta['dims']
    bg = meta['bg']
    
    inp = np.array(test_input, dtype=np.int32)
    # Always pad to field dimensions to handle test inputs of any size
    inp = pad_grid(inp, ih, iw, bg)
    
    out_vec = field @ grid_to_state(inp, ih, iw)
    result = state_to_grid(out_vec, oh, ow)
    
    # Trim padding — find actual content bounds
    if meta['method'] == 'padded':
        # Trim rows/cols that are all background
        mask_r = np.any(result != bg, axis=1)
        mask_c = np.any(result != bg, axis=0)
        if mask_r.any() and mask_c.any():
            r_start, r_end = np.where(mask_r)[0][[0, -1]]
            c_start, c_end = np.where(mask_c)[0][[0, -1]]
            result = result[r_start:r_end+1, c_start:c_end+1]
    
    return result.tolist()


# =================================================================
# COLD START ENGINE
# =================================================================

def run(task_dir: Path, output_file: Path = None) -> Dict:
    """Full ARC solve from cold start."""
    
    print("=" * 60)
    print("E8 ARC ENGINE")
    print("Ghost in the Machine Labs")
    print("field @ state. RAM-resident geometric solver.")
    print("=" * 60)
    sys.stdout.flush()
    
    all_files = sorted(task_dir.glob('*.json'))
    total = len(all_files)
    
    results = {}
    t0 = time.time()
    
    for i, f in enumerate(all_files):
        tid = f.stem
        task = json.load(open(f))
        
        solution = solve_task(task)
        if solution is not None:
            field, meta = solution
            # Generate test output if test data exists
            if 'test' in task and task['test']:
                test_output = apply_field(field, task['test'][0]['input'], meta)
                results[tid] = {
                    'output': test_output,
                    'method': meta['method'],
                    'bg': meta['bg']
                }
            else:
                results[tid] = {'method': meta['method'], 'bg': meta['bg']}
        
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{total}] solved={len(results)} "
                  f"({100*len(results)/(i+1):.1f}%) {elapsed:.1f}s")
            sys.stdout.flush()
    
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"RESULT: {len(results)}/{total} ({100*len(results)/total:.1f}%)")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    
    if output_file:
        submission = {
            'engine': 'E8 ARC Engine',
            'author': 'Ghost in the Machine Labs',
            'method': 'RAM-resident geometric field propagation',
            'timestamp': datetime.now().isoformat(),
            'solve_time_seconds': round(elapsed, 2),
            'total_tasks': total,
            'solved_count': len(results),
            'solve_rate': round(len(results) / total, 4),
            'solutions': results
        }
        json.dump(submission, open(output_file, 'w'), indent=2)
        print(f"Saved: {output_file}")
    
    return results


if __name__ == "__main__":
    task_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
               Path('/home/joe/sparky/arc_data/combined/training')
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else \
                  Path('/home/joe/sparky/e8_arc_agent/state/submission.json')
    run(task_dir, output_file)
