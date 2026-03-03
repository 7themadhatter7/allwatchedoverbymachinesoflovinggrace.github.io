#!/usr/bin/env python3
"""
Mother's Complete Language — Self-Reading Dictionary + Learned Grammar
Ghost in the Machine Labs

Mother reads her own vocabulary from numpy/scipy at startup.
She learns grammar from 564 solved ARC codes.
She discovers semantic bridges from signature↔code correlations.

No hand-curation. The language defines itself.

VOCABULARY: Introspected from numpy, scipy.ndimage, Python builtins
GRAMMAR:    Extracted from AST patterns in solved codes
SEMANTICS:  Correlated from signature features ↔ code patterns
"""
import numpy as np
from scipy import ndimage
import inspect, re, ast, json
from pathlib import Path

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

# ─── Atomic Primitives ───
from mother_primitives import (
    get_all_primitives,
    grid_shape, unique_colors, color_counts, background_color, foreground_colors,
    has_symmetry_h, has_symmetry_v, has_symmetry_diag, is_periodic,
    count_objects, edge_cells, neighbor_count, neighbor_count_8,
    find_objects, find_objects_by_color, extract_object, extract_largest,
    extract_smallest, extract_by_color, extract_row, extract_col,
    extract_subgrid, extract_unique_pattern, crop_to_content,
    split_grid_h, split_grid_v, find_grid_dividers,
    rotate_90, rotate_180, rotate_270, flip_h, flip_v,
    scale_up, scale_down, tile_grid, mirror_h, mirror_v, mirror_both,
    recolor, swap_colors, fill_color, keep_only, remove_color,
    dilate, erode, get_outline, fill_interior,
    gravity_down, gravity_up, gravity_left, gravity_right,
    pad_grid, remove_border,
    empty_grid, place_object, stack_h, stack_v, overlay,
    stamp_pattern, replace_objects, draw_line, flood_fill,
    detect_transform, detect_color_mapping, detect_scale_factor, detect_tile_pattern,
)


# ─── PART 1: SELF-READING DICTIONARY ────────────────────────────
# Mother probes every callable in numpy/scipy with test grids.
# If it works on a 2D int array → it's a word she can use.

@dataclass
class Word:
    """A single operation Mother can use."""
    name: str           # e.g. 'np.rot90', 'ndimage.label'
    module: str         # 'np', 'ndimage', 'builtin', 'method'
    callable_ref: object  # The actual function
    input_type: str     # 'grid', 'bool_grid', 'scalar', 'pair'
    output_type: str    # 'grid', 'scalar', 'tuple', 'labels'
    output_shape: str   # 'same', 'smaller', 'larger', 'scalar', 'varies'
    params: List[str]   # Parameter names (excluding input)
    category: str       # Geometric meaning category
    frequency: int = 0  # How often used in solved codes


# Test grids for probing ops
_TEST_GRIDS = [
    np.array([[0,1,2],[3,4,5],[6,7,8]], dtype=int),
    np.array([[0,0,1],[0,1,1],[1,0,0]], dtype=int),
    np.array([[1,2,3],[4,5,6]], dtype=int),  # non-square
]
_TEST_BOOL = [g > 0 for g in _TEST_GRIDS]


def _classify_output(inp, out):
    """Classify what an op does to the grid shape."""
    if not isinstance(out, np.ndarray):
        return 'scalar'
    if out.ndim != 2:
        return 'other'
    if out.shape == inp.shape:
        return 'same'
    if out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]:
        return 'smaller'
    if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
        return 'larger'
    return 'varies'


def _categorize_op(name, doc, output_shape):
    """Assign geometric meaning category based on name and behavior."""
    name_lower = name.lower()
    doc_lower = (doc or '')[:500].lower()
    
    # Spatial transforms
    if any(w in name_lower for w in ['rot', 'flip', 'transpose', 'permute', 'swap', 'roll']):
        return 'spatial'
    # Morphology
    if any(w in name_lower for w in ['dilat', 'eros', 'fill', 'binary', 'morpho', 'open', 'clos']):
        return 'morphology'
    # Measurement
    if output_shape == 'scalar' or any(w in name_lower for w in ['count', 'sum', 'mean', 'max', 'min', 'size', 'trace']):
        return 'measure'
    # Masks/logic
    if any(w in name_lower for w in ['where', 'equal', 'greater', 'less', 'logical', 'bitwise', 'isin', 'nonzero']):
        return 'mask'
    # Object detection
    if any(w in name_lower for w in ['label', 'find_object', 'connected', 'argwhere', 'center_of_mass']):
        return 'objects'
    # Color/value manipulation  
    if any(w in name_lower for w in ['unique', 'bincount', 'sort', 'argsort', 'clip', 'histogram']):
        return 'color'
    # Tiling/stacking
    if any(w in name_lower for w in ['tile', 'repeat', 'stack', 'concat', 'pad', 'block']):
        return 'tiling'
    # Extraction
    if any(w in name_lower for w in ['diag', 'tril', 'triu', 'extract', 'take', 'select', 'compress']):
        return 'extraction'
    # Creation
    if any(w in name_lower for w in ['zeros', 'ones', 'full', 'empty', 'identity', 'eye']):
        return 'creation'
    # Filtering
    if any(w in name_lower for w in ['filter', 'convolve', 'correlate', 'median', 'gauss', 'sobel', 'laplace', 'prewitt']):
        return 'filtering'
    # Shape manipulation
    if any(w in name_lower for w in ['reshape', 'ravel', 'flatten', 'squeeze', 'expand']):
        return 'reshape'
    # Distance/geometry
    if any(w in name_lower for w in ['distance', 'gradient']):
        return 'distance'
    # Reduction
    if output_shape in ('smaller', 'scalar'):
        return 'reduction'
    return 'general'


def _probe_numpy():
    """Probe all numpy ops. Returns list of Words that work on 2D int arrays."""
    words = []
    
    # Skip: trig, complex, datetime, file I/O, type aliases, string ops
    SKIP_PATTERNS = {
        'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'power', 'float_power',
        'complex', 'datetime', 'timedelta', 'busday', 'save', 'load', 'file',
        'genfrom', 'fromfile', 'fromstring', 'frombuffer', 'fromregex', 'fromiter',
        'string', 'format', 'set_print', 'info', 'may_share', 'shares_memory',
        'result_type', 'min_scalar', 'common_type', 'mintypecode',
        'void', 'bytes_', 'bool_', 'recarray', 'memmap', 'flatiter',
        'broadcast', 'ufunc', 'dtype', 'generic', 'ndindex', 'ndenumerate',
        'errstate', 'seterr', 'geterr', 'finfo', 'iinfo', 'sctypedict',
        'frexp', 'ldexp', 'modf', 'spacing', 'nextafter', 'signbit',
        'nan', 'inf', 'pi', 'e', 'euler_gamma', 'newaxis',
        'iscomplex', 'isreal', 'isscalar', 'isfortran', 'iterable',
        'promote_types', 'can_cast', 'lookfor', 'source', 'who',
        'deprecat', 'testing', 'char', 'rec', 'ma', 'ctypeslib',
        'acosh', 'asinh', 'atanh', 'arccos', 'arcsin', 'arctan',
        'acos', 'asin', 'atan', 'hypot', 'deg2rad', 'rad2deg',
        'degrees', 'radians', 'unwrap', 'angle', 'cbrt',
        'cosh', 'sinh', 'tanh', 'expm1', 'log1p', 'log2', 'log10',
        'logaddexp', 'ceil', 'floor', 'trunc', 'rint', 'fix',
        'i0', 'heaviside', 'fabs', 'copysign',
        'poly', 'vander', 'trapezoid', 'trapz', 'gradient',
        'corrcoef', 'cov', 'histogram', 'histogram2d', 'histogramdd',
        'histogram_bin_edges', 'percentile', 'quantile', 'nanpercentile',
        'nanquantile', 'nanstd', 'nanvar', 'nanmean', 'nanmedian',
        'nanmin', 'nanmax', 'nansum', 'nanprod', 'nancumprod', 'nancumsum',
        'nanargmin', 'nanargmax', 'average', 'std', 'var', 'median',
        'piecewise', 'interp', 'convolve', 'cross', 'inner', 'outer',
        'matmul', 'dot', 'tensordot', 'einsum', 'vecdot', 'vecmat', 'matvec',
        'kron', 'linalg',
    }
    
    for name in sorted(dir(np)):
        if name.startswith('_'): continue
        if any(s in name.lower() for s in SKIP_PATTERNS): continue
        fn = getattr(np, name, None)
        if fn is None or not callable(fn): continue
        
        # Probe with test grids
        for grid, bgrid in zip(_TEST_GRIDS[:2], _TEST_BOOL[:2]):
            try:
                result = fn(grid)
                if isinstance(result, (np.ndarray, int, float, bool, np.integer, np.floating, np.bool_)):
                    out_shape = _classify_output(grid, result)
                    if out_shape != 'other':
                        doc = getattr(fn, '__doc__', '') or ''
                        cat = _categorize_op(name, doc, out_shape)
                        # Get params
                        try:
                            sig = inspect.signature(fn)
                            params = [p for p in sig.parameters if p not in ('a', 'x', 'ar', 'input', 'self')]
                        except:
                            params = []
                        
                        out_type = 'grid' if isinstance(result, np.ndarray) and result.ndim == 2 else 'scalar'
                        words.append(Word(
                            name=f'np.{name}', module='np', callable_ref=fn,
                            input_type='grid', output_type=out_type,
                            output_shape=out_shape, params=params[:5],
                            category=cat
                        ))
                        break
            except:
                pass
    
    return words


def _probe_ndimage():
    """Probe all scipy.ndimage ops."""
    words = []
    
    SKIP = {'test', 'fourier', 'spline', 'geometric_transform', 'affine_transform',
            'map_coordinates', 'vectorized_filter', 'generic_filter', 'generic_filter1d',
            'generic_gradient_magnitude', 'generic_laplace', 'iterate_structure',
            'generate_binary_structure', 'morphological_laplace', 'morphological_gradient',
            'grey_closing', 'grey_dilation', 'grey_erosion', 'grey_opening',
            'correlate1d', 'convolve1d', 'uniform_filter1d', 'minimum_filter1d',
            'maximum_filter1d', 'gaussian_filter1d', 'gaussian_gradient_magnitude',
            'gaussian_laplace', 'percentile_filter', 'rank_filter',
            'rotate', 'shift', 'zoom',
            'black_tophat', 'white_tophat', 'watershed_ift'}
    
    for name in sorted(dir(ndimage)):
        if name.startswith('_'): continue
        if name in SKIP: continue
        fn = getattr(ndimage, name, None)
        if fn is None or not callable(fn): continue
        
        for grid, bgrid in zip(_TEST_GRIDS[:2], _TEST_BOOL[:2]):
            for test_input in [bgrid, grid]:
                try:
                    result = fn(test_input)
                    doc = getattr(fn, '__doc__', '') or ''
                    cat = _categorize_op(name, doc, 'varies')
                    
                    if isinstance(result, tuple):
                        out_type = 'tuple'
                        out_shape = 'varies'
                    elif isinstance(result, np.ndarray):
                        out_type = 'grid' if result.ndim == 2 else 'array'
                        out_shape = _classify_output(grid, result) if result.ndim == 2 else 'other'
                    else:
                        out_type = 'scalar'
                        out_shape = 'scalar'
                    
                    inp_type = 'bool_grid' if test_input.dtype == bool else 'grid'
                    
                    try:
                        sig = inspect.signature(fn)
                        params = [p for p in sig.parameters if p not in ('input', 'self')]
                    except:
                        params = []
                    
                    words.append(Word(
                        name=f'ndimage.{name}', module='ndimage', callable_ref=fn,
                        input_type=inp_type, output_type=out_type,
                        output_shape=out_shape, params=params[:5],
                        category=cat
                    ))
                    break
                except:
                    pass
            else:
                continue
            break
    
    return words


def _builtin_words():
    """Python builtins critical for ARC reasoning — the REAL language."""
    words = []
    
    builtins = [
        # Iteration — the backbone of ARC reasoning
        ('for_range', 'builtin', 'iterate', 'grid', 'grid', 'same',
         'for i in range(h): for j in range(w):'),
        ('for_enumerate', 'builtin', 'iterate', 'grid', 'grid', 'same',
         'for i, row in enumerate(grid):'),
        ('while_stable', 'builtin', 'iterate', 'grid', 'grid', 'same',
         'while changed: ... changed = (prev != grid).any()'),
        
        # Collection ops — how Mother thinks about sets of things
        ('Counter', 'builtin', 'measure', 'grid', 'dict', 'scalar',
         'Counter(grid.flatten())'),
        ('sorted_by', 'builtin', 'color', 'list', 'list', 'same',
         'sorted(items, key=lambda x: criterion)'),
        ('set_ops', 'builtin', 'mask', 'list', 'set', 'varies',
         'set(a) - set(b), set(a) & set(b), set(a) | set(b)'),
        ('zip_grids', 'builtin', 'spatial', 'pair', 'grid', 'same',
         'zip(grid_a, grid_b)'),
        ('enumerate_items', 'builtin', 'iterate', 'list', 'list', 'same',
         'enumerate(items)'),
        
        # Aggregation — how Mother summarizes
        ('max_val', 'builtin', 'measure', 'list', 'scalar', 'scalar',
         'max(items) or max(items, key=fn)'),
        ('min_val', 'builtin', 'measure', 'list', 'scalar', 'scalar',
         'min(items) or min(items, key=fn)'),
        ('sum_val', 'builtin', 'measure', 'list', 'scalar', 'scalar',
         'sum(items)'),
        ('len_val', 'builtin', 'measure', 'list', 'scalar', 'scalar',
         'len(items)'),
        ('any_check', 'builtin', 'mask', 'list', 'scalar', 'scalar',
         'any(condition for x in items)'),
        ('all_check', 'builtin', 'mask', 'list', 'scalar', 'scalar',
         'all(condition for x in items)'),
        ('abs_val', 'builtin', 'measure', 'scalar', 'scalar', 'scalar',
         'abs(x)'),
        
        # List comprehensions — Mother's sentence builder
        ('list_comp', 'builtin', 'general', 'list', 'list', 'varies',
         '[f(x) for x in items if condition]'),
        ('nested_comp', 'builtin', 'general', 'grid', 'grid', 'same',
         '[[f(cell) for cell in row] for row in grid]'),
        ('dict_comp', 'builtin', 'general', 'list', 'dict', 'varies',
         '{k: v for k, v in items}'),
        
        # Grid access patterns — how Mother reads
        ('row_slice', 'builtin', 'extraction', 'grid', 'grid', 'smaller',
         'grid[r1:r2]'),
        ('col_slice', 'builtin', 'extraction', 'grid', 'grid', 'smaller',
         'grid[:, c1:c2]'),
        ('cell_access', 'builtin', 'extraction', 'grid', 'scalar', 'scalar',
         'grid[r][c]'),
        ('subgrid', 'builtin', 'extraction', 'grid', 'grid', 'smaller',
         'grid[r1:r2, c1:c2]'),
        
        # Grid construction — how Mother writes
        ('list_to_grid', 'builtin', 'creation', 'list', 'grid', 'varies',
         'np.array(list_of_lists)'),
        ('grid_copy', 'builtin', 'creation', 'grid', 'grid', 'same',
         'grid.copy() or [row[:] for row in grid]'),
        ('append_row', 'builtin', 'tiling', 'grid', 'grid', 'larger',
         'result.append(row)'),
        ('insert_row', 'builtin', 'tiling', 'grid', 'grid', 'larger',
         'result.insert(idx, row)'),
        
        # Conditional logic — how Mother decides
        ('if_else', 'builtin', 'general', 'scalar', 'scalar', 'scalar',
         'x if condition else y'),
        ('ternary_grid', 'builtin', 'general', 'grid', 'grid', 'same',
         '[[a if cond else b for ...] for ...]'),
        
        # Type conversion — Mother's translator
        ('tolist', 'builtin', 'general', 'grid', 'list', 'same',
         'grid.tolist()'),
        ('flatten', 'builtin', 'reshape', 'grid', 'list', 'scalar',
         'grid.flatten().tolist()'),
        ('astype', 'builtin', 'general', 'grid', 'grid', 'same',
         'grid.astype(int)'),
    
        # === ARC EXPANSION: Color ===
        ('color_count', 'builtin', 'color', 'grid', 'dict', 'scalar',
         'Counter(grid.flatten())'),
        ('unique_colors', 'builtin', 'color', 'grid', 'list', 'scalar',
         'list(set(grid.flatten()))'),
        ('color_mask', 'builtin', 'color', 'grid', 'grid', 'same',
         '(grid == color).astype(int)'),
        ('recolor', 'builtin', 'color', 'grid', 'grid', 'same',
         'np.where(grid == old, new, grid)'),
        ('majority_color', 'builtin', 'color', 'grid', 'scalar', 'scalar',
         'Counter(grid.flatten()).most_common(1)[0][0]'),
        ('background_color', 'builtin', 'color', 'grid', 'scalar', 'scalar',
         'Counter(grid.flatten()).most_common(1)[0][0]'),
        ('color_map', 'builtin', 'color', 'grid', 'grid', 'same',
         'np.vectorize(mapping.get)(grid)'),
        # === Mask expanded ===
        ('where_color', 'builtin', 'mask', 'grid', 'list', 'varies',
         'list(zip(*np.where(grid == color)))'),
        ('nonzero_mask', 'builtin', 'mask', 'grid', 'grid', 'same',
         '(grid != 0).astype(int)'),
        ('mask_apply', 'builtin', 'mask', 'grid', 'grid', 'same',
         'np.where(mask, grid, fill)'),
        ('mask_invert', 'builtin', 'mask', 'grid', 'grid', 'same',
         '1 - mask'),
        ('mask_and', 'builtin', 'mask', 'grid', 'grid', 'same',
         'mask1 & mask2'),
        ('mask_or', 'builtin', 'mask', 'grid', 'grid', 'same',
         'mask1 | mask2'),
        ('boundary_mask', 'builtin', 'mask', 'grid', 'grid', 'same',
         'dilate(mask) & ~mask'),
        # === Objects expanded ===
        ('connected_components', 'builtin', 'objects', 'grid', 'grid', 'same',
         'ndimage.label(grid != bg)[0]'),
        ('object_count', 'builtin', 'objects', 'grid', 'scalar', 'scalar',
         'ndimage.label(grid != bg)[1]'),
        ('bounding_box', 'builtin', 'objects', 'grid', 'list', 'scalar',
         'ndimage.find_objects(labels)'),
        ('crop_object', 'builtin', 'objects', 'grid', 'grid', 'smaller',
         'grid[r1:r2, c1:c2]'),
        # === Pattern matching ===
        ('find_pattern', 'builtin', 'detection', 'grid', 'list', 'varies',
         'find all occurrences of subgrid'),
        ('repeating_unit', 'builtin', 'detection', 'grid', 'grid', 'smaller',
         'find minimal tiling unit'),
        ('find_dividers', 'builtin', 'detection', 'grid', 'list', 'varies',
         'find uniform color rows/cols'),
        # === Symmetry ===
        ('has_h_symmetry', 'builtin', 'detection', 'grid', 'scalar', 'scalar',
         'np.array_equal(grid, np.flipud(grid))'),
        ('has_v_symmetry', 'builtin', 'detection', 'grid', 'scalar', 'scalar',
         'np.array_equal(grid, np.fliplr(grid))'),
        ('symmetrize_h', 'builtin', 'spatial', 'grid', 'grid', 'same',
         'mirror top half to bottom'),
        ('symmetrize_v', 'builtin', 'spatial', 'grid', 'grid', 'same',
         'mirror left half to right'),
        ('complete_symmetry', 'builtin', 'spatial', 'grid', 'grid', 'same',
         'fill missing cells for symmetry'),
        # === Grid ops ===
        ('pad_grid', 'builtin', 'spatial', 'grid', 'grid', 'larger',
         'np.pad(grid, pad_width, constant_values=fill)'),
        ('trim_grid', 'builtin', 'spatial', 'grid', 'grid', 'smaller',
         'remove border uniform color rows/cols'),
        ('tile_grid', 'builtin', 'tiling', 'grid', 'grid', 'larger',
         'np.tile(grid, (rows, cols))'),
        # === Analysis ===
        ('row_counts', 'builtin', 'measure', 'grid', 'list', 'varies',
         '[Counter(row) for row in grid]'),
        ('col_counts', 'builtin', 'measure', 'grid', 'list', 'varies',
         '[Counter(col) for col in grid.T]'),
        ('grid_diff', 'builtin', 'measure', 'grid', 'grid', 'same',
         'element-wise difference between grids'),
        ('grid_equal', 'builtin', 'measure', 'grid', 'scalar', 'scalar',
         'np.array_equal(grid1, grid2)'),
]
    
    for name, module, cat, inp, out, shape, snippet in builtins:
        words.append(Word(
            name=name, module=module, callable_ref=None,
            input_type=inp, output_type=out,
            output_shape=shape, params=[],
            category=cat, frequency=0
        ))
    
    return words



def _primitive_words():
    """Build Words from the atomic primitive registry."""
    words = []
    registry = get_all_primitives()
    
    # Category mapping: primitive level -> Word category
    level_to_cat = {
        'perception': 'measure',
        'extraction': 'extraction',
        'transformation': 'spatial',
        'construction': 'creation',
        'detection': 'detection',
    }
    
    # Input/output type mapping for each primitive
    prim_signatures = {
        # Perception
        'grid_shape': ('grid', 'tuple', 'scalar', []),
        'unique_colors': ('grid', 'list', 'scalar', []),
        'color_counts': ('grid', 'dict', 'scalar', []),
        'background_color': ('grid', 'scalar', 'scalar', []),
        'foreground_colors': ('grid', 'list', 'scalar', []),
        'has_symmetry_h': ('grid', 'scalar', 'scalar', []),
        'has_symmetry_v': ('grid', 'scalar', 'scalar', []),
        'has_symmetry_diag': ('grid', 'scalar', 'scalar', []),
        'is_periodic': ('grid', 'tuple', 'scalar', []),
        'count_objects': ('grid', 'scalar', 'scalar', ['bg']),
        'edge_cells': ('grid', 'grid', 'same', []),
        'neighbor_count': ('grid', 'grid', 'same', ['bg']),
        'neighbor_count_8': ('grid', 'grid', 'same', ['bg']),
        # Extraction
        'find_objects': ('grid', 'list', 'varies', ['bg', 'connectivity']),
        'find_objects_by_color': ('grid', 'dict', 'varies', ['bg']),
        'extract_object': ('grid', 'grid', 'smaller', ['obj', 'bg']),
        'extract_largest': ('grid', 'grid', 'smaller', ['bg']),
        'extract_smallest': ('grid', 'grid', 'smaller', ['bg']),
        'extract_by_color': ('grid', 'grid', 'smaller', ['color', 'bg']),
        'extract_row': ('grid', 'grid', 'smaller', ['r']),
        'extract_col': ('grid', 'grid', 'smaller', ['c']),
        'extract_subgrid': ('grid', 'grid', 'smaller', ['r1', 'c1', 'r2', 'c2']),
        'extract_unique_pattern': ('grid', 'grid', 'smaller', ['bg']),
        'crop_to_content': ('grid', 'grid', 'smaller', ['bg']),
        'split_grid_h': ('grid', 'list', 'varies', []),
        'split_grid_v': ('grid', 'list', 'varies', []),
        'find_grid_dividers': ('grid', 'dict', 'scalar', []),
        # Transformation
        'rotate_90': ('grid', 'grid', 'same', []),
        'rotate_180': ('grid', 'grid', 'same', []),
        'rotate_270': ('grid', 'grid', 'same', []),
        'flip_h': ('grid', 'grid', 'same', []),
        'flip_v': ('grid', 'grid', 'same', []),
        'transpose': ('grid', 'grid', 'same', []),
        'scale_up': ('grid', 'grid', 'larger', ['factor']),
        'scale_down': ('grid', 'grid', 'smaller', ['factor']),
        'tile_grid': ('grid', 'grid', 'larger', ['rows', 'cols']),
        'mirror_h': ('grid', 'grid', 'larger', []),
        'mirror_v': ('grid', 'grid', 'larger', []),
        'mirror_both': ('grid', 'grid', 'larger', []),
        'recolor': ('grid', 'grid', 'same', ['mapping']),
        'swap_colors': ('grid', 'grid', 'same', ['c1', 'c2']),
        'fill_color': ('grid', 'grid', 'same', ['target', 'replacement']),
        'keep_only': ('grid', 'grid', 'same', ['color', 'bg']),
        'remove_color': ('grid', 'grid', 'same', ['color', 'bg']),
        'dilate': ('grid', 'grid', 'same', ['bg']),
        'erode': ('grid', 'grid', 'same', ['bg']),
        'get_outline': ('grid', 'grid', 'same', ['bg']),
        'fill_interior': ('grid', 'grid', 'same', ['fill_color', 'bg']),
        'gravity_down': ('grid', 'grid', 'same', ['bg']),
        'gravity_up': ('grid', 'grid', 'same', ['bg']),
        'gravity_left': ('grid', 'grid', 'same', ['bg']),
        'gravity_right': ('grid', 'grid', 'same', ['bg']),
        'pad_grid': ('grid', 'grid', 'larger', ['n', 'val']),
        'remove_border': ('grid', 'grid', 'smaller', ['n']),
        # Construction
        'empty_grid': ('scalar', 'grid', 'varies', ['h', 'w', 'val']),
        'place_object': ('grid', 'grid', 'same', ['obj', 'r', 'c', 'bg']),
        'stack_h': ('pair', 'grid', 'larger', []),
        'stack_v': ('pair', 'grid', 'larger', []),
        'overlay': ('pair', 'grid', 'same', ['bg']),
        'stamp_pattern': ('grid', 'grid', 'same', ['pattern', 'positions', 'bg']),
        'replace_objects': ('grid', 'grid', 'same', ['template', 'bg']),
        'draw_line': ('grid', 'grid', 'same', ['r1', 'c1', 'r2', 'c2', 'color']),
        'flood_fill': ('grid', 'grid', 'same', ['r', 'c', 'new_color']),
        # Detection
        'detect_transform': ('pair', 'scalar', 'scalar', []),
        'detect_color_mapping': ('pair', 'dict', 'scalar', []),
        'detect_scale_factor': ('pair', 'scalar', 'scalar', []),
        'detect_tile_pattern': ('pair', 'tuple', 'scalar', []),
    }
    
    for level, ops in registry.items():
        cat = level_to_cat.get(level, 'general')
        for name, fn in ops.items():
            sig = prim_signatures.get(name, ('grid', 'grid', 'varies', []))
            words.append(Word(
                name=f'prim.{name}',
                module='primitive',
                callable_ref=fn,
                input_type=sig[0],
                output_type=sig[1],
                output_shape=sig[2],
                params=sig[3],
                category=cat,
                frequency=0,
            ))
    
    return words

def build_dictionary():
    """Build complete dictionary by introspecting runtime modules."""
    np_words = _probe_numpy()
    nd_words = _probe_ndimage()
    bi_words = _builtin_words()
    
    prim_words = _primitive_words()
    all_words = np_words + nd_words + bi_words + prim_words
    
    # Deduplicate by name
    seen = set()
    unique = []
    for w in all_words:
        if w.name not in seen:
            seen.add(w.name)
            unique.append(w)
    
    return unique


# ─── PART 2: LEARNED GRAMMAR ────────────────────────────────────
# Mother reads solved codes, extracts AST patterns, learns what
# sentence structures produce working ARC solutions.

@dataclass
class GrammarPattern:
    """A composition pattern learned from solved codes."""
    name: str
    skeleton: str           # Abstract pattern template
    frequency: int          # How many codes use this pattern
    example_tids: List[str] # Example task IDs that use it
    ops_used: Set[str]      # Which words appear in this pattern
    signature_features: Dict[str, float]  # Average sig features when this pattern works


def _extract_patterns_from_code(code: str) -> List[str]:
    """Extract abstract composition patterns from a code string."""
    patterns = []
    
    # Pattern: for_loop + conditional + assignment (most common ARC pattern)
    if re.search(r'for\s+\w+\s+in\s+range', code):
        if 'if ' in code:
            patterns.append('iterate_conditional_assign')
        else:
            patterns.append('iterate_assign')
    
    # Pattern: color mapping via Counter
    if 'Counter' in code and ('most_common' in code or 'items()' in code):
        patterns.append('color_count_map')
    
    # Pattern: object labeling + per-object transform
    if 'ndimage.label' in code or 'label(' in code:
        if 'for' in code:
            patterns.append('label_iterate_objects')
        else:
            patterns.append('label_extract')
    
    # Pattern: np.where color replacement
    if 'np.where' in code:
        patterns.append('conditional_replace')
    
    # Pattern: grid construction from scratch
    if ('np.zeros' in code or 'np.full' in code) and 'for' in code:
        patterns.append('construct_grid')
    
    # Pattern: copy + modify
    if '.copy()' in code and ('for' in code or 'np.where' in code):
        patterns.append('copy_modify')
    
    # Pattern: subgrid extraction via slicing
    if re.search(r'\[\s*\w+\s*:\s*\w+\s*\]', code) or re.search(r'\[\s*\w+\s*:\s*\w+\s*,', code):
        patterns.append('slice_extract')
    
    # Pattern: tiling/stacking
    if any(w in code for w in ['np.tile', 'np.hstack', 'np.vstack', 'np.concatenate']):
        patterns.append('tile_stack')
    
    # Pattern: rotation/flip/transpose
    if any(w in code for w in ['np.rot90', 'np.fliplr', 'np.flipud', '.T', 'transpose']):
        patterns.append('spatial_transform')
    
    # Pattern: while loop (iterative until stable)
    if 'while' in code:
        patterns.append('iterate_until_stable')
    
    # Pattern: sorted/reorder
    if 'sorted(' in code or '.sort()' in code:
        patterns.append('sort_reorder')
    
    # Pattern: nested list comprehension (grid building)
    if '[[' in code and 'for' in code:
        patterns.append('nested_comprehension')
    
    # Pattern: zip (parallel iteration)
    if 'zip(' in code:
        patterns.append('parallel_iterate')
    
    # Pattern: set operations
    if 'set(' in code:
        patterns.append('set_logic')
    
    # Pattern: .append building
    if '.append(' in code:
        patterns.append('incremental_build')
    
    # Pattern: np.array_equal comparison
    if 'array_equal' in code or '== ' in code:
        patterns.append('comparison_check')
    
    # Pattern: shape-dependent branching
    if '.shape' in code and 'if' in code:
        patterns.append('shape_conditional')
    
    # Pattern: min/max finding
    if ('max(' in code or 'min(' in code) and 'for' in code:
        patterns.append('extremum_search')
    
    # Pattern: flood fill / BFS / DFS  
    if any(w in code for w in ['stack.append', 'queue', 'visited', 'neighbors']):
        patterns.append('flood_fill_search')
    
    return patterns if patterns else ['direct_transform']


def learn_grammar(codes: Dict[str, str], task_sigs: Dict[str, np.ndarray]) -> List[GrammarPattern]:
    """Learn grammar patterns from solved codes + their signatures."""
    pattern_data = defaultdict(lambda: {
        'count': 0, 'tids': [], 'ops': set(), 'sigs': []
    })
    
    for tid, code in codes.items():
        patterns = _extract_patterns_from_code(code)
        
        # Extract which numpy/scipy ops this code uses
        ops = set()
        for m in re.finditer(r'np\.([a-z_]+)', code):
            ops.add(f'np.{m.group(1)}')
        for m in re.finditer(r'ndimage\.([a-z_]+)', code):
            ops.add(f'ndimage.{m.group(1)}')
        
        sig = task_sigs.get(tid)
        
        for pat in patterns:
            pd = pattern_data[pat]
            pd['count'] += 1
            pd['tids'].append(tid)
            pd['ops'].update(ops)
            if sig is not None:
                pd['sigs'].append(sig)
    
    # Convert to GrammarPattern objects
    grammar = []
    for name, pd in sorted(pattern_data.items(), key=lambda x: -x[1]['count']):
        avg_sig = {}
        if pd['sigs']:
            sig_array = np.array(pd['sigs'])
            avg_sig = {
                'same_size': float(np.mean(np.abs(sig_array[:, 6] - 1.0) < 0.01)),
                'has_objects': float(np.mean(sig_array[:, 40] > 0.1)),
                'color_change': float(np.mean(sig_array[:, 11] > 0.01)),
                'size_change': float(np.mean(np.abs(sig_array[:, 6] - 1.0) > 0.01)),
                'has_symmetry': float(np.mean(sig_array[:, 16] > 0.5)),
            }
        
        grammar.append(GrammarPattern(
            name=name,
            skeleton=_pattern_skeleton(name),
            frequency=pd['count'],
            example_tids=pd['tids'][:5],
            ops_used=pd['ops'],
            signature_features=avg_sig
        ))
    
    return grammar


def _pattern_skeleton(name):
    """Return abstract code skeleton for a grammar pattern."""
    skeletons = {
        'iterate_conditional_assign': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if {CONDITION}:
                out[i,j] = {VALUE}
    return out.tolist()''',
        
        'iterate_assign': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w):
            out[i,j] = {TRANSFORM}
    return out.tolist()''',
        
        'color_count_map': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    
    from collections import Counter
    counts = Counter(g.flatten())
    {COLOR_MAP_LOGIC}
    return out.tolist()''',
        
        'label_iterate_objects': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    from scipy import ndimage
    labeled, n = ndimage.label(g {LABEL_MASK})
    out = g.copy()
    for obj_id in range(1, n+1):
        mask = labeled == obj_id
        {PER_OBJECT_TRANSFORM}
    return out.tolist()''',
        
        'conditional_replace': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = np.where({CONDITION}, {TRUE_VAL}, {FALSE_VAL})
    return out.tolist()''',
        
        'construct_grid': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = np.zeros(({OUT_H}, {OUT_W}), dtype=int)
    {FILL_LOGIC}
    return out.tolist()''',
        
        'copy_modify': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = g.copy()
    {MODIFICATIONS}
    return out.tolist()''',
        
        'slice_extract': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    {FIND_BOUNDS}
    out = g[r1:r2, c1:c2]
    return out.tolist()''',
        
        'tile_stack': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = np.tile(g, ({TILE_H}, {TILE_W}))
    return out.tolist()''',
        
        'spatial_transform': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = {SPATIAL_OP}(g{PARAMS})
    return out.tolist()''',
        
        'iterate_until_stable': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    changed = True
    while changed:
        prev = g.copy()
        {ITERATION_BODY}
        changed = not np.array_equal(prev, g)
    return g.tolist()''',
        
        'sort_reorder': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    {SORT_LOGIC}
    return out.tolist()''',
        
        'nested_comprehension': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = [[{CELL_EXPR} for j in range(g.shape[1])] for i in range(g.shape[0])]
    return out''',
        
        'parallel_iterate': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = []
    for row_a, row_b in zip({ITER_A}, {ITER_B}):
        out.append({ROW_COMBINE})
    return out''',
        
        'incremental_build': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    out = []
    {BUILD_LOGIC}
    return out''',
        
        'flood_fill_search': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    visited = set()
    {SEARCH_LOGIC}
    return out.tolist()''',
        
        'direct_transform': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    return {TRANSFORM}.tolist()''',
        
        'label_extract': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    from scipy import ndimage
    labeled, n = ndimage.label(g {LABEL_MASK})
    {EXTRACT_LOGIC}
    return out.tolist()''',
        
        'set_logic': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    {SET_OPERATIONS}
    return out.tolist()''',
        
        'comparison_check': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    {COMPARISON_LOGIC}
    return out.tolist()''',
        
        'shape_conditional': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    if {SHAPE_CONDITION}:
        {BRANCH_A}
    else:
        {BRANCH_B}
    return out.tolist()''',
        
        'extremum_search': '''
def solve(grid):
    g = np.array(grid, dtype=int)
    {FIND_EXTREMUM}
    return out.tolist()''',
    }
    return skeletons.get(name, skeletons['direct_transform'])


# ─── PART 3: SEMANTIC BRIDGE ────────────────────────────────────
# Correlates signature features with which words and grammar
# patterns are most likely to produce correct solutions.

@dataclass  
class SemanticBridge:
    """Maps a geometric signature feature to relevant vocabulary + grammar."""
    feature: str
    condition: str          # How to detect this feature in a signature
    relevant_words: List[str]    # Word names most useful for this feature
    relevant_grammar: List[str]  # Grammar patterns most useful
    weight: float           # How strongly this feature should influence selection


def build_semantic_bridges(words: List[Word], grammar: List[GrammarPattern],
                           codes: Dict[str, str], task_sigs: Dict[str, np.ndarray]) -> List[SemanticBridge]:
    """Build bridges by correlating sig features with code patterns in solved tasks."""
    bridges = []
    
    # For each signature feature, find which ops/patterns correlate with it
    sig_features = {
        'same_size': lambda s: abs(s[6]-1.0) < 0.01 and abs(s[7]-1.0) < 0.01,
        'shrinks': lambda s: s[6] < 0.95 or s[7] < 0.95,
        'grows': lambda s: s[6] > 1.05 or s[7] > 1.05,
        'color_changes': lambda s: s[11] > 0.01,
        'no_color_change': lambda s: s[11] < 0.01,
        'has_objects': lambda s: s[40] > 0.1,
        'many_colors': lambda s: s[8] > 0.3,
        'few_colors': lambda s: s[8] < 0.2,
        'h_symmetry': lambda s: s[16] > 0.5,
        'v_symmetry': lambda s: s[17] > 0.5,
        'high_edge_density': lambda s: s[32] > 0.3,
        'tile_pattern': lambda s: s[30] > 0.5,
        'small_output': lambda s: s[2] * 30 < 5 and s[3] * 30 < 5 and s[2] > 0,
    }
    
    for feat_name, feat_fn in sig_features.items():
        # Find which solved tasks have this feature
        matching_tids = []
        for tid, sig in task_sigs.items():
            if tid in codes:
                try:
                    if feat_fn(sig):
                        matching_tids.append(tid)
                except:
                    pass
        
        if not matching_tids:
            continue
        
        # Count which ops appear in codes with this feature
        op_counts = Counter()
        pattern_counts = Counter()
        for tid in matching_tids:
            code = codes[tid]
            for m in re.finditer(r'np\.([a-z_]+)', code):
                op_counts[f'np.{m.group(1)}'] += 1
            for m in re.finditer(r'ndimage\.([a-z_]+)', code):
                op_counts[f'ndimage.{m.group(1)}'] += 1
            for pat in _extract_patterns_from_code(code):
                pattern_counts[pat] += 1
        
        # Top ops and patterns for this feature
        top_ops = [op for op, _ in op_counts.most_common(15)]
        top_patterns = [pat for pat, _ in pattern_counts.most_common(8)]
        
        bridges.append(SemanticBridge(
            feature=feat_name,
            condition=f'sig_features["{feat_name}"](sig)',
            relevant_words=top_ops,
            relevant_grammar=top_patterns,
            weight=len(matching_tids) / max(len(codes), 1)
        ))
    
    return bridges


# ─── PART 4: MOTHER COMPLETE ────────────────────────────────────
# Puts it all together. Mother reads signature, selects vocabulary
# via semantic bridges, fills grammar skeletons, produces code.

class MotherComplete:
    """Mother's complete language system — self-reading, self-learning."""
    
    def __init__(self, words, grammar, bridges, codes=None):
        self.words = {w.name: w for w in words}
        self.words_by_cat = defaultdict(list)
        for w in words:
            self.words_by_cat[w.category].append(w)
        self.grammar = {g.name: g for g in grammar}
        self.grammar_ranked = sorted(grammar, key=lambda g: -g.frequency)
        self.bridges = bridges
        self.codes = codes or {}
        
        # Index words by output shape
        self.words_same = [w for w in words if w.output_shape == 'same']
        self.words_smaller = [w for w in words if w.output_shape == 'smaller']
        self.words_larger = [w for w in words if w.output_shape == 'larger']
        
        # Signature feature functions
        self.sig_features = {
            'same_size': lambda s: abs(s[6]-1.0) < 0.01 and abs(s[7]-1.0) < 0.01,
            'shrinks': lambda s: s[6] < 0.95 or s[7] < 0.95,
            'grows': lambda s: s[6] > 1.05 or s[7] > 1.05,
            'color_changes': lambda s: s[11] > 0.01,
            'no_color_change': lambda s: s[11] < 0.01,
            'has_objects': lambda s: s[40] > 0.1,
            'many_colors': lambda s: s[8] > 0.3,
            'few_colors': lambda s: s[8] < 0.2,
            'h_symmetry': lambda s: s[16] > 0.5,
            'v_symmetry': lambda s: s[17] > 0.5,
            'high_edge_density': lambda s: s[32] > 0.3,
            'tile_pattern': lambda s: s[30] > 0.5,
            'small_output': lambda s: s[2] * 30 < 5 and s[3] * 30 < 5 and s[2] > 0,
        }
    
    def read_signature(self, sig):
        """Read geometric signature → active features."""
        active = {}
        for name, fn in self.sig_features.items():
            try:
                active[name] = fn(sig)
            except:
                active[name] = False
        return active
    
    def select_vocabulary(self, features):
        """Select relevant words based on active features."""
        selected = set()
        for bridge in self.bridges:
            if features.get(bridge.feature, False):
                for word_name in bridge.relevant_words:
                    if word_name in self.words:
                        selected.add(word_name)
        
        # Always include core operations
        core = ['np.array', 'np.zeros', 'np.where', 'np.copy',
                'for_range', 'list_comp', 'if_else', 'Counter',
                'tolist', 'grid_copy']
        for c in core:
            if c in self.words:
                selected.add(c)
        
        return selected
    
    def select_grammar(self, features):
        """Select relevant grammar patterns based on active features."""
        selected = set()
        for bridge in self.bridges:
            if features.get(bridge.feature, False):
                for pat in bridge.relevant_grammar:
                    if pat in self.grammar:
                        selected.add(pat)
        
        # Always include the most common patterns
        for gp in self.grammar_ranked[:5]:
            selected.add(gp.name)
        
        return selected
    
    def compose(self, task, sig, exclude=None):
        """Generate candidate solutions. The main interface."""
        features = self.read_signature(sig)
        vocab = self.select_vocabulary(features)
        grammars = self.select_grammar(features)
        exclude = exclude or set()
        
        candidates = []
        train = task.get('train', [])
        if not train:
            return candidates
        
        inp0 = np.array(train[0]['input'], dtype=int)
        out0 = np.array(train[0]['output'], dtype=int)
        h_in, w_in = inp0.shape
        h_out, w_out = out0.shape
        same_size = (h_in == h_out and w_in == w_out)
        
        # Learn color mapping from first training pair
        color_map = {}
        if same_size:
            for i in range(h_in):
                for j in range(w_in):
                    ci, co = int(inp0[i,j]), int(out0[i,j])
                    if ci != co:
                        if ci not in color_map:
                            color_map[ci] = co
        
        # Learn background color (most common in input)
        
        from collections import Counter as C
        in_counts = C(inp0.flatten().tolist())
        bg = in_counts.most_common(1)[0][0] if in_counts else 0
        out_counts = C(out0.flatten().tolist())
        
        # Colors present
        in_colors = set(inp0.flatten().tolist())
        out_colors = set(out0.flatten().tolist())
        new_colors = out_colors - in_colors
        lost_colors = in_colors - out_colors
        
        # Generate candidates from each selected grammar pattern
        for gname in grammars:
            try:
                new_candidates = self._fill_pattern(
                    gname, task, inp0, out0, bg, color_map,
                    in_colors, out_colors, new_colors, features, vocab
                )
                for c in new_candidates:
                    if c not in exclude and c not in candidates:
                        candidates.append(c)
            except:
                pass
        

        # ─── Atomic Primitive Candidates ───
        try:
            prim_cands = _generate_primitive_candidates(task, inp0, out0, bg, color_map, features)
            for c in prim_cands:
                h = hash(c)
                if h not in exclude and c not in candidates:
                    candidates.append(c)
        except:
            pass
        
        # Core Knowledge priors (Spelke/Chollet)
        try:
            ck_candidates = generate_core_knowledge_candidates(task, sig, bg)
            for c in ck_candidates:
                if c not in exclude and c not in candidates:
                    candidates.append(c)
        except:
            pass
        return candidates
    
    def _fill_pattern(self, pattern_name, task, inp0, out0, bg, color_map,
                      in_colors, out_colors, new_colors, features, vocab):
        """Fill a grammar pattern with concrete operations. Returns list of code strings."""
        candidates = []
        train = task['train']
        h_in, w_in = inp0.shape
        h_out, w_out = out0.shape
        same_size = (h_in == h_out and w_in == w_out)
        
        from collections import Counter as C
        
        if pattern_name == 'iterate_conditional_assign' and same_size:
            # Color map: pixel-by-pixel replacement
            if color_map:
                map_str = str(color_map)
                candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    cmap = {map_str}
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if g[i,j] in cmap:
                out[i,j] = cmap[g[i,j]]
    return out.tolist()''')
            
            # Neighbor-based rules: if surrounded by X, become Y
            # Check what each changed pixel's neighbors look like
            if color_map:
                for old_c, new_c in color_map.items():
                    candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for i in range(h):
        for j in range(w):
            if g[i,j] == {old_c}:
                # Count neighbors
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbors.append(g[ni,nj])
                if len(neighbors) > 0 and all(n != {old_c} and n != {bg} for n in neighbors):
                    out[i,j] = {new_c}
    return out.tolist()''')
        
        elif pattern_name == 'conditional_replace' and same_size:
            # np.where replacements
            if color_map:
                for old_c, new_c in color_map.items():
                    candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    out = np.where(g == {old_c}, {new_c}, g)
    return out.tolist()''')
            
            # Replace background with something
            if new_colors:
                for nc in new_colors:
                    candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    out = np.where(g == {bg}, {nc}, g)
    return out.tolist()''')
        
        elif pattern_name == 'color_count_map':
            # Most common color operations
            candidates.append(f'''def solve(grid):
    import numpy as np
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    counts = Counter(g.flatten().tolist())
    bg = counts.most_common(1)[0][0]
    out = np.where(g == bg, 0, g)
    return out.tolist()''')
            
            # Replace minority colors
            candidates.append(f'''def solve(grid):
    import numpy as np
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(g.flatten().tolist())
    if len(counts) >= 2:
        bg = counts.most_common(1)[0][0]
        minority = counts.most_common()[-1][0]
        majority_nonbg = [c for c, n in counts.most_common() if c != bg][0] if len(counts) > 2 else bg
        out = np.where(g == minority, majority_nonbg, g)
        return out.tolist()
    return g.tolist()''')
        
        elif pattern_name == 'label_iterate_objects':
            # Per-object: fill, extract, recolor
            candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    out = g.copy()
    for obj_id in range(1, n+1):
        obj_mask = labeled == obj_id
        coords = np.argwhere(obj_mask)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        # Fill bounding box
        for i in range(r1, r2):
            for j in range(c1, c2):
                if out[i,j] == {bg}:
                    out[i,j] = g[coords[0][0], coords[0][1]]
    return out.tolist()''')
            
            if not same_size:
                # Extract largest/smallest object
                for which in ['largest', 'smallest']:
                    candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    sizes = []
    for obj_id in range(1, n+1):
        sizes.append((obj_id, int((labeled == obj_id).sum())))
    sizes.sort(key=lambda x: x[1], reverse={'True' if which == 'largest' else 'False'})
    target = sizes[0][0]
    obj_mask = labeled == target
    coords = np.argwhere(obj_mask)
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0) + 1
    return g[r1:r2, c1:c2].tolist()''')
        
        elif pattern_name == 'spatial_transform':
            # All spatial transforms
            for op in ['np.rot90(g)', 'np.rot90(g, 2)', 'np.rot90(g, 3)',
                       'np.fliplr(g)', 'np.flipud(g)', 'g.T',
                       'np.fliplr(np.flipud(g))', 'np.rot90(np.fliplr(g))']:
                candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return {op}.tolist()''')
        
        elif pattern_name == 'tile_stack':
            # Try exact tiling ratios
            if h_out > h_in and w_out > w_in:
                rh = h_out / h_in
                rw = w_out / w_in
                if abs(rh - round(rh)) < 0.01 and abs(rw - round(rw)) < 0.01:
                    candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.tile(g, ({int(round(rh))}, {int(round(rw))})).tolist()''')
            
            # Horizontal/vertical stacking
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.hstack([g, np.fliplr(g)]).tolist()''')
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.vstack([g, np.flipud(g)]).tolist()''')
        
        elif pattern_name == 'slice_extract':
            # Extract non-background bounding box
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    coords = np.argwhere(mask)
    if len(coords) == 0: return g.tolist()
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0) + 1
    return g[r1:r2, c1:c2].tolist()''')
            
            # Extract unique colored region
            for color in in_colors:
                if color == bg: continue
                candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    mask = g == {color}
    coords = np.argwhere(mask)
    if len(coords) == 0: return g.tolist()
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0) + 1
    return g[r1:r2, c1:c2].tolist()''')
        
        elif pattern_name == 'iterate_until_stable' and same_size:
            # Flood fill enclosed regions
            candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g == {bg}
    labeled, n = ndimage.label(mask)
    out = g.copy()
    # Find regions that don't touch the border
    border_labels = set()
    h, w = g.shape
    for i in range(h):
        if mask[i,0]: border_labels.add(labeled[i,0])
        if mask[i,w-1]: border_labels.add(labeled[i,w-1])
    for j in range(w):
        if mask[0,j]: border_labels.add(labeled[0,j])
        if mask[h-1,j]: border_labels.add(labeled[h-1,j])
    for lbl in range(1, n+1):
        if lbl not in border_labels:
            # Find what color surrounds this region
            region_mask = labeled == lbl
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(region_mask)
            border_mask = dilated & ~region_mask & ~mask
            if border_mask.any():
                
                from collections import Counter
                border_colors = Counter(g[border_mask].tolist())
                fill_color = border_colors.most_common(1)[0][0]
                out[region_mask] = fill_color
    return out.tolist()''')
            
            # Gravity: drop non-bg cells down
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, {bg})
    for j in range(w):
        col = [g[i,j] for i in range(h) if g[i,j] != {bg}]
        for idx, val in enumerate(col):
            out[h - len(col) + idx, j] = val
    return out.tolist()''')
        
        elif pattern_name == 'construct_grid':
            if not same_size:
                # Count objects → output size
                candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    # Count as output
    
    from collections import Counter
    colors = Counter()
    for obj_id in range(1, n+1):
        obj_mask = labeled == obj_id
        vals = g[obj_mask]
        c = Counter(vals.tolist()).most_common(1)[0][0]
        colors[c] += 1
    out = np.zeros(({h_out}, {w_out}), dtype=int)
    # Fill based on counts
    for (color, count) in colors.most_common():
        for k in range(count):
            r, c = divmod(k, {w_out})
            if r < {h_out}:
                out[r, c] = color
    return out.tolist()''')
        
        elif pattern_name == 'nested_comprehension' and same_size:
            # XOR-like: combine with flip
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    flipped = np.fliplr(g)
    out = np.where((g != {bg}) | (flipped != {bg}), np.maximum(g, flipped), {bg})
    return out.tolist()''')
            
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    flipped = np.flipud(g)
    out = np.where((g != {bg}) | (flipped != {bg}), np.maximum(g, flipped), {bg})
    return out.tolist()''')
        
        elif pattern_name == 'sort_reorder':
            # Sort rows by non-bg count
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    rows = list(g)
    rows.sort(key=lambda r: sum(1 for x in r if x != {bg}))
    return np.array(rows).tolist()''')
            # Sort columns
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    cols = [g[:, j] for j in range(g.shape[1])]
    cols.sort(key=lambda c: sum(1 for x in c if x != {bg}))
    return np.column_stack(cols).tolist()''')
        
        elif pattern_name == 'copy_modify' and same_size:
            # Mirror/complete symmetry
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    # Complete horizontal symmetry
    for i in range(h):
        for j in range(w):
            mirror_j = w - 1 - j
            if out[i,j] == {bg} and out[i,mirror_j] != {bg}:
                out[i,j] = out[i,mirror_j]
            elif out[i,mirror_j] == {bg} and out[i,j] != {bg}:
                out[i,mirror_j] = out[i,j]
    return out.tolist()''')
            
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    # Complete vertical symmetry
    for i in range(h):
        for j in range(w):
            mirror_i = h - 1 - i
            if out[i,j] == {bg} and out[mirror_i,j] != {bg}:
                out[i,j] = out[mirror_i,j]
            elif out[mirror_i,j] == {bg} and out[i,j] != {bg}:
                out[mirror_i,j] = out[i,j]
    return out.tolist()''')
        
        elif pattern_name == 'incremental_build':
            if not same_size:
                # Build output row by row from input analysis
                candidates.append(f'''def solve(grid):
    import numpy as np
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = []
    for i in range(h):
        row = g[i].tolist()
        non_bg = [x for x in row if x != {bg}]
        if non_bg:
            out.append(non_bg)
    if not out: return g.tolist()
    return out''')
        
        elif pattern_name == 'parallel_iterate' and same_size:
            # Overlay two rotations/flips
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    r = np.rot90(g, 2)
    out = np.where(g != {bg}, g, r)
    return out.tolist()''')
        
        elif pattern_name == 'flood_fill_search' and same_size:
            # Draw lines between same-colored pixels
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for color in set(g.flatten().tolist()):
        if color == {bg}: continue
        coords = list(zip(*np.where(g == color)))
        if len(coords) == 2:
            r1, c1 = coords[0]
            r2, c2 = coords[1]
            if r1 == r2:  # same row, draw horizontal
                for j in range(min(c1,c2), max(c1,c2)+1):
                    out[r1,j] = color
            elif c1 == c2:  # same col, draw vertical
                for i in range(min(r1,r2), max(r1,r2)+1):
                    out[i,c1] = color
    return out.tolist()''')
        
        return candidates


# ─── PART 5: INTEGRATION ────────────────────────────────────────


def _composition_grammar():
    """Composition grammar patterns from atomic primitives.
    
    These are not hardcoded solutions -- they are composition TEMPLATES
    that Mother adapts by selecting different primitives at each slot.
    """
    patterns = []
    
    def P(name, skeleton, ops):
        patterns.append(GrammarPattern(
            name=name,
            skeleton=skeleton,
            frequency=0,
            example_tids=[],
            ops_used=set(ops),
            signature_features={},
        ))
    
    # Perceive → Extract → Transform → Construct
    P('perceive_extract_transform',
      'objs=find_objects(g,bg); r=transform(extract(g,objs[i],bg))',
      ['find_objects', 'extract_object'])
    
    # Split → Process each → Reassemble
    P('split_process_reassemble',
      'parts=split(g); proc=[transform(p) for p in parts]; r=stack(proc)',
      ['split_grid_h', 'split_grid_v', 'stack_h', 'stack_v'])
    
    # Detect color mapping → Apply
    P('detect_apply_color_map',
      'mapping=detect_color_mapping(inp,out); r=recolor(test,mapping)',
      ['detect_color_mapping', 'recolor'])
    
    # Find unique pattern → Use as template
    P('unique_as_template',
      'tmpl=extract_unique_pattern(g); r=apply_template(g,tmpl)',
      ['extract_unique_pattern'])
    
    # Detect geometry → Apply transform
    P('geometric_transform',
      'name=detect_transform(inp,out); r=apply(test)',
      ['detect_transform', 'rotate_90', 'flip_h', 'flip_v', 'transpose'])
    
    # Fill holes → Clean
    P('fill_and_clean',
      'filled=fill_interior(g); r=clean(filled)',
      ['fill_interior', 'get_outline', 'erode', 'dilate'])
    
    # Gravity settle
    P('gravity_settle',
      'r=gravity_direction(g,bg)',
      ['gravity_down', 'gravity_up', 'gravity_left', 'gravity_right'])
    
    # Scale/tile detection
    P('scale_or_tile',
      'f=detect_scale_or_tile(inp,out); r=scale_or_tile(test,f)',
      ['detect_scale_factor', 'detect_tile_pattern', 'scale_up', 'tile_grid'])
    
    # Color mask → Overlay
    P('color_mask_overlay',
      'mask=keep_only(g,c); r=overlay(base,mask)',
      ['keep_only', 'remove_color', 'overlay', 'extract_by_color'])
    
    # Find objects → Sort/filter → Reconstruct
    P('object_sort_filter',
      'objs=find_objects(g); filt=filter(objs); r=reconstruct(filt)',
      ['find_objects', 'extract_object', 'place_object', 'empty_grid'])
    
    # Crop → Transform → Embed back
    P('crop_transform_embed',
      'crop=crop_to_content(g); t=transform(crop); r=place_back(t)',
      ['crop_to_content', 'place_object'])
    
    # Mirror/symmetry completion
    P('symmetry_completion',
      'r=mirror_direction(g)',
      ['mirror_h', 'mirror_v', 'mirror_both'])
    
    # Draw connections between objects
    P('connect_objects',
      'objs=find_objects(g); r=draw_connections(g,objs)',
      ['find_objects', 'draw_line', 'flood_fill'])
    
    # Stamp pattern at positions
    P('stamp_at_positions',
      'pat=extract_pattern(g); pos=find_positions(g); r=stamp_pattern(canvas,pat,pos)',
      ['extract_smallest', 'find_objects', 'stamp_pattern', 'empty_grid'])
    
    return patterns



def _generate_primitive_candidates(task, inp0, out0, bg, color_map, features):
    """Generate candidate code strings using atomic primitives.
    
    Each candidate is a complete `def solve(grid):` function that Mother
    can adapt. These are TEMPLATES she fills with task-specific parameters.
    Not hardcoded solutions — geometric exploration.
    """
    import numpy as np
    from collections import Counter
    
    candidates = []
    train = task['train']
    h_in, w_in = inp0.shape
    h_out, w_out = out0.shape
    same_size = (h_in == h_out and w_in == w_out)
    
    # ── Detection-based: detect transform and apply ──
    from mother_primitives import (
        detect_transform, detect_color_mapping, detect_scale_factor, detect_tile_pattern,
        find_objects, extract_object, crop_to_content, background_color,
        rotate_90, rotate_180, rotate_270, flip_h, flip_v,
        scale_up, tile_grid, mirror_h, mirror_v, mirror_both,
        gravity_down, gravity_up, gravity_left, gravity_right,
        fill_interior, get_outline, dilate, erode, recolor,
    )
    
    # 1. Single-transform detection
    xform = detect_transform(inp0, out0)
    if xform and xform != 'identity':
        xform_map = {
            'rot90': 'np.rot90(g, -1)', 'rot180': 'np.rot90(g, -2)', 'rot270': 'np.rot90(g, -3)',
            'flip_h': 'np.fliplr(g)', 'flip_v': 'np.flipud(g)', 'transpose': 'g.T',
            'crop': 'crop_to_content(g)',
        }
        if xform in xform_map:
            candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return {xform_map[xform]}.tolist()""")
    
    # 2. Color mapping detection
    cmap = detect_color_mapping(inp0, out0)
    if cmap:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    out = g.copy()
    for old, new in {repr(cmap)}.items():
        out[g == old] = new
    return out.tolist()""")
    
    # 3. Scale factor detection
    sf = detect_scale_factor(inp0, out0)
    if sf:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.repeat(np.repeat(g, {sf}, axis=0), {sf}, axis=1).tolist()""")
    
    # 4. Tile detection
    tp = detect_tile_pattern(inp0, out0)
    if tp:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.tile(g, {repr(tp)}).tolist()""")
    
    # 5. Gravity in each direction
    if same_size:
        for direction, code in [
            ('down', """
    h, w = g.shape
    out = np.full_like(g, bg)
    for c in range(w):
        vals = [g[r, c] for r in range(h) if g[r, c] != bg]
        for i, v in enumerate(reversed(vals)):
            out[h-1-i, c] = v"""),
            ('up', """
    h, w = g.shape
    out = np.full_like(g, bg)
    for c in range(w):
        vals = [g[r, c] for r in range(h) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[i, c] = v"""),
            ('left', """
    h, w = g.shape
    out = np.full_like(g, bg)
    for r in range(h):
        vals = [g[r, c] for c in range(w) if g[r, c] != bg]
        for i, v in enumerate(vals):
            out[r, i] = v"""),
            ('right', """
    h, w = g.shape
    out = np.full_like(g, bg)
    for r in range(h):
        vals = [g[r, c] for c in range(w) if g[r, c] != bg]
        for i, v in enumerate(reversed(vals)):
            out[r, w-1-i] = v"""),
        ]:
            candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    bg = {bg}{code}
    return out.tolist()""")
    
    # 6. Mirror operations (output larger)
    if h_out == h_in and w_out == 2 * w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.concatenate([g, np.fliplr(g)], axis=1).tolist()""")
    if h_out == 2 * h_in and w_out == w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    return np.concatenate([g, np.flipud(g)], axis=0).tolist()""")
    if h_out == 2 * h_in and w_out == 2 * w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    top = np.concatenate([g, np.fliplr(g)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0).tolist()""")
    
    # 7. Crop to content
    if h_out <= h_in and w_out <= w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg)
    rows, cols = np.where(mask)
    if len(rows) == 0: return g.tolist()
    return g[rows.min():rows.max()+1, cols.min():cols.max()+1].tolist()""")
    
    # 8. Fill interior holes
    if same_size:
        candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg)
    filled = ndimage.binary_fill_holes(mask)
    out = g.copy()
    new_cells = filled & ~mask
    if new_cells.any():
        from scipy.ndimage import distance_transform_edt
        dist, idx = distance_transform_edt(~mask, return_indices=True)
        out[new_cells] = g[idx[0][new_cells], idx[1][new_cells]]
    return out.tolist()""")
    
    # 9. Get outline
    if same_size:
        candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    inner = ndimage.binary_erosion(mask, structure=kernel)
    out = g.copy()
    out[inner] = bg
    return out.tolist()""")
    
    # 10. Extract largest object
    if h_out < h_in or w_out < w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg).astype(int)
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    biggest = np.argmax(sizes) + 1
    obj_mask = (labeled == biggest)
    rows, cols = np.where(obj_mask)
    crop = g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()
    crop[~obj_mask[rows.min():rows.max()+1, cols.min():cols.max()+1]] = bg
    return crop.tolist()""")
    
    # 11. Extract smallest object
    if h_out < h_in or w_out < w_in:
        candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg).astype(int)
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    smallest = np.argmin(sizes) + 1
    obj_mask = (labeled == smallest)
    rows, cols = np.where(obj_mask)
    crop = g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()
    crop[~obj_mask[rows.min():rows.max()+1, cols.min():cols.max()+1]] = bg
    return crop.tolist()""")
    
    # 12. Object-level transforms: extract each, transform, reassemble
    if same_size:
        for xf_name, xf_code in [
            ('rot90', 'np.rot90(obj_crop, -1)'),
            ('rot180', 'np.rot90(obj_crop, -2)'),
            ('flip_h', 'np.fliplr(obj_crop)'),
            ('flip_v', 'np.flipud(obj_crop)'),
        ]:
            candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg).astype(int)
    labeled, n = ndimage.label(mask)
    out = np.full_like(g, bg)
    for i in range(1, n+1):
        obj_mask = (labeled == i)
        rows, cols = np.where(obj_mask)
        r1, c1 = rows.min(), cols.min()
        r2, c2 = rows.max()+1, cols.max()+1
        obj_crop = g[r1:r2, c1:c2].copy()
        obj_crop[~obj_mask[r1:r2, c1:c2]] = bg
        transformed = {xf_code}
        th, tw = transformed.shape
        for dr in range(th):
            for dc in range(tw):
                if transformed[dr, dc] != bg:
                    tr, tc = r1+dr, c1+dc
                    if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                        out[tr, tc] = transformed[dr, dc]
    return out.tolist()""")
    
    # 13. Sort objects by size (keep only largest/smallest N)
    # 14. Dilate/erode operations
    if same_size:
        for morph, code in [
            ('dilate', """
    from scipy.ndimage import binary_dilation, distance_transform_edt
    mask = (g != bg)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    expanded = binary_dilation(mask, structure=kernel)
    out = g.copy()
    new_cells = expanded & ~mask
    if new_cells.any():
        dist, idx = distance_transform_edt(~mask, return_indices=True)
        out[new_cells] = g[idx[0][new_cells], idx[1][new_cells]]"""),
            ('erode', """
    from scipy.ndimage import binary_erosion
    mask = (g != bg)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    eroded = binary_erosion(mask, structure=kernel)
    out = g.copy()
    out[mask & ~eroded] = bg"""),
        ]:
            candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    bg = {bg}{code}
    return out.tolist()""")
    
    # 15. Color-per-object: assign unique color to each object
    if same_size:
        out_colors_unique = sorted(set(out0.flatten().tolist()) - {bg})
        if len(out_colors_unique) >= 2:
            candidates.append(f"""def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    bg = {bg}
    mask = (g != bg).astype(int)
    labeled, n = ndimage.label(mask)
    colors = {repr(out_colors_unique)}
    out = np.full_like(g, bg)
    for i in range(1, n+1):
        c = colors[(i-1) % len(colors)]
        out[labeled == i] = c
    return out.tolist()""")
    
    # 16. Split grid at dividers, process panels
    # Check for divider lines
    divider_rows = []
    for r in range(h_in):
        vals = np.unique(inp0[r])
        if len(vals) == 1 and vals[0] != bg:
            divider_rows.append(r)
    divider_cols = []
    for c in range(w_in):
        vals = np.unique(inp0[:, c])
        if len(vals) == 1 and vals[0] != bg:
            divider_cols.append(c)
    
    if divider_rows or divider_cols:
        # Try overlay of panels
        candidates.append(f"""def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    bg = {bg}
    h, w = g.shape
    # Find divider rows/cols
    div_r = [r for r in range(h) if len(np.unique(g[r])) == 1 and np.unique(g[r])[0] != bg]
    div_c = [c for c in range(w) if len(np.unique(g[:, c])) == 1 and np.unique(g[:, c])[0] != bg]
    
    if div_r:
        # Split into panels at horizontal dividers
        bounds = [0] + [r for r in div_r] + [h]
        panels = [g[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1) if bounds[i+1] - bounds[i] > 0]
        panels = [p for p in panels if p.shape[0] > 0 and not (len(np.unique(p)) == 1)]
        if len(panels) >= 2:
            # Overlay: non-bg wins
            base = panels[0].copy()
            for p in panels[1:]:
                if p.shape == base.shape:
                    mask = (p != bg)
                    base[mask] = p[mask]
            return base.tolist()
    
    if div_c:
        bounds = [0] + [c for c in div_c] + [w]
        panels = [g[:, bounds[i]:bounds[i+1]] for i in range(len(bounds)-1) if bounds[i+1] - bounds[i] > 0]
        panels = [p for p in panels if p.shape[1] > 0 and not (len(np.unique(p)) == 1)]
        if len(panels) >= 2:
            base = panels[0].copy()
            for p in panels[1:]:
                if p.shape == base.shape:
                    mask = (p != bg)
                    base[mask] = p[mask]
            return base.tolist()
    
    return g.tolist()""")
    
    return candidates



def load_complete_language(state_dir=None):
    """Load Mother's complete language system."""
    if state_dir is None:
        state_dir = Path('/home/joe/sparky/e8_arc_agent/state')
    
    # 1. Build self-reading dictionary
    words = build_dictionary()
    
    # 2. Load solved codes for grammar learning
    codes = {}
    task_sigs = {}
    for fp in state_dir.glob('*.json'):
        try:
            d = json.load(open(fp))
            sc = d.get('solved_codes', {})
            if isinstance(sc, dict):
                for tid, code in sc.items():
                    if code and 'def ' in str(code):
                        codes[tid] = code
        except:
            pass
    
    # 3. Load signatures if available
    data_dir = Path('/home/joe/sparky/arc_data/combined/training')
    try:
        from codebook_expansion import SignatureExtractor
        for fp in data_dir.glob('*.json'):
            tid = fp.stem
            if tid in codes:
                task = json.load(open(fp))
                sig = SignatureExtractor.extract(task)
                task_sigs[tid] = np.array(sig.vector, dtype=np.float32)
    except:
        pass
    
    # 4. Learn grammar from solved codes
    grammar = learn_grammar(codes, task_sigs)
    # Add atomic composition grammar
    comp_grammar = _composition_grammar()
    grammar.extend(comp_grammar)
    
    # 5. Build semantic bridges
    bridges = build_semantic_bridges(words, grammar, codes, task_sigs)
    
    # 6. Count word frequencies from solved codes
    for tid, code in codes.items():
        for m in re.finditer(r'np\.([a-z_]+)', code):
            wname = f'np.{m.group(1)}'
            if wname in {w.name for w in words}:
                for w in words:
                    if w.name == wname:
                        w.frequency += 1
                        break
    
    mother = MotherComplete(words, grammar, bridges, codes)
    
    # Stats
    cats = Counter(w.category for w in words)
    print(f"Mother Complete Language loaded:")
    print(f"  Dictionary: {len(words)} words ({len([w for w in words if w.module == 'np'])} numpy, "
          f"{len([w for w in words if w.module == 'ndimage'])} ndimage, "
          f"{len([w for w in words if w.module == 'builtin'])} builtin)")
    print(f"  Grammar: {len(grammar)} patterns (learned from {len(codes)} solved codes)")
    print(f"  Semantic bridges: {len(bridges)}")
    print(f"  Categories: {dict(cats.most_common())}")
    
    return mother


def vocabulary_stats(mother):
    """Print what Mother knows."""
    lines = [f"Mother Complete: {len(mother.words)} words, {len(mother.grammar)} grammar, {len(mother.bridges)} bridges"]
    
    cats = defaultdict(list)
    for w in mother.words.values():
        cats[w.category].append(w.name)
    
    for cat in sorted(cats):
        names = sorted(cats[cat])
        lines.append(f"  {cat} ({len(names)}): {', '.join(names[:8])}{'...' if len(names) > 8 else ''}")
    
    lines.append(f"\nGrammar patterns (top 10):")
    for gp in mother.grammar_ranked[:10]:
        lines.append(f"  {gp.name}: freq={gp.frequency}, ops={len(gp.ops_used)}")
    
    lines.append(f"\nSemantic bridges:")
    for b in mother.bridges:
        lines.append(f"  {b.feature}: {len(b.relevant_words)} words, {len(b.relevant_grammar)} patterns, weight={b.weight:.2f}")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    mother = load_complete_language()
    print()
    print(vocabulary_stats(mother))
# ─── PART 6: CORE KNOWLEDGE PRIORS ──────────────────────────────
# From Spelke & Kinzler (2007) "Core Knowledge" and Chollet (2019)
# "On the Measure of Intelligence" Section III.
#
# These are NOT documentation — they are OPERATIONAL PRIMITIVES
# that Mother uses to reason about ARC tasks. Each prior generates
# code candidates that express its reasoning pattern.
#
# The 4 Core Knowledge Systems (Spelke):
#   1. OBJECTNESS — cohesion, persistence, contact
#   2. AGENTNESS — goal-directedness, intentionality  
#   3. NUMBER — counting, elementary arithmetic
#   4. GEOMETRY — topology, symmetry, containment
#
# ARC-AGI adds (from Chollet 2019, Section III.5.1):
#   5. TOPOLOGY — inside/outside, connectivity, boundaries

def generate_core_knowledge_candidates(task, sig, bg=0):
    """Generate candidates from Core Knowledge reasoning primitives.
    
    Each function below implements one cognitive prior as code generation.
    These represent how a 4-year-old human reasons about visual puzzles.
    """
    import numpy as np
    
    from collections import Counter
    
    candidates = []
    train = task.get('train', [])
    if not train:
        return candidates
    
    inp0 = np.array(train[0]['input'], dtype=int)
    out0 = np.array(train[0]['output'], dtype=int)
    h_in, w_in = inp0.shape
    h_out, w_out = out0.shape
    same_size = (h_in == h_out and w_in == w_out)
    
    in_counts = Counter(inp0.flatten().tolist())
    bg = in_counts.most_common(1)[0][0] if in_counts else 0
    in_colors = set(inp0.flatten().tolist())
    out_colors = set(out0.flatten().tolist())
    
    # ═══════════════════════════════════════════════════════════
    # PRIOR 1: OBJECTNESS
    # Spelke: Objects are cohesive (connected), persistent (don't
    # vanish), and interact through contact.
    # ═══════════════════════════════════════════════════════════
    
    # 1a. Object cohesion: connected components are single objects
    #     → label, extract bounding box, operate per-object
    candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    objects = []
    for obj_id in range(1, n+1):
        coords = np.argwhere(labeled == obj_id)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        obj = g[r1:r2, c1:c2].copy()
        obj_mask = labeled[r1:r2, c1:c2] == obj_id
        objects.append((obj, obj_mask, r1, c1, r2, c2, int(obj_mask.sum())))
    objects.sort(key=lambda x: -x[6])
    if objects:
        obj, msk, r1, c1, r2, c2, sz = objects[0]
        return obj.tolist()
    return g.tolist()''')
    
    # 1b. Object persistence: objects behind occlusion still exist
    #     → if a color patch is partially hidden by another, reconstruct it
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    for obj_id in range(1, n+1):
        obj_mask = labeled == obj_id
        coords = np.argwhere(obj_mask)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        # If bounding box has bg holes, fill them with object color
        
        from collections import Counter
        obj_colors = Counter(g[obj_mask].tolist())
        main_color = obj_colors.most_common(1)[0][0]
        for i in range(r1, r2):
            for j in range(c1, c2):
                if out[i,j] == {bg}:
                    out[i,j] = main_color
    return out.tolist()''')
    
    # 1c. Contact principle: objects interact when touching
    #     → find adjacent objects, merge or transfer properties
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    # Find which objects are adjacent (touching)
    from scipy.ndimage import binary_dilation
    for obj_id in range(1, n+1):
        obj_mask = labeled == obj_id
        dilated = binary_dilation(obj_mask)
        neighbors = set()
        for other_id in range(1, n+1):
            if other_id != obj_id:
                if (dilated & (labeled == other_id)).any():
                    neighbors.add(other_id)
        # If touching another object, adopt its color
        if neighbors:
            
            from collections import Counter
            obj_colors = Counter(g[obj_mask].tolist())
            my_color = obj_colors.most_common(1)[0][0]
            for nid in neighbors:
                n_mask = labeled == nid
                n_colors = Counter(g[n_mask].tolist())
                n_color = n_colors.most_common(1)[0][0]
                if n_color != my_color:
                    out[obj_mask] = n_color
                    break
    return out.tolist()''')

    # ═══════════════════════════════════════════════════════════
    # PRIOR 2: AGENTNESS / GOAL-DIRECTEDNESS
    # Chollet: input→output can be modeled as start→end state 
    # of an intentional process. Objects "want" to reach goals.
    # ═══════════════════════════════════════════════════════════
    
    # 2a. Movement toward target: objects move to fill gaps or reach markers
    if same_size:
        for color in in_colors:
            if color == bg: continue
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, {bg})
    # Find objects of color {color} and move them toward center
    coords = list(zip(*np.where(g == {color})))
    if not coords: return g.tolist()
    cr = sum(r for r,c in coords) // len(coords)
    cc = sum(c for r,c in coords) // len(coords)
    # Copy all non-{color} cells as-is
    for i in range(h):
        for j in range(w):
            if g[i,j] != {color} and g[i,j] != {bg}:
                out[i,j] = g[i,j]
    # Place {color} cells shifted toward center of grid
    gh, gw = h//2, w//2
    dr = 1 if cr < gh else (-1 if cr > gh else 0)
    dc = 1 if cc < gw else (-1 if cc > gw else 0)
    for r, c in coords:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr,nc] = {color}
        else:
            out[r,c] = {color}
    return out.tolist()''')
    
    # 2b. Projection: objects extend lines in their direction
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for color in set(g.flatten().tolist()):
        if color == {bg}: continue
        coords = list(zip(*np.where(g == color)))
        if len(coords) < 2: continue
        # Find if points form a line and extend it
        rows = [r for r,c in coords]
        cols = [c for r,c in coords]
        if len(set(rows)) == 1:  # horizontal line
            r = rows[0]
            c_min, c_max = min(cols), max(cols)
            for j in range(c_min, c_max+1):
                out[r,j] = color
        elif len(set(cols)) == 1:  # vertical line
            c = cols[0]
            r_min, r_max = min(rows), max(rows)
            for i in range(r_min, r_max+1):
                out[i,c] = color
    return out.tolist()''')

    # ═══════════════════════════════════════════════════════════
    # PRIOR 3: NUMBER / COUNTING
    # Spelke: infants can count small quantities, compare sizes,
    # do elementary arithmetic (more/less/equal)
    # ═══════════════════════════════════════════════════════════
    
    # 3a. Count objects → output encodes count
    if not same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    # Count objects per color
    color_counts = Counter()
    for obj_id in range(1, n+1):
        vals = g[labeled == obj_id]
        c = Counter(vals.tolist()).most_common(1)[0][0]
        color_counts[c] += 1
    # Build output: one row per color, width = count
    rows = []
    for color, count in sorted(color_counts.items()):
        rows.append([color] * count)
    if not rows: return [[0]]
    max_w = max(len(r) for r in rows)
    out = []
    for r in rows:
        out.append(r + [{bg}] * (max_w - len(r)))
    return out''')
    
    # 3b. Majority rule: most common color/object wins
    candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    # Find most common object shape
    shapes = []
    for obj_id in range(1, n+1):
        coords = np.argwhere(labeled == obj_id)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        obj = g[r1:r2, c1:c2].copy()
        obj[labeled[r1:r2, c1:c2] != obj_id] = {bg}
        shapes.append((obj.tobytes(), obj, r1, c1))
    shape_counts = Counter(s[0] for s in shapes)
    most_common_shape = shape_counts.most_common(1)[0][0]
    for s_bytes, obj, r1, c1 in shapes:
        if s_bytes == most_common_shape:
            return obj.tolist()
    return g.tolist()''')
    
    # 3c. Size comparison: select object by relative size
    if not same_size:
        for selector in ['smallest', 'second']:
            candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    objs = []
    for obj_id in range(1, n+1):
        obj_mask = labeled == obj_id
        coords = np.argwhere(obj_mask)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        objs.append((int(obj_mask.sum()), g[r1:r2, c1:c2].copy(), r1, c1))
    objs.sort(key=lambda x: x[0])
    idx = 0 if '{selector}' == 'smallest' else min(1, len(objs)-1)
    return objs[idx][1].tolist()''')

    # ═══════════════════════════════════════════════════════════
    # PRIOR 4: GEOMETRY / TOPOLOGY
    # Spelke: understanding of spatial relationships, symmetry,
    # rotation, containment (inside/outside), connectivity
    # ═══════════════════════════════════════════════════════════
    
    # 4a. Symmetry completion: detect and complete partial symmetry
    if same_size:
        # 4-way symmetry completion
        candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    # Complete 4-fold symmetry
    for i in range(h):
        for j in range(w):
            mi, mj = h-1-i, w-1-j
            cells = [out[i,j], out[i,mj], out[mi,j], out[mi,mj]]
            non_bg = [c for c in cells if c != {bg}]
            if non_bg:
                fill = max(set(non_bg), key=non_bg.count)
                if out[i,j] == {bg}: out[i,j] = fill
                if out[i,mj] == {bg}: out[i,mj] = fill
                if out[mi,j] == {bg}: out[mi,j] = fill
                if out[mi,mj] == {bg}: out[mi,mj] = fill
    return out.tolist()''')
    
    # 4b. Inside/outside (containment): fill enclosed regions
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    # Find non-bg colors that form boundaries
    for color in set(g.flatten().tolist()):
        if color == {bg}: continue
        boundary = g == color
        # Label bg regions
        bg_mask = g == {bg}
        labeled, n = ndimage.label(bg_mask)
        # Find interior regions (don't touch border)
        border_labels = set()
        for i in range(h):
            if bg_mask[i,0]: border_labels.add(labeled[i,0])
            if bg_mask[i,w-1]: border_labels.add(labeled[i,w-1])
        for j in range(w):
            if bg_mask[0,j]: border_labels.add(labeled[0,j])
            if bg_mask[h-1,j]: border_labels.add(labeled[h-1,j])
        for lbl in range(1, n+1):
            if lbl not in border_labels:
                out[labeled == lbl] = color
        break  # try first non-bg color as boundary
    return out.tolist()''')
    
    # 4c. Connectivity: draw lines between same-colored points
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for color in set(g.flatten().tolist()):
        if color == {bg}: continue
        points = list(zip(*np.where(g == color)))
        if len(points) == 2:
            r1, c1 = points[0]
            r2, c2 = points[1]
            # Draw line: Bresenham-like
            dr = 0 if r1==r2 else (1 if r2>r1 else -1)
            dc = 0 if c1==c2 else (1 if c2>c1 else -1)
            r, c = r1, c1
            while (r, c) != (r2, c2):
                out[r, c] = color
                if r != r2: r += dr
                if c != c2: c += dc
            out[r2, c2] = color
    return out.tolist()''')
    
    # 4d. Scaling: output is input scaled by integer factor
    if not same_size and h_out > h_in and w_out > w_in:
        rh = h_out / h_in
        rw = w_out / w_in
        if abs(rh - round(rh)) < 0.01 and abs(rw - round(rw)) < 0.01:
            sh, sw = int(round(rh)), int(round(rw))
            candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h*{sh}, w*{sw}), dtype=int)
    for i in range(h):
        for j in range(w):
            out[i*{sh}:(i+1)*{sh}, j*{sw}:(j+1)*{sw}] = g[i,j]
    return out.tolist()''')
    
    # 4e. Downscaling: output is input shrunk by integer factor
    if not same_size and h_out < h_in and w_out < w_in:
        if h_in % h_out == 0 and w_in % w_out == 0:
            sh = h_in // h_out
            sw = w_in // w_out
            candidates.append(f'''def solve(grid):
    import numpy as np
    
    from collections import Counter
    g = np.array(grid, dtype=int)
    h, w = g.shape
    oh, ow = {h_out}, {w_out}
    out = np.zeros((oh, ow), dtype=int)
    for i in range(oh):
        for j in range(ow):
            block = g[i*{sh}:(i+1)*{sh}, j*{sw}:(j+1)*{sw}]
            counts = Counter(block.flatten().tolist())
            # Most common non-bg, or bg
            non_bg = [(c, n) for c, n in counts.items() if c != {bg}]
            if non_bg:
                out[i,j] = max(non_bg, key=lambda x: x[1])[0]
            else:
                out[i,j] = {bg}
    return out.tolist()''')
    
    # 4f. Pattern replication within grid: each cell expands to a sub-pattern
    if not same_size and h_out > h_in:
        # Check if output is input⊗pattern (Kronecker-like)
        candidates.append(f'''def solve(grid):
    import numpy as np
    g = np.array(grid, dtype=int)
    h, w = g.shape
    # Each non-bg cell becomes a copy of the pattern defined by non-bg cells
    mask = g != {bg}
    if not mask.any(): return g.tolist()
    coords = np.argwhere(mask)
    r1, c1 = coords.min(axis=0)
    r2, c2 = coords.max(axis=0) + 1
    pattern = g[r1:r2, c1:c2]
    ph, pw = pattern.shape
    out = np.full((h*ph, w*pw), {bg}, dtype=int)
    for i in range(h):
        for j in range(w):
            if g[i,j] != {bg}:
                out[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = pattern
    return out.tolist()''')
    
    # 4g. Rotation to match: try all rotations of objects
    if not same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    mask = g != {bg}
    labeled, n = ndimage.label(mask)
    if n == 0: return g.tolist()
    # Extract each object, try rotation
    objs = []
    for obj_id in range(1, n+1):
        coords = np.argwhere(labeled == obj_id)
        r1, c1 = coords.min(axis=0)
        r2, c2 = coords.max(axis=0) + 1
        obj = g[r1:r2, c1:c2].copy()
        obj[labeled[r1:r2, c1:c2] != obj_id] = {bg}
        objs.append(obj)
    if len(objs) >= 2:
        # Return overlay of first two objects
        a, b = objs[0], objs[1]
        # Resize b to match a if needed
        if a.shape != b.shape:
            b = np.rot90(b)
        if a.shape == b.shape:
            out = np.where(a != {bg}, a, b)
            return out.tolist()
    if objs:
        return objs[0].tolist()
    return g.tolist()''')

    # ═══════════════════════════════════════════════════════════
    # PRIOR 5: TOPOLOGY (ARC-specific extension)
    # Boundaries, borders, edge detection, region adjacency
    # ═══════════════════════════════════════════════════════════
    
    # 5a. Border/edge coloring
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    mask = g != {bg}
    # Find edges: pixels adjacent to bg
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(mask)
    border = mask & ~interior
    
    from collections import Counter
    border_colors = Counter(g[border].tolist())
    # Color interior differently from border
    if border_colors:
        border_color = border_colors.most_common(1)[0][0]
        interior_colors = Counter(g[interior].tolist()) if interior.any() else Counter()
        if interior_colors:
            int_color = interior_colors.most_common(1)[0][0]
            out[border] = int_color
            out[interior] = border_color
    return out.tolist()''')
    
    # 5b. Boundary tracing: outline objects
    if same_size:
        candidates.append(f'''def solve(grid):
    import numpy as np
    from scipy import ndimage
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, {bg})
    mask = g != {bg}
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(mask)
    border = mask & ~interior
    out[border] = g[border]
    return out.tolist()''')
    
    return candidates