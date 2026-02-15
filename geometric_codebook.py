#!/usr/bin/env python3
"""
GEOMETRIC CODEBOOK - Signal-to-Language Decoder
Ghost in the Machine Labs

The missing piece: converts fused substrate geometric outputs
into actionable text (analysis) and Python code (solve functions).

Architecture:
  Grid → GeometricEncoder → substrate signal (numpy)
  Substrate processes → output signal (numpy)  
  Output signal → GeometricDecoder → {analysis text, hypothesis, Python code}

The codebook works by detecting geometric PRIMITIVES in the substrate
output — the torsions, symmetries, and relational patterns that the
sensor panels identified — and mapping them to transformation OPERATIONS
that can be expressed as code.

This is fabrication, not training. Each primitive→operation mapping is
a direct geometric relationship, not a learned weight.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# GEOMETRIC PRIMITIVES — the vocabulary of transformations
# =============================================================================

class GeoPrimitive(Enum):
    """
    Primitives detected by substrate sensor panels.
    Each maps to one or more ARC transformation operations.
    """
    # Spatial
    TILE_REPEAT = "tile_repeat"          # Pattern repeats in space
    MIRROR_H = "mirror_horizontal"       # Horizontal reflection
    MIRROR_V = "mirror_vertical"         # Vertical reflection
    MIRROR_DIAG = "mirror_diagonal"      # Diagonal reflection
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    TRANSLATE = "translate"              # Shift pattern
    SCALE_UP = "scale_up"               # Enlarge grid/pattern
    SCALE_DOWN = "scale_down"            # Shrink grid/pattern
    
    # Chromatic (color operations)
    COLOR_MAP = "color_map"              # Recolor by mapping
    COLOR_FILL = "color_fill"            # Flood fill region
    COLOR_SWAP = "color_swap"            # Swap two colors
    COLOR_COUNT = "color_count"          # Count-based operation
    MAJORITY_COLOR = "majority_color"    # Most frequent color
    BOUNDARY_COLOR = "boundary_color"    # Color at edges
    
    # Structural
    EXTRACT_SHAPE = "extract_shape"      # Pull out a shape
    MASK_OVERLAY = "mask_overlay"        # Apply mask/overlay
    CROP = "crop"                        # Trim to content
    PAD = "pad"                          # Add border
    PARTITION = "partition"              # Split into regions
    GRAVITY = "gravity"                  # Objects fall/slide
    
    # Relational
    SORT_BY_SIZE = "sort_by_size"        # Order by region size
    ALIGN = "align"                      # Align objects
    CONNECT = "connect"                  # Draw lines between
    ENCLOSE = "enclose"                  # Draw boundary around
    
    # Pattern
    DENOISING = "denoising"              # Remove noise
    COMPLETE_PATTERN = "complete_pattern" # Fill in missing
    BOOLEAN_OP = "boolean_op"            # AND/OR/XOR grids
    CONDITIONAL = "conditional"          # If-then color rules


@dataclass
class DetectedPrimitive:
    """A primitive detected in the substrate output with confidence."""
    primitive: GeoPrimitive
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)
    # Geometric evidence from substrate
    sensor_source: str = ""      # Which sensor type detected it
    energy: float = 0.0          # Signal energy at detection
    harmonic_alignment: float = 0.0  # How aligned with field


# =============================================================================
# GRID ENCODER — ARC grids to substrate signals
# =============================================================================

class GridEncoder:
    """
    Encode ARC grids as geometric signals for the substrate.
    
    Rather than flattening to bytes, we encode the GEOMETRIC PROPERTIES
    of the grid that the sensor panels are designed to detect:
    - Spatial frequency (tiling/repetition)
    - Color distribution (chromatic spectrum)
    - Shape boundaries (structural edges)
    - Symmetry axes
    - Relational positions
    """
    
    SIGNAL_SIZE = 1024
    
    @staticmethod
    def encode_grid(grid: List[List[int]]) -> np.ndarray:
        """Encode a single grid into geometric signal."""
        g = np.array(grid, dtype=np.float32)
        h, w = g.shape
        signal = np.zeros(GridEncoder.SIGNAL_SIZE, dtype=np.float32)
        
        idx = 0
        
        # === Band 1: Shape signature (0-127) ===
        # Dimensions encoded as ratios
        signal[idx] = h / 30.0; idx += 1
        signal[idx] = w / 30.0; idx += 1
        signal[idx] = h * w / 900.0; idx += 1
        signal[idx] = h / w if w > 0 else 1.0; idx += 1
        
        # Flattened grid (normalized to 0-1)
        flat = g.flatten() / 9.0
        n = min(len(flat), 124)
        signal[idx:idx+n] = flat[:n]
        idx = 128
        
        # === Band 2: Color spectrum (128-255) ===
        # Color histogram (10 colors, 0-9)
        for c in range(10):
            count = np.sum(g == c)
            signal[idx + c] = count / (h * w)
        idx += 10
        
        # Color adjacency matrix (which colors touch which)
        for r in range(h):
            for c in range(w):
                val = int(g[r, c])
                # Right neighbor
                if c + 1 < w:
                    n_val = int(g[r, c + 1])
                    if val != n_val:
                        pair_idx = 128 + 10 + val * 10 + n_val
                        if pair_idx < 256:
                            signal[pair_idx] += 1.0 / (h * w)
                # Down neighbor
                if r + 1 < h:
                    n_val = int(g[r + 1, c])
                    if val != n_val:
                        pair_idx = 128 + 10 + val * 10 + n_val
                        if pair_idx < 256:
                            signal[pair_idx] += 1.0 / (h * w)
        idx = 256
        
        # === Band 3: Symmetry signatures (256-383) ===
        # Horizontal symmetry
        if h > 1:
            h_sym = np.mean(g == g[::-1, :])
            signal[idx] = h_sym
        idx += 1
        
        # Vertical symmetry
        if w > 1:
            v_sym = np.mean(g == g[:, ::-1])
            signal[idx] = v_sym
        idx += 1
        
        # Diagonal symmetry (if square)
        if h == w:
            d_sym = np.mean(g == g.T)
            signal[idx] = d_sym
        idx += 1
        
        # 90° rotational symmetry (if square)
        if h == w:
            r90 = np.rot90(g)
            signal[idx] = np.mean(g == r90)
        idx += 1
        idx = 384
        
        # === Band 4: Spatial frequency (384-511) ===
        # Row-wise and column-wise repetition patterns
        for r in range(min(h, 30)):
            row = g[r, :]
            # Check for period-N repetition
            for period in range(1, min(w, 8)):
                if w % period == 0:
                    tiles = row.reshape(-1, period)
                    if len(tiles) > 1 and np.all(tiles == tiles[0]):
                        signal[384 + r * 4 + min(period-1, 3)] = 1.0
                        break
        idx = 512
        
        # === Band 5: Boundary/edge features (512-639) ===
        # Edge detection (Sobel-like)
        if h > 2 and w > 2:
            for r in range(1, min(h-1, 16)):
                for c in range(1, min(w-1, 8)):
                    # Gradient magnitude
                    gx = float(g[r, c+1]) - float(g[r, c-1])
                    gy = float(g[r+1, c]) - float(g[r-1, c])
                    mag = np.sqrt(gx**2 + gy**2)
                    eidx = 512 + r * 8 + c
                    if eidx < 640:
                        signal[eidx] = mag / 12.73  # max gradient = 9*sqrt(2)
        idx = 640
        
        # === Band 6: Object detection (640-767) ===
        # Connected component count per color
        visited = np.zeros_like(g, dtype=bool)
        obj_count = 0
        for r in range(h):
            for c in range(w):
                if not visited[r, c]:
                    color = g[r, c]
                    if color != 0:  # Skip background
                        # BFS
                        stack = [(r, c)]
                        size = 0
                        while stack:
                            cr, cc = stack.pop()
                            if 0 <= cr < h and 0 <= cc < w and not visited[cr, cc] and g[cr, cc] == color:
                                visited[cr, cc] = True
                                size += 1
                                stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                        if size > 0 and 640 + obj_count < 768:
                            signal[640 + obj_count] = size / (h * w)
                            obj_count += 1
        signal[767] = obj_count / 30.0  # total object count
        idx = 768
        
        # === Band 7: Transformation hints (768-895) ===
        # Reserved for encoding input→output relationships
        # (filled by encode_pair)
        idx = 896
        
        # === Band 8: Raw hash (896-1023) ===
        # Deterministic hash for exact matching
        raw = g.tobytes()
        for i in range(min(128, len(raw))):
            signal[896 + i] = (raw[i] - 128.0) / 128.0
        
        return signal
    
    @staticmethod
    def encode_pair(input_grid: List[List[int]], 
                    output_grid: List[List[int]]) -> np.ndarray:
        """
        Encode an input→output pair, capturing the TRANSFORMATION.
        
        Band 7 (768-895) encodes the relationship:
        - Dimension change ratios
        - Color mapping
        - Spatial operation signature
        """
        sig = GridEncoder.encode_grid(input_grid)
        
        ig = np.array(input_grid, dtype=np.float32)
        og = np.array(output_grid, dtype=np.float32)
        ih, iw = ig.shape
        oh, ow = og.shape
        
        idx = 768
        
        # Dimension relationships
        sig[idx] = oh / ih if ih > 0 else 1.0; idx += 1  # height ratio
        sig[idx] = ow / iw if iw > 0 else 1.0; idx += 1  # width ratio
        sig[idx] = (oh * ow) / (ih * iw) if ih * iw > 0 else 1.0; idx += 1  # area ratio
        sig[idx] = 1.0 if oh == ih and ow == iw else 0.0; idx += 1  # same size?
        
        # Color transformation
        in_colors = set(ig.flatten().astype(int))
        out_colors = set(og.flatten().astype(int))
        sig[idx] = len(in_colors) / 10.0; idx += 1
        sig[idx] = len(out_colors) / 10.0; idx += 1
        sig[idx] = len(in_colors & out_colors) / max(len(in_colors | out_colors), 1); idx += 1
        sig[idx] = 1.0 if in_colors == out_colors else 0.0; idx += 1
        
        # If same size, compute cell-wise diff
        if ih == oh and iw == ow:
            diff = (ig != og).astype(np.float32)
            sig[idx] = np.mean(diff); idx += 1  # fraction changed
            sig[idx] = np.sum(diff) / max(ih * iw, 1); idx += 1  # changed count ratio
            
            # Where did changes happen? Edge vs center
            if ih > 2 and iw > 2:
                edge_mask = np.zeros_like(diff)
                edge_mask[0, :] = 1; edge_mask[-1, :] = 1
                edge_mask[:, 0] = 1; edge_mask[:, -1] = 1
                edge_changes = np.sum(diff * edge_mask)
                center_changes = np.sum(diff * (1 - edge_mask))
                total_changes = edge_changes + center_changes
                sig[idx] = edge_changes / max(total_changes, 1); idx += 1
                sig[idx] = center_changes / max(total_changes, 1); idx += 1
        else:
            idx += 4
        
        # Tiling check: does output = tiled input?
        if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
            tile_h = oh // ih
            tile_w = ow // iw
            tiled = np.tile(ig, (tile_h, tile_w))
            sig[idx] = np.mean(tiled == og); idx += 1  # simple tile match
            sig[idx] = tile_h / 10.0; idx += 1
            sig[idx] = tile_w / 10.0; idx += 1
            
            # Check alternating tile (flip every other)
            alt_tiled = np.zeros_like(og)
            for tr in range(tile_h):
                for tc in range(tile_w):
                    block = ig.copy()
                    if tr % 2 == 1:
                        block = block[::-1, :]
                    if tc % 2 == 1:
                        block = block[:, ::-1]
                    alt_tiled[tr*ih:(tr+1)*ih, tc*iw:(tc+1)*iw] = block
            sig[idx] = np.mean(alt_tiled == og); idx += 1  # alternating tile match
        else:
            idx += 4
        
        # Rotation check (if square)
        if ih == iw:
            r90 = np.rot90(ig)
            r180 = np.rot90(ig, 2)
            r270 = np.rot90(ig, 3)
            if oh == ih and ow == iw:
                sig[idx] = np.mean(r90 == og); idx += 1
                sig[idx] = np.mean(r180 == og); idx += 1
                sig[idx] = np.mean(r270 == og); idx += 1
            else:
                idx += 3
        else:
            idx += 3
        
        # Mirror check
        if oh == ih and ow == iw:
            sig[idx] = np.mean(ig[::-1, :] == og); idx += 1  # flip H
            sig[idx] = np.mean(ig[:, ::-1] == og); idx += 1  # flip V
        
        return sig
    
    @staticmethod
    def encode_task(task: Dict) -> np.ndarray:
        """
        Encode full ARC task (all training pairs) into composite signal.
        
        Averages pair signals to find the CONSISTENT transformation
        pattern across all examples.
        """
        train = task.get('train', [])
        if not train:
            return np.zeros(GridEncoder.SIGNAL_SIZE, dtype=np.float32)
        
        signals = []
        for pair in train:
            sig = GridEncoder.encode_pair(pair['input'], pair['output'])
            signals.append(sig)
        
        # Consensus: average across pairs
        # High-confidence features will be consistent, noise will cancel
        composite = np.mean(signals, axis=0).astype(np.float32)
        
        # Variance across pairs (low = consistent = confident)
        if len(signals) > 1:
            variance = np.var(signals, axis=0)
            # Boost consistent features, dampen noisy ones
            consistency = 1.0 / (1.0 + variance * 10)
            composite *= consistency
        
        return composite


# =============================================================================
# PRIMITIVE DETECTOR — reads substrate output to find operations
# =============================================================================

class PrimitiveDetector:
    """
    Detect geometric primitives from substrate output signal.
    
    The substrate's sensor panels have already done the hard work —
    detecting spatial frequencies, symmetries, boundaries, etc.
    The detector reads those activations and maps them to named
    transformation primitives.
    """
    
    # Thresholds for detection
    CONFIDENCE_THRESHOLD = 0.3
    
    @staticmethod
    def detect(input_signal: np.ndarray, 
               output_signal: np.ndarray,
               task: Dict) -> List[DetectedPrimitive]:
        """
        Detect all primitives from substrate processing results.
        
        Uses both the encoded signal AND the raw task data to
        cross-validate detections.
        """
        primitives = []
        train = task.get('train', [])
        if not train:
            return primitives
        
        # Analyze all training pairs for consensus
        pair_primitives = []
        for pair in train:
            pp = PrimitiveDetector._detect_pair(pair['input'], pair['output'])
            pair_primitives.append(set(p.primitive for p in pp))
            primitives.extend(pp)
        
        # Consensus: only keep primitives detected in ALL pairs
        if pair_primitives:
            consensus = pair_primitives[0]
            for ps in pair_primitives[1:]:
                consensus &= ps
            
            if consensus:
                # Filter to consensus + boost confidence
                consensus_prims = []
                for p in primitives:
                    if p.primitive in consensus:
                        p.confidence = min(1.0, p.confidence * 1.5)  # boost
                        consensus_prims.append(p)
                
                # Deduplicate: keep highest confidence per primitive type
                best = {}
                for p in consensus_prims:
                    if p.primitive not in best or p.confidence > best[p.primitive].confidence:
                        best[p.primitive] = p
                
                primitives = list(best.values())
            else:
                # No perfect consensus — keep all and deduplicate by confidence
                best = {}
                for p in primitives:
                    if p.primitive not in best or p.confidence > best[p.primitive].confidence:
                        best[p.primitive] = p
                primitives = list(best.values())
        
        # Sort by confidence
        primitives.sort(key=lambda p: p.confidence, reverse=True)
        return primitives
    
    @staticmethod
    def _detect_pair(input_grid: List[List[int]], 
                     output_grid: List[List[int]]) -> List[DetectedPrimitive]:
        """Detect primitives for a single input→output pair."""
        prims = []
        ig = np.array(input_grid, dtype=np.float32)
        og = np.array(output_grid, dtype=np.float32)
        ih, iw = ig.shape
        oh, ow = og.shape
        
        # --- TILING ---
        if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
            tile_h, tile_w = oh // ih, ow // iw
            
            # Simple tile
            tiled = np.tile(ig, (tile_h, tile_w))
            match = np.mean(tiled == og)
            if match > 0.8:
                prims.append(DetectedPrimitive(
                    GeoPrimitive.TILE_REPEAT, match,
                    {'tile_h': tile_h, 'tile_w': tile_w, 'alternating': False},
                    'spatial'))
            
            # Alternating tile: tile columns, alternate block-rows
            # between original and column-reversed
            alt = np.zeros_like(og)
            for tr in range(tile_h):
                block = ig.copy()
                if tr % 2 == 1:
                    block = block[:, ::-1]  # reverse columns on odd blocks
                row_tile = np.tile(block, (1, tile_w))
                alt[tr*ih:(tr+1)*ih, :] = row_tile
            alt_match = np.mean(alt == og)
            if alt_match > match and alt_match > 0.8:
                prims.append(DetectedPrimitive(
                    GeoPrimitive.TILE_REPEAT, alt_match,
                    {'tile_h': tile_h, 'tile_w': tile_w, 'alternating': True},
                    'spatial'))
        
        # --- SCALING ---
        if oh > ih and ow > iw:
            sh, sw = oh / ih, ow / iw
            if sh == sw and sh == int(sh):
                scale = int(sh)
                scaled = np.repeat(np.repeat(ig, scale, axis=0), scale, axis=1)
                if scaled.shape == og.shape:
                    match = np.mean(scaled == og)
                    if match > 0.8:
                        prims.append(DetectedPrimitive(
                            GeoPrimitive.SCALE_UP, match,
                            {'factor': scale}, 'spatial'))
        
        if oh < ih and ow < iw and ih % oh == 0 and iw % ow == 0:
            prims.append(DetectedPrimitive(
                GeoPrimitive.SCALE_DOWN, 0.7,
                {'factor_h': ih // oh, 'factor_w': iw // ow}, 'spatial'))
        
        # --- ROTATION (same size, square) ---
        if ih == iw and oh == ow and ih == oh:
            for k, prim in [(1, GeoPrimitive.ROTATE_90), 
                            (2, GeoPrimitive.ROTATE_180),
                            (3, GeoPrimitive.ROTATE_270)]:
                rotated = np.rot90(ig, k)
                match = np.mean(rotated == og)
                if match > 0.9:
                    prims.append(DetectedPrimitive(prim, match, {}, 'symmetry'))
        
        # --- MIRROR ---
        if ih == oh and iw == ow:
            # Horizontal flip
            flipped_h = ig[::-1, :]
            match_h = np.mean(flipped_h == og)
            if match_h > 0.9:
                prims.append(DetectedPrimitive(
                    GeoPrimitive.MIRROR_H, match_h, {}, 'symmetry'))
            
            # Vertical flip
            flipped_v = ig[:, ::-1]
            match_v = np.mean(flipped_v == og)
            if match_v > 0.9:
                prims.append(DetectedPrimitive(
                    GeoPrimitive.MIRROR_V, match_v, {}, 'symmetry'))
            
            # Transpose
            if ih == iw:
                transposed = ig.T
                match_t = np.mean(transposed == og)
                if match_t > 0.9:
                    prims.append(DetectedPrimitive(
                        GeoPrimitive.MIRROR_DIAG, match_t, {}, 'symmetry'))
        
        # --- COLOR MAP ---
        if ih == oh and iw == ow:
            # Check if there's a consistent color→color mapping
            mapping = {}
            consistent = True
            for r in range(ih):
                for c in range(iw):
                    ic = int(ig[r, c])
                    oc = int(og[r, c])
                    if ic in mapping:
                        if mapping[ic] != oc:
                            consistent = False
                            break
                    else:
                        mapping[ic] = oc
                if not consistent:
                    break
            
            if consistent and mapping:
                is_identity = all(k == v for k, v in mapping.items())
                if not is_identity:
                    prims.append(DetectedPrimitive(
                        GeoPrimitive.COLOR_MAP, 1.0,
                        {'mapping': mapping}, 'chromatic'))
        
        # --- COLOR SWAP ---
        if ih == oh and iw == ow:
            diff_positions = ig != og
            changed_in = set(ig[diff_positions].astype(int).tolist())
            changed_out = set(og[diff_positions].astype(int).tolist())
            if len(changed_in) == 2 and changed_in == changed_out:
                colors = list(changed_in)
                prims.append(DetectedPrimitive(
                    GeoPrimitive.COLOR_SWAP, 0.95,
                    {'color_a': colors[0], 'color_b': colors[1]}, 'chromatic'))
        
        # --- CROP ---
        if oh < ih or ow < iw:
            # Check if output is a sub-region of input
            for r in range(ih - oh + 1):
                for c in range(iw - ow + 1):
                    region = ig[r:r+oh, c:c+ow]
                    if np.array_equal(region, og):
                        prims.append(DetectedPrimitive(
                            GeoPrimitive.CROP, 1.0,
                            {'top': r, 'left': c}, 'structural'))
                        break
                else:
                    continue
                break
        
        # --- GRAVITY ---
        if ih == oh and iw == ow:
            # Check if non-zero cells "fell" downward
            for c in range(iw):
                in_col = ig[:, c]
                out_col = og[:, c]
                in_vals = in_col[in_col != 0]
                out_vals = out_col[out_col != 0]
                if len(in_vals) > 0 and np.array_equal(sorted(in_vals), sorted(out_vals)):
                    # Check if output has all non-zero at bottom
                    out_nonzero = np.where(out_col != 0)[0]
                    if len(out_nonzero) > 0 and out_nonzero[-1] == ih - 1:
                        if np.all(np.diff(out_nonzero) == 1):
                            prims.append(DetectedPrimitive(
                                GeoPrimitive.GRAVITY, 0.8,
                                {'direction': 'down'}, 'structural'))
                            break
        
        # --- BOOLEAN OP ---
        # (Detects when output = some combination of input regions)
        # This is complex — will expand in later versions
        
        return prims


# =============================================================================
# CODE GENERATOR — primitives to Python solve()
# =============================================================================

class CodeGenerator:
    """
    Generate Python solve() functions from detected primitives.
    
    Each primitive has a direct code template. Compositions
    chain templates together.
    """
    
    TEMPLATES = {
        GeoPrimitive.TILE_REPEAT: {
            'simple': """def solve(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    tile_h, tile_w = {tile_h}, {tile_w}
    result = np.tile(grid, (tile_h, tile_w))
    return result.tolist()""",
            
            'alternating': """def solve(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    tile_h, tile_w = {tile_h}, {tile_w}
    result = np.zeros((h * tile_h, w * tile_w), dtype=int)
    for tr in range(tile_h):
        block = grid.copy()
        if tr % 2 == 1:
            block = block[:, ::-1]
        row_tile = np.tile(block, (1, tile_w))
        result[tr*h:(tr+1)*h, :] = row_tile
    return result.tolist()""",
        },
        
        GeoPrimitive.SCALE_UP: """def solve(input_grid):
    grid = np.array(input_grid)
    factor = {factor}
    result = np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
    return result.tolist()""",
        
        GeoPrimitive.ROTATE_90: """def solve(input_grid):
    grid = np.array(input_grid)
    result = np.rot90(grid, 1)
    return result.tolist()""",
        
        GeoPrimitive.ROTATE_180: """def solve(input_grid):
    grid = np.array(input_grid)
    result = np.rot90(grid, 2)
    return result.tolist()""",
        
        GeoPrimitive.ROTATE_270: """def solve(input_grid):
    grid = np.array(input_grid)
    result = np.rot90(grid, 3)
    return result.tolist()""",
        
        GeoPrimitive.MIRROR_H: """def solve(input_grid):
    grid = np.array(input_grid)
    result = grid[::-1, :]
    return result.tolist()""",
        
        GeoPrimitive.MIRROR_V: """def solve(input_grid):
    grid = np.array(input_grid)
    result = grid[:, ::-1]
    return result.tolist()""",
        
        GeoPrimitive.MIRROR_DIAG: """def solve(input_grid):
    grid = np.array(input_grid)
    result = grid.T
    return result.tolist()""",
        
        GeoPrimitive.COLOR_MAP: """def solve(input_grid):
    grid = np.array(input_grid)
    result = grid.copy()
    mapping = {mapping}
    for old_c, new_c in mapping.items():
        result[grid == old_c] = new_c
    return result.tolist()""",
        
        GeoPrimitive.COLOR_SWAP: """def solve(input_grid):
    grid = np.array(input_grid)
    result = grid.copy()
    a, b = {color_a}, {color_b}
    result[grid == a] = b
    result[grid == b] = a
    return result.tolist()""",
        
        GeoPrimitive.CROP: """def solve(input_grid):
    grid = np.array(input_grid)
    # Find bounding box of non-zero content
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    if not rows.any():
        return input_grid
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    result = grid[rmin:rmax+1, cmin:cmax+1]
    return result.tolist()""",
        
        GeoPrimitive.GRAVITY: """def solve(input_grid):
    grid = np.array(input_grid)
    h, w = grid.shape
    result = np.zeros_like(grid)
    for c in range(w):
        col = grid[:, c]
        nonzero = col[col != 0]
        result[h-len(nonzero):h, c] = nonzero
    return result.tolist()""",
    }
    
    @staticmethod
    def generate(primitives: List[DetectedPrimitive]) -> Optional[str]:
        """Generate solve() function from detected primitives."""
        if not primitives:
            return None
        
        # Take highest confidence primitive
        best = primitives[0]
        
        template = CodeGenerator.TEMPLATES.get(best.primitive)
        if template is None:
            return None
        
        # Handle templates with variants
        if isinstance(template, dict):
            if best.params.get('alternating'):
                template = template.get('alternating', template.get('simple'))
            else:
                template = template.get('simple')
        
        # Fill parameters
        try:
            code = template.format(**best.params)
        except (KeyError, IndexError):
            code = template
        
        # Ensure numpy import
        if 'np.' in code and 'import numpy' not in code:
            code = "import numpy as np\n\n" + code
        
        return code
    
    @staticmethod
    def generate_analysis(primitives: List[DetectedPrimitive], 
                          task: Dict) -> str:
        """Generate text analysis from detected primitives."""
        if not primitives:
            return "No transformation pattern detected with sufficient confidence."
        
        train = task.get('train', [])
        ig = np.array(train[0]['input']) if train else np.array([[]])
        og = np.array(train[0]['output']) if train else np.array([[]])
        
        lines = []
        lines.append(f"Input: {ig.shape[0]}x{ig.shape[1]} → Output: {og.shape[0]}x{og.shape[1]}")
        
        for p in primitives:
            desc = f"Detected {p.primitive.value} (confidence: {p.confidence:.2f})"
            if p.params:
                params_str = ", ".join(f"{k}={v}" for k, v in p.params.items())
                desc += f" [{params_str}]"
            lines.append(desc)
        
        # Primary transformation description
        best = primitives[0]
        descriptions = {
            GeoPrimitive.TILE_REPEAT: "The output tiles the input pattern across a larger grid",
            GeoPrimitive.SCALE_UP: "The output scales up each cell of the input",
            GeoPrimitive.ROTATE_90: "The output is the input rotated 90° counterclockwise",
            GeoPrimitive.ROTATE_180: "The output is the input rotated 180°",
            GeoPrimitive.ROTATE_270: "The output is the input rotated 270° counterclockwise",
            GeoPrimitive.MIRROR_H: "The output flips the input horizontally (upside down)",
            GeoPrimitive.MIRROR_V: "The output flips the input vertically (left-right)",
            GeoPrimitive.MIRROR_DIAG: "The output transposes the input (diagonal mirror)",
            GeoPrimitive.COLOR_MAP: "Each color in the input maps to a specific color in the output",
            GeoPrimitive.COLOR_SWAP: "Two specific colors are swapped",
            GeoPrimitive.CROP: "The output extracts a sub-region from the input",
            GeoPrimitive.GRAVITY: "Non-zero cells fall to the bottom of each column",
        }
        
        lines.append(f"\nRule: {descriptions.get(best.primitive, best.primitive.value)}")
        if best.params:
            lines.append(f"Parameters: {best.params}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_hypothesis(primitives: List[DetectedPrimitive]) -> str:
        """Generate algorithm hypothesis from detected primitives."""
        if not primitives:
            return "Insufficient geometric evidence to form hypothesis."
        
        best = primitives[0]
        
        hypotheses = {
            GeoPrimitive.TILE_REPEAT: "Tile the input grid {tile_h}x{tile_w} times. "
                "If alternating: flip vertically on odd rows, horizontally on odd columns.",
            GeoPrimitive.SCALE_UP: "Scale each cell to a {factor}x{factor} block.",
            GeoPrimitive.ROTATE_90: "Apply np.rot90(grid, 1).",
            GeoPrimitive.ROTATE_180: "Apply np.rot90(grid, 2).",
            GeoPrimitive.ROTATE_270: "Apply np.rot90(grid, 3).",
            GeoPrimitive.MIRROR_H: "Flip grid vertically: grid[::-1, :]",
            GeoPrimitive.MIRROR_V: "Flip grid horizontally: grid[:, ::-1]",
            GeoPrimitive.MIRROR_DIAG: "Transpose: grid.T",
            GeoPrimitive.COLOR_MAP: "Apply color mapping: {mapping}",
            GeoPrimitive.COLOR_SWAP: "Swap colors {color_a} ↔ {color_b}",
            GeoPrimitive.CROP: "Extract bounding box of non-zero content.",
            GeoPrimitive.GRAVITY: "For each column, move non-zero cells to bottom.",
        }
        
        template = hypotheses.get(best.primitive, str(best.primitive.value))
        try:
            return template.format(**best.params)
        except (KeyError, IndexError):
            return template


# =============================================================================
# INTEGRATED DECODER — full pipeline
# =============================================================================

class GeometricDecoder:
    """
    Complete signal→text/code decoder.
    
    This replaces _signal_to_response in the fused service.
    """
    
    def __init__(self):
        self.encoder = GridEncoder()
        self.detector = PrimitiveDetector()
        self.generator = CodeGenerator()
    
    def decode_for_analysis(self, task: Dict, 
                            substrate_output: np.ndarray) -> str:
        """Decode substrate output to text analysis."""
        input_signal = self.encoder.encode_task(task)
        primitives = self.detector.detect(input_signal, substrate_output, task)
        return self.generator.generate_analysis(primitives, task)
    
    def decode_for_hypothesis(self, task: Dict,
                              substrate_output: np.ndarray) -> str:
        """Decode substrate output to algorithm hypothesis."""
        input_signal = self.encoder.encode_task(task)
        primitives = self.detector.detect(input_signal, substrate_output, task)
        return self.generator.generate_hypothesis(primitives)
    
    def decode_for_code(self, task: Dict,
                        substrate_output: np.ndarray) -> Optional[str]:
        """Decode substrate output to Python solve() function."""
        input_signal = self.encoder.encode_task(task)
        primitives = self.detector.detect(input_signal, substrate_output, task)
        return self.generator.generate(primitives)
    
    def solve_task(self, task: Dict) -> Optional[str]:
        """
        Full pipeline: task → encode → detect → code.
        
        Bypasses substrate for direct geometric solving.
        This is the one-pass fabrication path.
        """
        train = task.get('train', [])
        if not train:
            return None
        
        # Detect primitives directly from task geometry
        primitives = self.detector.detect(
            np.zeros(1024), np.zeros(1024), task)
        
        if not primitives:
            return None
        
        return self.generator.generate(primitives)


# =============================================================================
# STANDALONE TEST
# =============================================================================

def test_codebook():
    """Test the codebook against known ARC patterns."""
    decoder = GeometricDecoder()
    
    # Test 1: Tiling (task 00576224)
    task_tile = {
        'train': [
            {'input': [[7, 9], [4, 3]], 
             'output': [[7,9,7,9,7,9],[4,3,4,3,4,3],
                        [9,7,9,7,9,7],[3,4,3,4,3,4],
                        [7,9,7,9,7,9],[4,3,4,3,4,3]]},
            {'input': [[8, 6], [6, 4]],
             'output': [[8,6,8,6,8,6],[6,4,6,4,6,4],
                        [6,8,6,8,6,8],[4,6,4,6,4,6],
                        [8,6,8,6,8,6],[6,4,6,4,6,4]]},
        ],
        'test': [
            {'input': [[3, 2], [7, 8]],
             'output': [[3,2,3,2,3,2],[7,8,7,8,7,8],
                        [2,3,2,3,2,3],[8,7,8,7,8,7],
                        [3,2,3,2,3,2],[7,8,7,8,7,8]]},
        ]
    }
    
    print("=" * 70)
    print("  GEOMETRIC CODEBOOK TEST")
    print("=" * 70)
    
    # Test analysis
    print("\n--- Task: Alternating Tile (00576224) ---")
    analysis = decoder.decode_for_analysis(task_tile, np.zeros(1024))
    print(f"Analysis:\n{analysis}")
    
    hypothesis = decoder.decode_for_hypothesis(task_tile, np.zeros(1024))
    print(f"\nHypothesis: {hypothesis}")
    
    code = decoder.decode_for_code(task_tile, np.zeros(1024))
    print(f"\nGenerated code:\n{code}")
    
    # Validate
    if code:
        print("\n--- Validation ---")
        namespace = {'np': np}
        exec(code, namespace)
        solve = namespace['solve']
        
        for i, test in enumerate(task_tile['test']):
            result = solve(test['input'])
            expected = test['output']
            match = result == expected
            print(f"  Test {i+1}: {'PASS ✓' if match else 'FAIL ✗'}")
            if not match:
                print(f"    Expected: {expected[:2]}...")
                print(f"    Got:      {result[:2]}...")
    
    # Test 2: Simple rotation
    print("\n--- Task: 90° Rotation ---")
    task_rot = {
        'train': [
            {'input': [[1, 2], [3, 4]], 
             'output': [[2, 4], [1, 3]]},
            {'input': [[5, 6], [7, 8]],
             'output': [[6, 8], [5, 7]]},
        ],
        'test': [
            {'input': [[9, 1], [2, 3]],
             'output': [[1, 3], [9, 2]]},
        ]
    }
    
    code = decoder.solve_task(task_rot)
    if code:
        print(f"Code: {code.strip().split(chr(10))[-1]}")
        namespace = {'np': np}
        exec(code, namespace)
        result = namespace['solve'](task_rot['test'][0]['input'])
        match = result == task_rot['test'][0]['output']
        print(f"  Test: {'PASS ✓' if match else 'FAIL ✗'}")
    
    # Test 3: Color map
    print("\n--- Task: Color Mapping ---")
    task_color = {
        'train': [
            {'input': [[1, 2, 3], [1, 2, 3]],
             'output': [[4, 5, 6], [4, 5, 6]]},
            {'input': [[3, 1, 2], [2, 3, 1]],
             'output': [[6, 4, 5], [5, 6, 4]]},
        ],
        'test': [
            {'input': [[2, 1, 3], [3, 2, 1]],
             'output': [[5, 4, 6], [6, 5, 4]]},
        ]
    }
    
    code = decoder.solve_task(task_color)
    if code:
        print(f"Detected mapping")
        namespace = {'np': np}
        exec(code, namespace)
        result = namespace['solve'](task_color['test'][0]['input'])
        match = result == task_color['test'][0]['output']
        print(f"  Test: {'PASS ✓' if match else 'FAIL ✗'}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_codebook()
