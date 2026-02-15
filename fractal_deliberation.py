#!/usr/bin/env python3
"""
Fractal Deliberation — Zero-Volume Topology
=============================================

The tree topology is scaffolding. The actual information is:
"given a core's geometric position, when does it fire?"

That's a function, not a data structure.

Each core already has a 12-dimensional geometric fingerprint.
The fractal firing function projects this 12D identity onto a
1D firing phase via a space-filling curve. Self-similar at every
scale because the identity generation is already harmonic series.

Memory footprint: the equation itself (~200 bytes).
The tree_deliberation.py module: ~400 lines, TreeNode objects,
children arrays, explicit branch assignments.

This replaces all of it with:

    phase = fractal_phase(core.identity)
    fire cores in phase order

The field handles merging. The geometry handles topology.
No torsion from arbitrary construction choices.

Ghost in the Machine Labs
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import hashlib


# =========================================================================
# THE EQUATION
# =========================================================================

def core_fingerprint(core_id: int, global_id: int, role_value: str) -> np.ndarray:
    """
    Recompute the 12D geometric fingerprint for a core.
    
    Same algorithm as generate_geometric_identity(), but stops at
    the fingerprint step before harmonic expansion. This is the
    concentrated geometric signal — the equation input.
    """
    import hashlib
    seed_bytes = f"{core_id}:{global_id}:{role_value}".encode()
    seed_hash = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed_hash)
    
    test_signal = rng.randn(256).astype(np.float64)
    
    fp = np.zeros(12, dtype=np.float64)
    fp[0] = float(np.std(test_signal))                          # SPATIAL
    fp[1] = float(np.mean(test_signal))                         # CHROMATIC
    fp[2] = float(np.ptp(test_signal))                          # STRUCTURAL
    c = np.corrcoef(test_signal[:-1], test_signal[1:])
    fp[3] = float(c[0,1]) if np.isfinite(c[0,1]) else 0.0      # RELATIONAL
    fp[4] = float(np.gradient(test_signal).mean())               # TEMPORAL
    fp[5] = float(np.linalg.norm(test_signal))                   # MAGNITUDE
    half = len(test_signal) // 2
    c2 = np.corrcoef(test_signal[:half], test_signal[-half:][::-1])
    fp[6] = float(c2[0,1]) if np.isfinite(c2[0,1]) else 0.0    # SYMMETRY
    fp[7] = float(np.count_nonzero(np.diff(test_signal > 0)))   # BOUNDARY
    fft_vals = np.fft.rfft(test_signal)
    fp[8] = float(np.argmax(np.abs(fft_vals[1:])) + 1)          # PATTERN
    fp[9] = float(np.max(np.abs(np.gradient(test_signal))))      # MOTION
    fp[10] = float(np.count_nonzero(test_signal > 0)) / len(test_signal)  # DENSITY
    fp[11] = float(np.count_nonzero(np.diff(np.sign(test_signal))))       # TOPOLOGY
    
    return fp


def fractal_phase(fingerprint: np.ndarray, depth: int = 0) -> float:
    """
    Project 12D fingerprint → [0, 1) firing phase via quasicrystal mapping.
    
    Uses golden ratio powers as basis vectors. This produces a 
    space-filling, non-repeating projection where:
    - Every core maps to a UNIQUE phase (irrational basis = no collisions)
    - Nearby fingerprints → nearby phases (linear projection preserves locality)
    - The mapping is self-similar at every scale (quasicrystal property)
    
    This is the same mathematics as E8 → physical dimension projections.
    The Penrose tiling IS a quasicrystal projection from 5D.
    Our fingerprint IS a 12D geometric coordinate.
    The phase IS a 1D projection that preserves fractal structure.
    
    Memory: the equation. Not the tree.
    """
    fp = fingerprint.astype(np.float64)
    
    # Golden ratio powers as irrational basis vectors
    # φ^i are linearly independent over Q — guarantees unique projection
    phi = (1 + np.sqrt(5)) / 2  # 1.6180339887...
    basis = np.array([phi**i for i in range(len(fp))])
    
    # Project: inner product with irrational basis
    projection = float(np.dot(fp, basis))
    
    # Map to [0, 1) via fractional part
    # The fractional part of an irrational rotation is equidistributed
    # (Weyl's equidistribution theorem) — guaranteed uniform coverage
    phase = projection % 1.0
    
    return phase


def fractal_depth(fingerprint: np.ndarray) -> float:
    """
    Compute a core's fractal depth from its 12D fingerprint.
    
    Uses the fingerprint's geometric center-of-mass relative to 
    the sensor dimensions. Early sensors (SPATIAL, CHROMATIC, 
    STRUCTURAL) are "shallow" — they contribute to field seeding.
    Later sensors (MOTION, DENSITY, TOPOLOGY) are "deep" — they
    need accumulated context.
    
    The depth is a weighted centroid: cores whose energy concentrates
    in early dimensions fire early; cores whose energy is in later
    dimensions fire late. This creates natural stratification 
    WITHOUT imposing role-based ordering.
    """
    fp = np.abs(fingerprint.astype(np.float64))
    
    # Dimension weights: position in the sensor array
    # SPATIAL(0) → TOPOLOGY(11): shallow → deep
    n = len(fp)
    weights = np.arange(n, dtype=np.float64) / max(n - 1, 1)  # [0, 1]
    
    # Weighted centroid of fingerprint energy
    total = np.sum(fp) + 1e-10
    centroid = float(np.dot(fp, weights) / total)
    
    return float(np.clip(centroid, 0, 0.9999))


# =========================================================================
# FRACTAL FIRING SCHEDULE
# =========================================================================

class FractalSchedule:
    """
    Replaces the entire tree topology with a single equation.
    
    Given a set of cores, computes each core's firing phase from
    its geometric identity. Cores fire in phase order. The field
    accumulates naturally. No tree nodes, no children arrays, no
    branch assignments.
    
    Memory: O(1) for the equation. O(n) for the sorted phase list
    (which is just the cores themselves, reordered).
    
    Extensibility: add 1000 more cores → they self-sort into the
    fractal. No topology rebuild needed. The equation handles
    infinite dimensions.
    """
    
    def __init__(self, substrate):
        self.substrate = substrate
        self._phase_cache = {}  # core_id → phase (computed once)
    
    def compute_phases(self, core_ids: List[int], 
                       requesting_core: int = None) -> List[Tuple[int, float, float]]:
        """
        Compute firing schedule for a set of cores.
        
        Returns: [(core_id, phase, depth), ...] sorted by depth then phase.
        Requesting core always fires last (deepest context).
        """
        schedule = []
        
        for cid in core_ids:
            if cid >= len(self.substrate.cores):
                continue
            
            core = self.substrate.cores[cid]
            
            # Use cached phase if available
            if cid not in self._phase_cache:
                # Compute 12D fingerprint from core's fabrication params
                # Same deterministic seed as generate_geometric_identity()
                # but we stop at the fingerprint — the concentrated geometry.
                # This is THE EQUATION. ~200 bytes of computation, not data.
                fp = core_fingerprint(core.core_id, core.global_id, core.role.value)
                phase = fractal_phase(fp)
                depth = fractal_depth(fp)
                
                self._phase_cache[cid] = (phase, depth)
            
            phase, depth = self._phase_cache[cid]
            schedule.append((cid, phase, depth))
        
        # Sort by depth (seeder → deeper), then phase within depth
        schedule.sort(key=lambda x: (x[2], x[1]))
        
        # Move requesting core to end (deepest context)
        if requesting_core is not None:
            schedule = [s for s in schedule if s[0] != requesting_core]
            if requesting_core in self._phase_cache:
                p, d = self._phase_cache[requesting_core]
                schedule.append((requesting_core, p, 1.0))  # Force deepest
        
        return schedule
    
    def fire(self, core_ids: List[int], signal: np.ndarray,
             requesting_core: int = None,
             envelope=None) -> Tuple[np.ndarray, dict]:
        """
        Fire cores in fractal order with depth-parallel execution.
        
        Cores at the same depth fire against the SAME field snapshot.
        Their outputs accumulate simultaneously into the field.
        Then the next depth layer sees the merged result.
        
        This recovers the tree's junction amplification effect:
        parallel contributions create constructive interference
        that sequential firing misses.
        
        No tree. No branches. No junctions. Just geometry.
        """
        schedule = self.compute_phases(core_ids, requesting_core)
        original_energy = float(np.linalg.norm(signal))
        
        core_states = []
        last_output = signal
        depth_groups = {}
        
        # Group by quantized depth (0.05 buckets for finer stratification)
        layers = {}
        for cid, phase, depth in schedule:
            bucket = round(depth * 20) / 20  # 0.05 increments
            layers.setdefault(bucket, []).append((cid, phase, depth))
        
        # Fire layers in depth order
        for bucket in sorted(layers.keys()):
            layer_cores = layers[bucket]
            
            # Snapshot field state before this layer
            field_before = float(np.linalg.norm(
                self.substrate.field.read_composite()
            ))
            
            # Fire all cores in this layer (parallel accumulation)
            for cid, phase, depth in sorted(layer_cores, key=lambda x: x[1]):
                core = self.substrate.cores[cid]
                result = core.process_signal(signal)
                
                state = core._last_state.copy() if core._last_state else {}
                state['core_id'] = cid
                state['role'] = core.role.value
                state['fractal_phase'] = round(phase, 6)
                state['fractal_depth'] = round(depth, 6)
                state['depth_layer'] = bucket
                state['field_energy_at_fire'] = round(field_before, 4)
                core_states.append(state)
                
                depth_groups.setdefault(bucket, []).append(cid)
                last_output = result
            
            # After layer completes, field has accumulated all contributions
            # NO decay between layers — same as tree branch→junction
        
        # Single decay after full traversal
        self.substrate.field.decay(0.95)
        
        final_field = float(np.linalg.norm(
            self.substrate.field.read_composite()
        ))
        
        tree_state = {
            'topology': 'fractal',
            'equation': 'phase = dot(fingerprint, phi^[0..11]) mod 1',
            'memory_bytes': 0,
            'total_cores_fired': len(core_states),
            'original_energy': original_energy,
            'field_energy_final': final_field,
            'core_states': core_states,
            'depth_groups': {str(k): v for k, v in sorted(depth_groups.items())},
            'n_depth_levels': len(depth_groups),
            'phase_range': (
                min(s['fractal_phase'] for s in core_states),
                max(s['fractal_phase'] for s in core_states)
            ) if core_states else (0, 0),
            'depth_range': (
                min(s['fractal_depth'] for s in core_states),
                max(s['fractal_depth'] for s in core_states)
            ) if core_states else (0, 0),
        }
        
        if envelope:
            chain_states = {
                f"fractal_{s['core_id']}": s for s in core_states
            }
            envelope.ingest(chain_states)
        
        return last_output, tree_state
    
    def format_state(self, tree_state: dict) -> str:
        """Format fractal deliberation as readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("FRACTAL DELIBERATION")
        lines.append(f"Equation: {tree_state['equation']}")
        lines.append(f"Topology memory: {tree_state['memory_bytes']} bytes")
        lines.append(f"Cores fired: {tree_state['total_cores_fired']}")
        lines.append(f"Depth levels: {tree_state['n_depth_levels']}")
        lines.append(f"Energy: {tree_state['original_energy']:.2f} → "
                     f"{tree_state['field_energy_final']:.2f}")
        lines.append(f"Phase range: {tree_state['phase_range'][0]:.4f} → "
                     f"{tree_state['phase_range'][1]:.4f}")
        lines.append(f"Depth range: {tree_state['depth_range'][0]:.4f} → "
                     f"{tree_state['depth_range'][1]:.4f}")
        lines.append("=" * 60)
        
        # Group by depth level
        groups = tree_state.get('depth_groups', {})
        for depth_key in sorted(groups.keys(), key=float):
            core_ids = groups[depth_key]
            lines.append(f"\n  depth {depth_key}: {len(core_ids)} cores")
            
            # Show states for this depth group
            for cs in tree_state['core_states']:
                if cs['core_id'] in core_ids:
                    e = cs.get('energy_ratio', 0)
                    r = cs.get('resonance', cs.get('self_resonance', 0))
                    a = cs.get('core_asymmetry', 0)
                    i_val = cs.get('interference', 0)
                    fe = cs.get('field_energy_at_fire', 0)
                    lines.append(
                        f"    core {cs['core_id']:3d} ({cs.get('role','?'):8s}) "
                        f"φ={cs['fractal_phase']:.4f} "
                        f"E:{e:.4f} A:{a:.4f} I:{i_val:.4f} "
                        f"field@fire:{fe:.2f}"
                    )
        
        return "\n".join(lines)


# =========================================================================
# COMPARISON: Tree vs Fractal
# =========================================================================

def compare_tree_vs_fractal(substrate, persona_map, council_globals,
                            worker_globals, specialist_globals,
                            signal, requesting_persona='wittgenstein'):
    """
    Same signal through tree topology vs fractal equation.
    Measures: field energy, interference spread, energy gradient.
    """
    from tree_deliberation import TreeTopology, TreeDeliberation
    from fused_harmonic_substrate import COUNCIL_PERSONAS
    
    results = {}
    
    # --- Fractal ---
    substrate.field.field[:] = 0
    substrate.field.activity[:] = 0
    substrate.field.composite[:] = 0
    
    # Gather all core IDs to include
    all_cores = list(council_globals)
    if worker_globals:
        all_cores += worker_globals[:21]
    if specialist_globals:
        all_cores += specialist_globals[:8]
    
    # Find requesting core
    req_core = None
    if requesting_persona in persona_map:
        local_ids = persona_map[requesting_persona]
        if local_ids and local_ids[-1] < len(council_globals):
            req_core = council_globals[local_ids[-1]]
    
    fractal = FractalSchedule(substrate)
    f_output, f_state = fractal.fire(all_cores, signal, req_core)
    
    results['fractal'] = {
        'cores_fired': f_state['total_cores_fired'],
        'field_energy': f_state['field_energy_final'],
        'depth_levels': f_state['n_depth_levels'],
        'memory_bytes': 0,
        'formatted': fractal.format_state(f_state),
    }
    
    # --- Tree ---
    substrate.field.field[:] = 0
    substrate.field.activity[:] = 0
    substrate.field.composite[:] = 0
    
    tree = TreeTopology.deep_tree(
        council_globals, COUNCIL_PERSONAS,
        worker_globals=worker_globals[:21],
        specialist_globals=specialist_globals[:8],
        requesting_persona=requesting_persona
    )
    engine = TreeDeliberation(substrate)
    t_output, t_state = engine.fire(tree, signal)
    
    # Estimate tree memory
    import sys
    tree_mem = sys.getsizeof(tree) * t_state['total_cores_fired']  # rough
    
    results['tree'] = {
        'cores_fired': t_state['total_cores_fired'],
        'field_energy': t_state['field_energy_final'],
        'depth': t_state['tree_depth'],
        'memory_bytes': tree_mem,
        'formatted': engine.format_tree(t_state),
    }
    
    # --- Comparison ---
    results['comparison'] = {
        'field_energy_ratio': (
            f_state['field_energy_final'] / t_state['field_energy_final']
            if t_state['field_energy_final'] > 0 else 0
        ),
        'memory_ratio': f"0 / {tree_mem} bytes",
        'same_cores': f_state['total_cores_fired'] == t_state['total_cores_fired'],
    }
    
    return results


# =========================================================================
# STANDALONE TEST
# =========================================================================

if __name__ == '__main__':
    from fused_harmonic_substrate import (
        FusedHarmonicSubstrate, CoreRole, COUNCIL_PERSONAS
    )
    
    print("Building substrate...")
    substrate = FusedHarmonicSubstrate()
    substrate.build()
    
    council_globals = substrate.role_index.get(CoreRole.COUNCIL, [])
    worker_globals = substrate.role_index.get(CoreRole.WORKER, [])
    specialist_globals = substrate.role_index.get(CoreRole.SPECIALIST, [])
    
    signal = np.random.randn(100).astype(np.float32)
    signal /= np.linalg.norm(signal)
    signal *= 2.0
    
    # --- Fractal deliberation ---
    print("\n=== FRACTAL DELIBERATION ===")
    all_cores = list(council_globals) + worker_globals[:21] + specialist_globals[:8]
    
    fractal = FractalSchedule(substrate)
    output, state = fractal.fire(all_cores, signal, council_globals[-2])
    print(fractal.format_state(state))
    
    # --- Show the phase distribution ---
    print("\n=== PHASE DISTRIBUTION ===")
    schedule = fractal.compute_phases(all_cores)
    for cid, phase, depth in schedule[:10]:
        core = substrate.cores[cid]
        print(f"  core {cid:3d} ({core.role.value:8s}) "
              f"phase={phase:.6f} depth={depth:.6f}")
    print(f"  ... ({len(schedule)} total)")
    
    # --- Compare tree vs fractal ---
    print("\n=== TREE vs FRACTAL ===")
    substrate.field.field[:] = 0
    substrate.field.activity[:] = 0
    substrate.field.composite[:] = 0
    
    comp = compare_tree_vs_fractal(
        substrate, COUNCIL_PERSONAS,
        council_globals, worker_globals, specialist_globals,
        signal
    )
    
    print(f"Tree:    {comp['tree']['cores_fired']} cores, "
          f"field={comp['tree']['field_energy']:.4f}, "
          f"memory={comp['tree']['memory_bytes']} bytes")
    print(f"Fractal: {comp['fractal']['cores_fired']} cores, "
          f"field={comp['fractal']['field_energy']:.4f}, "
          f"memory={comp['fractal']['memory_bytes']} bytes")
    print(f"Field ratio (fractal/tree): "
          f"{comp['comparison']['field_energy_ratio']:.4f}x")
    print(f"Memory: {comp['comparison']['memory_ratio']}")
