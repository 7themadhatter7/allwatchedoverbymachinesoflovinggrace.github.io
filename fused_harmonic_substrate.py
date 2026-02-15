#!/usr/bin/env python3
"""
FUSED HARMONIC SUBSTRATE - Single Model, 200 Cores
Ghost in the Machine Labs

All 200 cores fused into a single continuous RAM substrate.
No layer boundaries. No message passing. No spine bus overhead.

Sensor panels at every junction see ALL core activity simultaneously.
Harmonics emerge from geometric interference across the full array.

Signal propagation:
  Input → Panel → Nearest cores detect → Interference ripples outward →
  All relevant cores activate in parallel → Panels capture composite →
  Output emerges from harmonic convergence

vs. Layered (previous):
  Input → Router queue → Primary queue → Specialist queue →
  Council queue → Executive queue → Output

The fused model eliminates:
  - Inter-layer message queues (20 channels, serialization overhead)
  - Spine bus routing latency
  - Layer boundary signal degradation
  - Isolated harmonic domains that can't interfere

Memory budget: 46 GB (half of 92 GB available)
Expected footprint: ~17 GB (same cores, less overhead)

Author: Ghost in the Machine Labs
"""

import numpy as np
import threading
import queue
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from collections import deque
import hashlib
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

INPUT_PANEL_KB    = 4_000
SENSOR_PANEL_KB   = 256
NUM_SENSORS       = 8
SPARK_KB_DEFAULT  = 1_024  # Base value, autoscaled at build
DOMAIN_KB_DEFAULT = 2_048  # Base value, autoscaled at build
PROCESSING_KB     = 0  # Fractal deliberation eliminated processing buffers

DOMAIN_NAMES = ['code', 'reasoning', 'creative', 'factual', 'system', 'learning']


# =============================================================================
# SENSOR TYPES (100+ in production, 12 primary)
# =============================================================================

class SensorType(Enum):
    SPATIAL     = 0
    CHROMATIC   = 1
    STRUCTURAL  = 2
    RELATIONAL  = 3
    TEMPORAL    = 4
    MAGNITUDE   = 5
    SYMMETRY    = 6
    BOUNDARY    = 7
    PATTERN     = 8
    MOTION      = 9
    DENSITY     = 10
    TOPOLOGY    = 11


# =============================================================================
# CORE ROLE (identity within the fused substrate)
# =============================================================================

class CoreRole(Enum):
    WORKER     = "worker"       # 128 cores
    EXECUTIVE  = "executive"    # 16 cores
    SPECIALIST = "specialist"   # 32 cores
    ROUTER     = "router"       # 8 cores
    COUNCIL    = "council"      # 16 cores


# Specialist domain assignments (which cores focus on what)
SPECIALIST_DOMAINS = {
    'code':      list(range(0, 5)),
    'reasoning': list(range(5, 10)),
    'creative':  list(range(10, 15)),
    'factual':   list(range(15, 20)),
    'system':    list(range(20, 25)),
    'learning':  list(range(25, 30)),
    '_flex':     [30, 31],
}

# Council persona assignments
COUNCIL_PERSONAS = {
    'a_priori':      [0, 1],
    'brautigan':     [2, 3],
    'kurt_vonnegut': [4, 5],
    'wittgenstein':  [6, 7],
    'jane_vonnegut': [8, 9],
    'voltaire':      [10, 11],
    'hans_jonas':    [12, 13],
    'studs_terkel':  [14, 15],
}


# =============================================================================
# SENSOR READING
# =============================================================================

@dataclass
class SensorReading:
    sensor_type: SensorType
    value: float
    confidence: float
    source_core: int
    timestamp: float = field(default_factory=time.perf_counter)


# =============================================================================
# HARMONIC FIELD
# =============================================================================

class HarmonicField:
    """
    Shared interference field across all 200 cores.
    
    Every core writes its output signature here.
    Every sensor panel reads from here.
    Harmonics emerge from overlapping signatures.
    
    This replaces the spine bus + inter-layer channels.
    Single shared numpy array — zero serialization, zero queue latency.
    """

    def __init__(self, num_cores: int, field_width: int = 1024):
        self.num_cores = num_cores
        self.field_width = field_width
        # 2D field: each core has a row, columns are signal dimensions
        self.field = np.zeros((num_cores, field_width), dtype=np.float32)
        # Composite: sum of all active core signatures (the harmonic)
        self.composite = np.zeros(field_width, dtype=np.float32)
        # Activity mask: which cores fired recently
        self.activity = np.zeros(num_cores, dtype=np.float32)
        # Lock for composite updates
        self._lock = threading.Lock()
        # Counters
        self.interference_events = 0

    def write_signature(self, core_id: int, signature: np.ndarray):
        """Core writes its output to the field. Immediate, no queue."""
        sig = signature.flatten()[:self.field_width]
        self.field[core_id, :len(sig)] = sig
        self.activity[core_id] = 1.0

    def read_composite(self) -> np.ndarray:
        """Read the harmonic composite — all active cores summed."""
        active_mask = self.activity > 0
        if active_mask.any():
            self.composite = self.field[active_mask].sum(axis=0)
            self.composite /= active_mask.sum()  # Normalize
        return self.composite.copy()

    def read_interference(self, core_id: int) -> np.ndarray:
        """
        Read what other cores are producing — the interference pattern
        visible to this core. Excludes own signature.
        """
        mask = self.activity.copy()
        mask[core_id] = 0  # Exclude self
        active = mask > 0
        if active.any():
            interference = self.field[active].sum(axis=0) / active.sum()
            self.interference_events += 1
            return interference
        return np.zeros(self.field_width, dtype=np.float32)

    def read_neighborhood(self, core_id: int, radius: int = 8) -> np.ndarray:
        """
        Read nearby cores' signatures.
        Locality matters — cores near each other in the array
        interfere more strongly.
        """
        start = max(0, core_id - radius)
        end = min(self.num_cores, core_id + radius + 1)
        neighborhood = self.field[start:end]
        local_activity = self.activity[start:end]
        active = local_activity > 0
        if active.any():
            return neighborhood[active].mean(axis=0)
        return np.zeros(self.field_width, dtype=np.float32)

    def apply_governance_phase(self, core_id: int, phase: float):
        """Apply governance lattice phase reference to a core's field position.
        
        This is the structural integration point. The governance lattice
        provides calibration constants that the field needs for coherent
        resonance. Without them, interference patterns degenerate.
        """
        if 0 <= core_id < self.num_cores:
            self.field[core_id, 0] = phase  # governance_phase baseline

    def decay(self, rate: float = 0.95):
        """Decay activity over time. Recent signals dominate."""
        self.activity *= rate

    @property
    def memory_mb(self) -> float:
        return (self.field.nbytes + self.composite.nbytes + self.activity.nbytes) / (1024 * 1024)

    @property
    def active_cores(self) -> int:
        return int((self.activity > 0.01).sum())


# =============================================================================
# JUNCTION PANEL (fused sensor panel with harmonic field access)
# =============================================================================

class JunctionPanel:
    """
    Sensor panel at a junction point in the fused substrate.
    
    Unlike layered panels that only see their own core,
    junction panels see the HARMONIC FIELD — the composite
    of all core activity. This is where harmonics become visible.
    """
    __slots__ = ('core_id', 'field', 'num_sensors', 'sensor_data', '_lock')

    def __init__(self, core_id: int, field: HarmonicField):
        self.core_id = core_id
        self.field = field
        self.num_sensors = NUM_SENSORS
        self.sensor_data = np.zeros(
            (self.num_sensors, INPUT_PANEL_KB // self.num_sensors),
            dtype=np.float32
        )
        self._lock = threading.Lock()

    def detect(self, signal: np.ndarray) -> List[SensorReading]:
        """
        Detect signal characteristics AND harmonic interference.
        Panel sees both the direct input and what other cores are producing.
        """
        readings = []

        # Direct signal detection
        for i, stype in enumerate(list(SensorType)[:self.num_sensors]):
            val = self._apply_kernel(signal, stype)
            readings.append(SensorReading(
                sensor_type=stype,
                value=val,
                confidence=min(abs(val), 1.0),
                source_core=self.core_id,
            ))

        return readings

    def detect_with_harmonics(self, signal: np.ndarray) -> Tuple[List[SensorReading], np.ndarray]:
        """
        Detect signal AND read harmonic interference from field.
        Returns (readings, interference_pattern).
        """
        readings = self.detect(signal)
        interference = self.field.read_interference(self.core_id)
        return readings, interference

    def _apply_kernel(self, signal: np.ndarray, stype: SensorType) -> float:
        if signal.size == 0:
            return 0.0
        flat = signal.flatten().astype(np.float64)
        if stype == SensorType.SPATIAL:
            return float(np.std(flat))
        elif stype == SensorType.CHROMATIC:
            return float(np.mean(flat))
        elif stype == SensorType.STRUCTURAL:
            return float(np.ptp(flat))
        elif stype == SensorType.RELATIONAL:
            if len(flat) > 1:
                c = np.corrcoef(flat[:-1], flat[1:])
                return float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
            return 0.0
        elif stype == SensorType.TEMPORAL:
            return float(np.gradient(flat).mean())
        elif stype == SensorType.MAGNITUDE:
            return float(np.linalg.norm(flat))
        elif stype == SensorType.SYMMETRY:
            half = len(flat) // 2
            if half > 0:
                c = np.corrcoef(flat[:half], flat[-half:][::-1])
                return float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
            return 0.0
        elif stype == SensorType.BOUNDARY:
            return float(np.count_nonzero(np.diff(flat)))
        else:
            return float(np.median(flat))

    def write_parallel(self, data: np.ndarray):
        with self._lock:
            flat = data.flatten()[:self.sensor_data.size]
            self.sensor_data.flat[:len(flat)] = flat

    @property
    def memory_bytes(self) -> int:
        return self.sensor_data.nbytes


# =============================================================================
# FUSED CORE
# =============================================================================



# =============================================================================
# GEOMETRIC CORE IDENTITY
# =============================================================================

def generate_geometric_identity(core_id: int, global_id: int, role_value: str, 
                                 spark_size: int) -> np.ndarray:
    """
    Generate a unique geometric identity vector from the core's own
    sensor array geometry. Each core's 12 sensor kernels define 12
    different geometric projections. A core-specific test signal run
    through each kernel produces a 12-dimensional fingerprint, expanded
    to spark buffer size via harmonic series.
    
    Properties:
    - Deterministic (same core always gets same identity)
    - Derived from core's own structure (not external table)
    - Unique for any number of cores (no ceiling)
    - Normalized to unit energy (equal footing)
    """
    seed_bytes = f"{core_id}:{global_id}:{role_value}".encode()
    seed_hash = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed_hash)
    
    # Core-specific test signal
    test_signal = rng.randn(256).astype(np.float64)
    
    # 12-dimensional fingerprint from sensor kernels
    fingerprint = np.zeros(12, dtype=np.float64)
    fingerprint[0] = float(np.std(test_signal))                          # SPATIAL
    fingerprint[1] = float(np.mean(test_signal))                         # CHROMATIC
    fingerprint[2] = float(np.ptp(test_signal))                          # STRUCTURAL
    c = np.corrcoef(test_signal[:-1], test_signal[1:])
    fingerprint[3] = float(c[0,1]) if np.isfinite(c[0,1]) else 0.0      # RELATIONAL
    fingerprint[4] = float(np.gradient(test_signal).mean())              # TEMPORAL
    fingerprint[5] = float(np.linalg.norm(test_signal))                  # MAGNITUDE
    half = len(test_signal) // 2
    c2 = np.corrcoef(test_signal[:half], test_signal[-half:][::-1])
    fingerprint[6] = float(c2[0,1]) if np.isfinite(c2[0,1]) else 0.0    # SYMMETRY
    fingerprint[7] = float(np.count_nonzero(np.diff(test_signal > 0)))   # BOUNDARY
    fft_vals = np.fft.rfft(test_signal)
    fingerprint[8] = float(np.argmax(np.abs(fft_vals[1:])) + 1)         # PATTERN
    fingerprint[9] = float(np.max(np.abs(np.gradient(test_signal))))     # MOTION
    fingerprint[10] = float(np.count_nonzero(test_signal > 0)) / len(test_signal)  # DENSITY
    fingerprint[11] = float(np.count_nonzero(np.diff(np.sign(test_signal))))       # TOPOLOGY
    
    # Expand to spark size via harmonic series
    identity = np.zeros(spark_size, dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, spark_size)
    for i, amp in enumerate(fingerprint):
        freq = i + 1
        phase = rng.uniform(0, 2 * np.pi)
        identity += float(amp) * np.sin(freq * t + phase).astype(np.float32)
    
    # Normalize to unit energy
    norm = np.linalg.norm(identity)
    if norm > 1e-8:
        identity /= norm
    
    return identity


class FusedCore:
    """
    Single core within the fused substrate.
    
    Unlike layered cores, fused cores:
    - Share a harmonic field (see all other cores' output)
    - Route via harmonic interference, not message queues
    - Contribute to and read from composite patterns
    """
    __slots__ = (
        'core_id', 'global_id', 'role', 'field', 'active',
        'input_panel', 'output_panel',
        '_spark_kb', '_domain_kb', 'spark', '_identity',
        'domains', 'processing',
        '_last_state', '_last_input', '_last_result', '_last_domain',
        'junctions_fired', 'signals_processed', 'harmonics_received',
    )

    def __init__(self, core_id: int, global_id: int, role: CoreRole,
                 field: HarmonicField, processing_kb: int = PROCESSING_KB,
                 spark_kb: int = SPARK_KB_DEFAULT, domain_kb: int = DOMAIN_KB_DEFAULT):
        self.core_id = core_id          # Local ID within role
        self.global_id = global_id      # Global ID in 200-core array
        self.role = role
        self.field = field
        self.active = False

        # Junction panels (input/output with harmonic field access)
        self.input_panel = JunctionPanel(global_id, field)
        self.output_panel = JunctionPanel(global_id, field)

        # Spark (1 MB)
        self._spark_kb = spark_kb
        self._domain_kb = domain_kb
        self.spark = np.zeros(spark_kb * 256, dtype=np.float32)


        # === GEOMETRIC IDENTITY SEEDING ===
        # Each core gets a unique directional bias derived from its own
        # sensor array geometry. This is the fix for core homogeneity:
        # without it, all cores produce identical output on identical input.
        self._identity = generate_geometric_identity(
            core_id, global_id, role.value, spark_kb * 256
        )
        self.spark[:] = self._identity
        
        # Seed domain buffers with rotated identity (each domain sees
        # the core's identity from a different geometric angle)
        # Domains
        self.domains: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(DOMAIN_NAMES):
            self.domains[name] = np.zeros(domain_kb * 256, dtype=np.float32)
            # Rotate identity into each domain buffer at different phase
            domain_seed = np.roll(self._identity, idx * (spark_kb * 256 // len(DOMAIN_NAMES)))
            seed_len = min(len(domain_seed), domain_kb * 256)
            self.domains[name][:seed_len] = domain_seed[:seed_len] * 0.1  # 10% seed strength

        # Processing buffer
        self.processing = np.zeros(0, dtype=np.float32)  # Fractal: no processing buffer

        # Geometric state (auto-computed after each process_signal)
        self._last_state = {}
        self._last_input = None
        self._last_result = None
        self._last_domain = 'unknown'

        # Counters
        self.junctions_fired = 0
        self.signals_processed = 0
        self.harmonics_received = 0

    @property
    def memory_mb(self) -> float:
        total = (
            self.input_panel.memory_bytes +
            self.output_panel.memory_bytes +
            self.spark.nbytes +
            sum(d.nbytes for d in self.domains.values()) +
            self.processing.nbytes
        )
        return total / (1024 * 1024)


    def compute_state(self, input_signal: np.ndarray = None) -> dict:
        """
        Compute this core's geometric state from its last processing cycle.
        
        State is derived from the relationship between input signal,
        processed result, and field interference — not from external
        thresholds. This is the core describing its own geometry.
        
        Returns dict with: energy_ratio, resonance, preservation,
        core_asymmetry, interference, domain
        """
        sig = input_signal if input_signal is not None else self._last_input
        result = self._last_result
        
        if sig is None or result is None:
            return {}
        
        sig_flat = sig.flatten().astype(np.float64)
        res_flat = result.flatten().astype(np.float64)
        
        sig_energy = float(np.linalg.norm(sig_flat))
        res_energy = float(np.linalg.norm(res_flat))
        
        # Energy ratio: amplification or attenuation
        energy_ratio = res_energy / sig_energy if sig_energy > 1e-8 else 0.0
        
        # Signal preservation: cosine similarity input->output
        min_len = min(len(sig_flat), len(res_flat))
        dot_sig = float(np.dot(sig_flat[:min_len], res_flat[:min_len]))
        preservation = dot_sig / (sig_energy * res_energy) if sig_energy > 1e-8 and res_energy > 1e-8 else 0.0
        
        # Self-resonance: correlation between first and second half of result
        # (internal coherence of the output)
        half = len(res_flat) // 2
        if half > 10:
            dot_half = float(np.dot(res_flat[:half], res_flat[half:2*half]))
            e_h1 = float(np.linalg.norm(res_flat[:half]))
            e_h2 = float(np.linalg.norm(res_flat[half:2*half]))
            resonance = dot_half / (e_h1 * e_h2) if e_h1 > 1e-8 and e_h2 > 1e-8 else 0.0
        else:
            resonance = 1.0
        
        # Identity asymmetry: how far result diverged from identity seed
        id_flat = self._identity.flatten().astype(np.float64)
        min_id = min(len(id_flat), len(res_flat))
        dot_id = float(np.dot(id_flat[:min_id], res_flat[:min_id]))
        id_energy = float(np.linalg.norm(id_flat[:min_id]))
        core_asymmetry = 1.0 - abs(dot_id / (id_energy * res_energy)) if id_energy > 1e-8 and res_energy > 1e-8 else 0.5
        
        # Interference: field energy visible to this core
        interference_vec = self.field.read_interference(self.global_id)
        interf_energy = float(np.linalg.norm(interference_vec))
        
        state = {
            'energy': round(res_energy, 4),
            'energy_ratio': round(energy_ratio, 4),
            'resonance': round(max(min(resonance, 1.0), -1.0), 4),
            'preservation': round(max(min(preservation, 1.0), -1.0), 4),
            'core_asymmetry': round(max(core_asymmetry, 0.0), 4),
            'interference': round(interf_energy, 4),
            'domain': self._last_domain,
            'core_id': self.global_id,
            'role': self.role.value,
        }
        self._last_state = state
        return state

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Full fused processing path:
        
        1. Input panel detects signal + reads harmonic field
        2. Harmonic interference modulates routing decision
        3. Domain processes signal with interference context
        4. Output written to harmonic field (visible to all other cores)
        5. Output panel captures result
        """
        # 1. Detect with harmonics
        readings, interference = self.input_panel.detect_with_harmonics(signal)
        self.harmonics_received += 1

        # 2. Route by sensor consensus + harmonic modulation
        domain = self._route_harmonic(readings, interference)

        # 3. Process with interference context
        domain_buf = self.domains.get(domain, self.domains['reasoning'])
        result = self._harmonic_process(signal, domain_buf, readings, interference)

        # 4. Write output to harmonic field (all other cores can see this)
        self.field.write_signature(self.global_id, result)
        # 5. Entrain: nudge composite 3% toward input signal (contact-retract)
        sig_flat = signal.flatten()[:self.field.field_width]
        if len(sig_flat) == self.field.field_width:
            with self.field._lock:
                self.field.composite *= 0.97
                self.field.composite += sig_flat * 0.03

        # 5. Output panel
        self.output_panel.write_parallel(result)

        self.junctions_fired += len(readings)
        self.signals_processed += 1

        # Store for geometric state computation
        self._last_input = signal
        self._last_result = result
        self._last_domain = domain
        self.compute_state()

        return result

    def _route_harmonic(self, readings: List[SensorReading],
                        interference: np.ndarray) -> str:
        """
        Route based on BOTH direct sensor readings AND harmonic interference.
        
        The interference pattern tells this core what other cores are
        working on. This enables emergent coordination without explicit
        message passing.
        """
        if not readings:
            return 'reasoning'

        # Direct routing from sensor consensus
        best = max(readings, key=lambda r: r.confidence)
        domain_map = {
            SensorType.SPATIAL: 'reasoning',
            SensorType.CHROMATIC: 'creative',
            SensorType.STRUCTURAL: 'code',
            SensorType.RELATIONAL: 'reasoning',
            SensorType.TEMPORAL: 'system',
            SensorType.MAGNITUDE: 'factual',
            SensorType.SYMMETRY: 'reasoning',
            SensorType.BOUNDARY: 'code',
            SensorType.PATTERN: 'learning',
            SensorType.MOTION: 'system',
            SensorType.DENSITY: 'factual',
            SensorType.TOPOLOGY: 'code',
        }
        direct_domain = domain_map.get(best.sensor_type, 'reasoning')

        # Harmonic modulation: if interference is strong in a different domain,
        # this core may shift its focus to complement rather than duplicate
        interference_energy = np.abs(interference)
        if interference_energy.sum() > 0:
            # Segment interference into domain bands
            band_size = len(interference) // len(DOMAIN_NAMES)
            domain_energies = {}
            for i, name in enumerate(DOMAIN_NAMES):
                start = i * band_size
                end = start + band_size
                domain_energies[name] = float(interference_energy[start:end].sum())

            # If direct domain is already heavily covered by other cores,
            # shift to least-covered domain (complementary processing)
            if domain_energies.get(direct_domain, 0) > np.mean(list(domain_energies.values())) * 1.5:
                # Other cores already handling this — find gap
                least_covered = min(domain_energies, key=domain_energies.get)
                # Only shift if confidence allows
                if best.confidence < 0.7:
                    return least_covered

        return direct_domain

    def _harmonic_process(self, signal: np.ndarray, domain_buf: np.ndarray,
                          readings: List[SensorReading],
                          interference: np.ndarray) -> np.ndarray:
        """
        Process signal with harmonic interference context.
        
        The interference pattern modulates processing:
        - Reinforcing patterns get amplified (constructive interference)
        - Conflicting patterns get attenuated (destructive interference)
        """
        sig_flat = signal.flatten()
        buf_slice = domain_buf[:len(sig_flat)]

        # Confidence from direct sensors
        alpha = np.mean([r.confidence for r in readings]) if readings else 0.5

        # Harmonic modulation
        min_len = min(len(sig_flat), len(interference))
        interf_slice = interference[:min_len]
        sig_flat_cmp = sig_flat[:min_len]
        # Correlation between signal and interference = harmonic alignment
        if len(sig_flat) > 1 and np.std(sig_flat) > 0 and np.std(interf_slice) > 0:
            alignment = np.corrcoef(sig_flat_cmp, interf_slice)[0, 1]
            if np.isfinite(alignment):
                # Constructive: boost signal. Destructive: rely more on domain state
                harmonic_weight = 0.1 * alignment  # Small modulation
            else:
                harmonic_weight = 0.0
        else:
            harmonic_weight = 0.0

        # Fused computation
        result = (alpha + harmonic_weight) * sig_flat + (1 - alpha - harmonic_weight) * buf_slice

        # Update domain state
        domain_buf[:len(sig_flat)] = result

        return result

    def reset_counters(self):
        self.junctions_fired = 0
        self.signals_processed = 0
        self.harmonics_received = 0


# =============================================================================
# FUSED HARMONIC SUBSTRATE
# =============================================================================

class FusedHarmonicSubstrate:
    """
    Single fused 200-core substrate.
    
    All cores share one harmonic field.
    Sensor panels at every junction see the full composite.
    No layers. No message queues. No spine bus.
    Just geometry and interference.
    """

    TOTAL_CORES = 4500  # Automax target — see _detect_capacity()

    # Core allocation
    ALLOCATION = {
        CoreRole.WORKER:     2900,
        CoreRole.EXECUTIVE:   350,
        CoreRole.SPECIALIST:  700,
        CoreRole.ROUTER:      200,
        CoreRole.COUNCIL:     350,
    }

    # Processing buffer sizing per role
    PROCESSING_KB = {
        CoreRole.WORKER:     0,  # Fractal: topology IS the equation
        CoreRole.EXECUTIVE:  0,  # No processing buffer needed
        CoreRole.SPECIALIST: 0,  # Identity → phase → field
        CoreRole.ROUTER:     0,  # All routing via harmonic interference
        CoreRole.COUNCIL:    0,  # Fractal deliberation replaces tree
    }

    def __init__(self):
        self.cores: List[FusedCore] = []
        self.field: Optional[HarmonicField] = None
        self.role_index: Dict[CoreRole, List[int]] = {}  # role → global IDs
        self.created_at = datetime.now().isoformat()
        self._built = False
        self._budget_mb = 0.0
        self._autotune_active = False
        self._min_cores = 200  # Never scale below this

    # ─── AUTOMAX & AUTOTUNE ──────────────────────────────────────────


    @staticmethod
    def _autoscale_buffers(budget_mb: float, panel_overhead_kb: int = 200,
                           target_pct: float = 0.80) -> dict:
        """
        Calculate optimal core count and buffer sizes for a RAM budget.
        
        No caps. Maximizes cores at minimum viable buffers.
        target_pct: fraction of budget to actually use (0.80 = 80%).
        Leaves headroom so runtime load doesn't spike a reduction.
        
        Minimum buffers: spark=8 KB, domain=16 KB
          - 2048 float32s minimum for geometric identity differentiation
        """
        MIN_SPARK_KB = 8
        MIN_DOMAIN_KB = 16
        
        usable_mb = budget_mb * target_pct
        usable_kb = usable_mb * 1024
        
        # Per-core at minimums: spark*2(+identity) + 6*domain + panels
        min_per_core_kb = (MIN_SPARK_KB * 2 + 6 * MIN_DOMAIN_KB + panel_overhead_kb)
        
        # Max cores at minimums
        max_cores = int(usable_kb / min_per_core_kb)
        
        if max_cores < 1:
            max_cores = 1
        
        # Distribute remaining budget into buffers
        kb_per_core = usable_kb / max_cores
        usable_per_core = kb_per_core - panel_overhead_kb
        
        # 14 shares: 2 spark(+identity) + 6*2 domains
        share = usable_per_core / 14
        
        spark_kb = max(int(share), MIN_SPARK_KB)
        domain_kb = max(int(share * 2), MIN_DOMAIN_KB)
        
        # Final per-core and core count
        actual_per_core_kb = (spark_kb * 2 + 6 * domain_kb + panel_overhead_kb)
        actual_cores = int(usable_kb / actual_per_core_kb)
        actual_per_core_mb = actual_per_core_kb / 1024
        
        return {
            "cores": actual_cores,
            "spark_kb": spark_kb,
            "domain_kb": domain_kb,
            "per_core_kb": actual_per_core_kb,
            "per_core_mb": actual_per_core_mb,
            "total_mb": actual_cores * actual_per_core_mb,
            "budget_mb": budget_mb,
            "target_pct": target_pct,
            "utilization": (actual_cores * actual_per_core_mb) / budget_mb * 100,
        }

    @staticmethod
    def _available_ram_mb() -> float:
        """Read available RAM from /proc/meminfo (Linux)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024  # kB → MB
        except Exception:
            pass
        return 0.0

    @classmethod
    def _detect_capacity(cls, reserve_pct: float = 0.20) -> dict:
        """
        Detect system capacity, return scaling plan.
        
        Reserve 30% of total for OS/Ollama/ARC orchestrator.
        Fill the rest with cores at 13.1 MB each.
        """
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_mb = int(line.split()[1]) / 1024
                        break
                else:
                    total_mb = 0
        except Exception:
            total_mb = 0

        available_mb = cls._available_ram_mb()
        reserve_mb = total_mb * reserve_pct
        budget_mb = total_mb - reserve_mb
        per_core_mb = 13.1
        max_cores = int(budget_mb / per_core_mb)

        return {
            "total_ram_mb": total_mb,
            "available_ram_mb": available_mb,
            "reserve_mb": reserve_mb,
            "budget_mb": budget_mb,
            "per_core_mb": per_core_mb,
            "max_cores": max_cores,
            "target_cores": max_cores  # uncapped, autoscale decides,
        }

    def _scale_allocation(self, target_cores: int) -> dict:
        """Scale allocation proportionally to target core count."""
        # Ratios from base allocation
        ratios = {
            CoreRole.WORKER:     0.6444,  # 2900/4500
            CoreRole.EXECUTIVE:  0.0778,  # 350/4500
            CoreRole.SPECIALIST: 0.1556,  # 700/4500
            CoreRole.ROUTER:     0.0444,  # 200/4500
            CoreRole.COUNCIL:    0.0778,  # 350/4500
        }
        allocation = {}
        assigned = 0
        roles = list(ratios.keys())
        for role in roles[:-1]:
            n = int(target_cores * ratios[role])
            allocation[role] = max(n, 2)  # At least 2 per role
            assigned += allocation[role]
        # Remainder to last role (council)
        allocation[roles[-1]] = max(target_cores - assigned, 2)
        return allocation

    def autotune_check(self) -> dict:
        """
        Runtime load check. Call periodically.
        Returns status dict. If memory pressure detected,
        recommends scale-down.
        """
        available = self._available_ram_mb()
        total_substrate = self.total_memory_mb
        
        # Thresholds
        critical_mb = 4000   # < 4 GB free = critical
        warning_mb = 8000    # < 8 GB free = warning
        
        status = {
            "available_ram_mb": available,
            "substrate_mb": total_substrate,
            "core_count": len(self.cores),
            "state": "ok",
            "recommendation": None,
        }
        
        if available < critical_mb:
            # Scale down to 50% of current
            target = max(len(self.cores) // 2, self._min_cores)
            status["state"] = "critical"
            status["recommendation"] = f"Scale to {target} cores (RAM critical: {available:.0f} MB free)"
        elif available < warning_mb:
            # Scale down to 75% of current
            target = max(int(len(self.cores) * 0.75), self._min_cores)
            status["state"] = "warning"
            status["recommendation"] = f"Scale to {target} cores (RAM warning: {available:.0f} MB free)"
        
        return status

    def build(self):
        """Fabricate the fused substrate with automax."""
        cap = self._detect_capacity()
        self._budget_mb = cap["budget_mb"]
        
        # Autoscale: calculate optimal cores + buffer sizes for budget
        scale = self._autoscale_buffers(cap["budget_mb"])
        self._spark_kb = scale["spark_kb"]
        self._domain_kb = scale["domain_kb"]
        actual_total = scale["cores"]
        
        print(f"  AUTOSCALE: {scale['cores']} cores @ {scale['per_core_kb']:.0f} KB/core")
        print(f"  AUTOSCALE: spark={scale['spark_kb']} KB, domain={scale['domain_kb']} KB")
        print(f"  AUTOSCALE: {scale['total_mb']:.1f} MB / {scale['budget_mb']:.1f} MB ({scale['utilization']:.1f}%)")
        
        effective_alloc = self._scale_allocation(actual_total)
        actual_total = sum(effective_alloc.values())
        print("=" * 70)
        print("  FUSED HARMONIC SUBSTRATE - SINGLE MODEL")
        print(f"  {actual_total} Cores, Shared Harmonic Field")
        print("  Ghost in the Machine Labs")
        print("=" * 70)
        print(f"  RAM: {cap['total_ram_mb']/1024:.1f} GB total, {cap['available_ram_mb']/1024:.1f} GB available")
        print(f"  Budget: {cap['budget_mb']/1024:.1f} GB, Reserve: {cap['reserve_mb']/1024:.1f} GB")
        print(f"\n  Creating harmonic field ({actual_total} cores)...")
        self.field = HarmonicField(actual_total, field_width=1024)
        print(f"    Field memory: {self.field.memory_mb:.1f} MB")
        import time as _time
        global_id = 0
        total_target = actual_total
        fab_start = _time.time()
        
        def _progress_reporter(stop_event):
            while not stop_event.is_set():
                elapsed = _time.time() - fab_start
                n = len(self.cores)
                try:
                    with open('/proc/meminfo') as mf:
                        for line in mf:
                            if line.startswith('MemAvailable:'):
                                avail_gb = int(line.split()[1]) / 1024 / 1024
                                break
                except:
                    avail_gb = 0
                rate = n / elapsed if elapsed > 0 else 0
                eta = (total_target - n) / rate if rate > 0 else 0
                print(f"\r  [{n:,}/{total_target:,}] {n/total_target*100:.1f}% | "
                      f"{rate:.0f} cores/s | RAM avail: {avail_gb:.1f} GB | "
                      f"ETA: {eta:.0f}s    ", end='', flush=True)
                stop_event.wait(2.0)
        
        stop_progress = threading.Event()
        progress_thread = threading.Thread(target=_progress_reporter, args=(stop_progress,), daemon=True)
        progress_thread.start()
        
        NUM_FAB_THREADS = 8
        
        for role in CoreRole:
            count = effective_alloc[role]
            proc_kb = self.PROCESSING_KB[role]
            self.role_index[role] = []
            
            print(f"\n  Fabricating {role.value.upper()} cores ({count:,})...")
            
            batch_cores = [None] * count
            batch_ids = list(range(global_id, global_id + count))
            
            def _fab_chunk(start, end, _role=role, _proc_kb=proc_kb):
                for i in range(start, end):
                    batch_cores[i] = FusedCore(
                        core_id=i,
                        global_id=batch_ids[i],
                        role=_role,
                        field=self.field,
                        processing_kb=_proc_kb,
                        spark_kb=self._spark_kb,
                        domain_kb=self._domain_kb,
                    )
            
            chunk_size = max(1, count // NUM_FAB_THREADS)
            threads = []
            for t in range(NUM_FAB_THREADS):
                start = t * chunk_size
                end = count if t == NUM_FAB_THREADS - 1 else min(start + chunk_size, count)
                if start >= count:
                    break
                th = threading.Thread(target=_fab_chunk, args=(start, end))
                threads.append(th)
                th.start()
            
            for th in threads:
                th.join()
            
            for i, core in enumerate(batch_cores):
                if core is not None:
                    self.cores.append(core)
                    self.role_index[role].append(batch_ids[i])
            
            global_id += count
            
            sample = batch_cores[0]
            total_mb = sample.memory_mb * count
            print(f"\n    {count:,} cores x {sample.memory_mb:.1f} MB = {total_mb:.1f} MB")
        
        stop_progress.set()
        progress_thread.join(timeout=3)
        elapsed = _time.time() - fab_start
        print(f"\n\n  Fabrication time: {elapsed:.1f}s ({len(self.cores)/elapsed:.0f} cores/s)")

        self._built = True

        print(f"\n{'='*70}")
        print(f"  FABRICATION COMPLETE")
        print(f"{'='*70}")
        self.print_summary()

    def print_summary(self):
        print(f"\n  {'Role':<15s} {'Cores':>6s} {'Per Core':>10s} {'Total MB':>10s} {'Total GB':>10s}")
        print(f"  {'-'*55}")

        total_mb = 0
        total_cores = 0
        for role in CoreRole:
            ids = self.role_index[role]
            count = len(ids)
            per_core = self.cores[ids[0]].memory_mb
            role_total = per_core * count
            total_mb += role_total
            total_cores += count
            print(f"  {role.value:<15s} {count:>6d} {per_core:>8.1f} MB {role_total:>10.1f} {role_total/1024:>10.2f}")

        field_mb = self.field.memory_mb if self.field else 0
        total_mb += field_mb
        print(f"  {'harmonic_field':<15s} {'—':>6s} {'':>10s} {field_mb:>10.1f} {field_mb/1024:>10.2f}")
        print(f"  {'-'*55}")
        print(f"  {'TOTAL':<15s} {total_cores:>6d} {'':>10s} {total_mb:>10.1f} {total_mb/1024:>10.2f}")
        print()
        print(f"  Budget (70%% available):   {self._budget_mb:,.0f} MB")
        print(f"  Model total:              {total_mb:,.0f} MB ({total_mb/1024:.2f} GB)")
        print(f"  Growth reserve:           {self._budget_mb - total_mb:,.0f} MB ({(self._budget_mb - total_mb)/1024:.1f} GB)")
        print(f"  Utilization of budget:    {total_mb/self._budget_mb*100:.1f}%")
        print()
        print(f"  vs. Layered model:        17,726 MB (17.31 GB)")
        print(f"  vs. Old Harmonic Stack:   ~89,000 MB (~87 GB via Ollama)")

    def get_cores_by_role(self, role: CoreRole) -> List[FusedCore]:
        return [self.cores[i] for i in self.role_index[role]]

    @property
    def total_memory_mb(self) -> float:
        field = self.field.memory_mb if self.field else 0
        return sum(c.memory_mb for c in self.cores) + field

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024

    def process_signal(self, signal: np.ndarray, target_role: CoreRole = None) -> np.ndarray:
        """
        Process signal through the fused substrate.
        
        If target_role specified, routes to that role's cores.
        Otherwise, router cores detect and dispatch.
        
        All cores see the harmonic field regardless.
        """
        if target_role:
            cores = self.get_cores_by_role(target_role)
        else:
            # Router cores detect and classify first
            routers = self.get_cores_by_role(CoreRole.ROUTER)
            cores = self.get_cores_by_role(CoreRole.WORKER)

            # Sample routers - 16 max, evenly spaced
            max_routers = min(16, len(routers))
            step = max(1, len(routers) // max_routers)
            sampled = routers[::step][:max_routers]
            for r in sampled:
                r.process_signal(signal)

        # Fire worker cluster - sqrt(N) cores, evenly spaced
        import math
        cluster_size = max(1, min(int(math.sqrt(len(cores))), 256))
        step = max(1, len(cores) // cluster_size)
        cluster = cores[::step][:cluster_size]
        results = []
        for core in cluster:
            result = core.process_signal(signal)
            results.append(result)

        # Decay field (older signals fade)
        self.field.decay(0.95)

        # Collect geometric states from all cores that fired
        self._last_fired_states = {}
        for i, core in enumerate(cores):
            if core._last_state:
                self._last_fired_states[f"{core.role.value}_{core.global_id}"] = core._last_state
        # Router states too
        if not target_role:
            for r in routers:
                if r._last_state:
                    self._last_fired_states[f"router_{r.global_id}"] = r._last_state

        if results:
            import numpy as np
            return np.mean(np.stack(results), axis=0)
        return signal

    def process_parallel(self, signals: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch of signals across worker cores."""
        workers = self.get_cores_by_role(CoreRole.WORKER)
        results = []
        for i, signal in enumerate(signals):
            core = workers[i % len(workers)]
            result = core.process_signal(signal)
            results.append(result)
        # Decay
        self.field.decay(0.95)
        return results

    def benchmark(self, iterations: int = 5000, signal_size: int = 100) -> dict:
        """Benchmark the fused substrate."""
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: FUSED 200-CORE SUBSTRATE")
        print(f"{'='*70}")

        results = {}
        signal = np.random.randn(signal_size).astype(np.float32)

        # Per-role single-core throughput
        for role in CoreRole:
            cores = self.get_cores_by_role(role)
            core = cores[0]
            core.reset_counters()

            start = time.perf_counter()
            for _ in range(iterations):
                core.process_signal(signal)
            elapsed = time.perf_counter() - start

            rate = iterations / elapsed
            jps = core.junctions_fired / elapsed
            hps = core.harmonics_received / elapsed

            results[role.value] = {
                'signals_per_sec': rate,
                'junctions_per_sec': jps,
                'harmonics_per_sec': hps,
                'cores': len(cores),
                'memory_mb': core.memory_mb * len(cores),
            }
            print(f"\n  {role.value.upper()} ({len(cores)} cores):")
            print(f"    {rate:,.0f} sig/s | {jps:,.0f} junctions/s | {hps:,.0f} harmonics/s")

            core.reset_counters()

        # Harmonic field activity
        print(f"\n  HARMONIC FIELD:")
        print(f"    Active cores: {self.field.active_cores}/{len(self.cores)}")
        print(f"    Interference events: {self.field.interference_events:,}")
        print(f"    Field memory: {self.field.memory_mb:.1f} MB")
        results['field'] = {
            'active_cores': self.field.active_cores,
            'interference_events': self.field.interference_events,
            'memory_mb': self.field.memory_mb,
        }

        # Sensor detection rate
        start = time.perf_counter()
        for _ in range(iterations):
            self.cores[0].input_panel.detect(signal)
        elapsed = time.perf_counter() - start
        det_rate = iterations / elapsed
        results['sensor_detection_per_sec'] = det_rate
        print(f"\n  SENSOR DETECTION:")
        print(f"    {det_rate:,.0f} full-panel detections/sec")

        # Harmonic convergence test:
        # Fire same signal through multiple cores, measure field convergence
        print(f"\n  HARMONIC CONVERGENCE TEST:")
        test_signal = np.random.randn(signal_size).astype(np.float32)
        convergence_readings = []
        for n_cores in [1, 4, 16, 64, 128]:
            # Reset field
            self.field.field[:] = 0
            self.field.activity[:] = 0
            self.field.interference_events = 0

            workers = self.get_cores_by_role(CoreRole.WORKER)
            for i in range(min(n_cores, len(workers))):
                workers[i].process_signal(test_signal)

            composite = self.field.read_composite()
            energy = float(np.linalg.norm(composite))
            convergence_readings.append((n_cores, energy, self.field.interference_events))
            print(f"    {n_cores:>3d} cores → field energy: {energy:.4f}, "
                  f"interference events: {self.field.interference_events}")

        results['convergence'] = convergence_readings

        return results

    def status(self) -> dict:
        report = {
            'name': 'Fused Harmonic Substrate',
            'version': '1.0.0',
            'created_at': self.created_at,
            'total_cores': len(self.cores),
            'total_memory_gb': round(self.total_memory_gb, 2),
            'budget_gb': 46.0,
            'reserve_gb': round(46.0 - self.total_memory_gb, 2),
            'roles': {},
            'harmonic_field': {
                'active_cores': self.field.active_cores if self.field else 0,
                'interference_events': self.field.interference_events if self.field else 0,
                'memory_mb': round(self.field.memory_mb, 1) if self.field else 0,
            },
        }
        for role in CoreRole:
            ids = self.role_index[role]
            cores = [self.cores[i] for i in ids]
            report['roles'][role.value] = {
                'cores': len(ids),
                'total_memory_mb': round(sum(c.memory_mb for c in cores), 1),
                'total_junctions': sum(c.junctions_fired for c in cores),
                'total_signals': sum(c.signals_processed for c in cores),
                'total_harmonics': sum(c.harmonics_received for c in cores),
            }
        return report

    def export_config(self, path: str):
        config = {
            'name': 'Fused Harmonic Substrate',
            'version': '1.0.0',
            'author': 'Ghost in the Machine Labs',
            'created_at': self.created_at,
            'architecture': 'fused_single_model',
            'total_cores': len(self.cores),
            'total_memory_gb': round(self.total_memory_gb, 2),
            'budget_gb': 46.0,
            'harmonic_field_width': 1024,
            'roles': {},
            'specialist_domains': SPECIALIST_DOMAINS,
            'council_personas': COUNCIL_PERSONAS,
        }
        for role in CoreRole:
            config['roles'][role.value] = {
                'cores': self.ALLOCATION[role],
                'processing_kb': self.PROCESSING_KB[role],
            }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n  Config exported: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    substrate = FusedHarmonicSubstrate()
    substrate.build()
    substrate.export_config('/tmp/fused_harmonic_config.json')
    results = substrate.benchmark(iterations=5000, signal_size=100)

    print(f"\n{'='*70}")
    print(f"  STATUS")
    print(f"{'='*70}")
    status = substrate.status()
    print(json.dumps(status, indent=2))

    print(f"\n  Model built. NOT deployed.")
    print(f"  To deploy: copy to target hardware and activate.")


if __name__ == "__main__":
    main()
