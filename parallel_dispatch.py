"""
Parallel Core Dispatch — Shared Memory Multiprocessing
======================================================
Ghost in the Machine Labs

Bypasses CPython GIL by running core clusters in separate processes
with the harmonic field backed by multiprocessing.shared_memory.

Architecture:
  - HarmonicField.field (num_cores x field_width) lives in shared memory
  - HarmonicField.activity (num_cores) lives in shared memory
  - Worker processes each own a slice of FusedCore objects
  - Signal dispatch: main process sends signal via pipe,
    workers fire their cores against shared field, reply done
  - Results collected via shared result buffer

Data flow per core.process_signal():
  READ:  field.field[*]         (shared, read all rows for interference)
  WRITE: field.field[core_id]   (shared, write own row only)
  WRITE: field.activity[core_id](shared, write own flag only)
  R/W:   core.domains[*]        (per-core, private to worker process)
  R/W:   core.spark             (per-core, private to worker process)
  
No write conflicts: each core only writes its own row.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from typing import List, Optional, Dict


class SharedField:
    """
    Harmonic field backed by shared memory.
    
    The field array and activity mask live in OS-level shared memory,
    visible to all worker processes without serialization or copying.
    
    Numpy arrays are attached to the shared memory blocks via
    np.ndarray(buffer=shm.buf) — zero-copy access.
    """
    
    def __init__(self, num_cores: int, field_width: int = 1024, create: bool = True):
        self.num_cores = num_cores
        self.field_width = field_width
        
        field_bytes = num_cores * field_width * 4  # float32
        activity_bytes = num_cores * 4             # float32
        result_bytes = field_width * 4             # float32 per core result slot
        
        if create:
            # Create shared memory blocks
            self._shm_field = shared_memory.SharedMemory(
                create=True, size=field_bytes, name='hf_field')
            self._shm_activity = shared_memory.SharedMemory(
                create=True, size=activity_bytes, name='hf_activity')
            
            # Attach numpy arrays
            self.field = np.ndarray(
                (num_cores, field_width), dtype=np.float32,
                buffer=self._shm_field.buf)
            self.activity = np.ndarray(
                num_cores, dtype=np.float32,
                buffer=self._shm_activity.buf)
            
            # Zero-initialize
            self.field[:] = 0.0
            self.activity[:] = 0.0
        else:
            # Attach to existing shared memory
            self._shm_field = shared_memory.SharedMemory(name='hf_field')
            self._shm_activity = shared_memory.SharedMemory(name='hf_activity')
            
            self.field = np.ndarray(
                (num_cores, field_width), dtype=np.float32,
                buffer=self._shm_field.buf)
            self.activity = np.ndarray(
                num_cores, dtype=np.float32,
                buffer=self._shm_activity.buf)
        
        # Shard metadata (local to each process, computed from shared data)
        self._num_shards = max(1, num_cores // 64)
        self._shard_size = max(1, (num_cores + self._num_shards - 1) // self._num_shards)
        
        self.interference_events = 0
        self._write_count = 0
    
    def write_signature(self, core_id: int, signature: np.ndarray):
        """Lock-free write to own row in shared memory."""
        sig = signature.flatten()[:self.field_width]
        n = len(sig)
        self.field[core_id, :n] = sig
        if n < self.field_width:
            self.field[core_id, n:] = 0.0
        self.activity[core_id] = 1.0
        self._write_count += 1
    
    def read_interference(self, core_id: int) -> np.ndarray:
        """Read composite minus self. Computed from shared memory."""
        active_mask = self.activity > 0
        total = int(active_mask.sum())
        if total <= 1:
            return np.zeros(self.field_width, dtype=np.float32)
        
        if self.activity[core_id] > 0:
            global_sum = self.field[active_mask].sum(axis=0)
            interference = (global_sum - self.field[core_id]) / (total - 1)
            self.interference_events += 1
            return interference.astype(np.float32)
        
        composite = self.field[active_mask].mean(axis=0)
        self.interference_events += 1
        return composite
    
    def read_composite(self) -> np.ndarray:
        active_mask = self.activity > 0
        if active_mask.any():
            return self.field[active_mask].mean(axis=0)
        return np.zeros(self.field_width, dtype=np.float32)
    
    def read_neighborhood(self, core_id: int, radius: int = 8) -> np.ndarray:
        start = max(0, core_id - radius)
        end = min(self.num_cores, core_id + radius + 1)
        local_activity = self.activity[start:end]
        active = local_activity > 0
        if active.any():
            return self.field[start:end][active].mean(axis=0)
        return np.zeros(self.field_width, dtype=np.float32)
    
    def entrain(self, core_id: int, signal: np.ndarray, rate: float = 0.03):
        sig_flat = signal.flatten()[:self.field_width]
        if len(sig_flat) == self.field_width:
            self.field[core_id] *= (1.0 - rate)
            self.field[core_id] += sig_flat * rate
    
    def apply_governance_phase(self, core_id: int, phase: float):
        if 0 <= core_id < self.num_cores:
            self.field[core_id, 0] = phase
    
    def decay(self, rate: float = 0.95):
        self.activity *= rate
    
    @property
    def memory_mb(self) -> float:
        return (self.field.nbytes + self.activity.nbytes) / (1024 * 1024)
    
    @property
    def active_cores(self) -> int:
        return int((self.activity > 0.01).sum())
    
    def cleanup(self):
        """Unlink shared memory (call from creating process only)."""
        try:
            self._shm_field.close()
            self._shm_field.unlink()
        except:
            pass
        try:
            self._shm_activity.close()
            self._shm_activity.unlink()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Worker Process
# ═══════════════════════════════════════════════════════════════════════

def _worker_loop(worker_id: int, core_specs: list, num_cores: int,
                 field_width: int, cmd_pipe, result_shm_name: str,
                 spark_kb: int, domain_kb: int):
    """
    Worker process main loop.
    
    Each worker owns a set of FusedCore objects and processes signals
    against the shared harmonic field.
    
    Protocol:
      cmd_pipe receives: ('fire', signal_bytes, core_indices)
                         ('quit',)
      result written to shared memory at result_shm[core_idx * field_width]
    """
    import sys
    import os
    
    # Import substrate classes in worker
    # These need to be importable from the worker
    sys.path.insert(0, os.path.expanduser('~/sparky'))
    from fused_harmonic_substrate import (
        FusedCore, CoreRole, HarmonicField, JunctionPanel,
        generate_geometric_identity, SensorType, SensorReading,
        DOMAIN_NAMES
    )
    
    # Attach to shared field
    field = SharedField(num_cores, field_width, create=False)
    
    # Attach to result buffer
    result_shm = shared_memory.SharedMemory(name=result_shm_name)
    result_buf = np.ndarray(
        (num_cores, field_width), dtype=np.float32,
        buffer=result_shm.buf)
    
    # Build cores (each worker builds only its assigned cores)
    # core_specs: list of (core_id, global_id, role_value)
    cores = {}
    for core_id, global_id, role_value in core_specs:
        role = CoreRole(role_value)
        core = FusedCore(core_id, global_id, role, field,
                        spark_kb=spark_kb, domain_kb=domain_kb)
        cores[global_id] = core
    
    # Main loop
    while True:
        try:
            msg = cmd_pipe.recv()
        except EOFError:
            break
        
        if msg[0] == 'quit':
            break
        
        if msg[0] == 'fire':
            signal = np.frombuffer(msg[1], dtype=np.float32).copy()
            core_indices = msg[2]
            
            for gid in core_indices:
                if gid in cores:
                    result = cores[gid].process_signal(signal)
                    # Write result to shared buffer
                    n = min(len(result.flatten()), field_width)
                    result_buf[gid, :n] = result.flatten()[:n]
            
            cmd_pipe.send(('done', core_indices))
    
    result_shm.close()
    field._shm_field.close()
    field._shm_activity.close()


class ParallelDispatcher:
    """
    Manages worker processes for parallel core firing.
    
    Usage:
        dispatcher = ParallelDispatcher(substrate, num_workers=8)
        dispatcher.start()
        results = dispatcher.fire(signal, core_indices)
        dispatcher.stop()
    """
    
    def __init__(self, num_cores: int, field_width: int = 1024,
                 num_workers: int = None, spark_kb: int = 8,
                 domain_kb: int = 16):
        self.num_cores = num_cores
        self.field_width = field_width
        self.spark_kb = spark_kb
        self.domain_kb = domain_kb
        
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 16)
        self.num_workers = num_workers
        
        self.shared_field = None
        self._result_shm = None
        self.result_buf = None
        self._workers = []
        self._pipes = []
        self._started = False
    
    def start(self, core_assignments: Dict[int, List] = None):
        """
        Start worker processes.
        
        core_assignments: {worker_id: [(core_id, global_id, role_value), ...]}
        If None, cores are distributed round-robin across workers.
        """
        # Cleanup any existing shared memory with these names
        for name in ['hf_field', 'hf_activity', 'hf_results']:
            try:
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except:
                pass
        
        # Create shared field
        self.shared_field = SharedField(
            self.num_cores, self.field_width, create=True)
        
        # Create result buffer in shared memory
        result_bytes = self.num_cores * self.field_width * 4
        self._result_shm = shared_memory.SharedMemory(
            create=True, size=result_bytes, name='hf_results')
        self.result_buf = np.ndarray(
            (self.num_cores, self.field_width), dtype=np.float32,
            buffer=self._result_shm.buf)
        self.result_buf[:] = 0.0
        
        if core_assignments is None:
            return  # No workers to start yet
        
        # Start worker processes
        for wid in range(self.num_workers):
            specs = core_assignments.get(wid, [])
            if not specs:
                continue
            
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_loop,
                args=(wid, specs, self.num_cores, self.field_width,
                      child_conn, 'hf_results', self.spark_kb,
                      self.domain_kb),
                daemon=True
            )
            p.start()
            self._workers.append(p)
            self._pipes.append(parent_conn)
        
        self._started = True
    
    def fire(self, signal: np.ndarray, 
             core_indices_per_worker: List[List[int]]) -> np.ndarray:
        """
        Fire signal through cores in parallel.
        
        core_indices_per_worker: list aligned with self._pipes,
          each element is list of global_ids for that worker to fire.
        
        Returns: averaged result array.
        """
        signal_bytes = signal.astype(np.float32).tobytes()
        
        # Dispatch to all workers
        active = []
        for i, (pipe, indices) in enumerate(
                zip(self._pipes, core_indices_per_worker)):
            if indices:
                pipe.send(('fire', signal_bytes, indices))
                active.append((i, indices))
        
        # Collect completions
        all_indices = []
        for i, indices in active:
            msg = self._pipes[i].recv()
            assert msg[0] == 'done'
            all_indices.extend(indices)
        
        # Average results from shared buffer
        if all_indices:
            results = self.result_buf[all_indices]
            return results.mean(axis=0)
        return signal.astype(np.float32)
    
    def stop(self):
        """Shutdown all workers and clean up shared memory."""
        for pipe in self._pipes:
            try:
                pipe.send(('quit',))
            except:
                pass
        
        for p in self._workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        
        self._workers.clear()
        self._pipes.clear()
        
        if self._result_shm:
            try:
                self._result_shm.close()
                self._result_shm.unlink()
            except:
                pass
        
        if self.shared_field:
            self.shared_field.cleanup()
        
        self._started = False


# ═══════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════

def create_parallel_substrate(substrate, num_workers: int = None):
    """
    Wrap an existing FusedHarmonicSubstrate with parallel dispatch.
    
    Returns (dispatcher, fire_fn) where fire_fn replaces
    substrate.process_signal for parallel execution.
    """
    num_cores = len(substrate.cores)
    field_width = substrate.field.field_width
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    
    # Build core assignments: round-robin across workers
    from fused_harmonic_substrate import CoreRole
    workers_cores = {i: [] for i in range(num_workers)}
    
    for core in substrate.cores:
        wid = core.global_id % num_workers
        workers_cores[wid].append(
            (core.core_id, core.global_id, core.role.value))
    
    # Create dispatcher
    dispatcher = ParallelDispatcher(
        num_cores=num_cores,
        field_width=field_width,
        num_workers=num_workers,
        spark_kb=substrate.cores[0]._spark_kb if substrate.cores else 8,
        domain_kb=substrate.cores[0]._domain_kb if substrate.cores else 16,
    )
    dispatcher.start(core_assignments=workers_cores)
    
    # Copy field state to shared memory
    dispatcher.shared_field.field[:] = substrate.field.field
    dispatcher.shared_field.activity[:] = substrate.field.activity
    
    return dispatcher


# ═══════════════════════════════════════════════════════════════════════
# Standalone benchmark
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.expanduser('~/sparky'))
    
    from fused_harmonic_substrate import (
        FusedHarmonicSubstrate, CoreRole, FusedCore
    )
    
    num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(mp.cpu_count(), 8)
    
    print(f'Building substrate: {num_cores} cores, {num_workers} workers')
    print(f'CPU cores available: {mp.cpu_count()}')
    
    # Build substrate
    s = FusedHarmonicSubstrate()
    s.TOTAL_CORES = num_cores
    cap = {'total_ram_mb': 4000, 'available_ram_mb': 3200, 'reserve_mb': 0,
           'budget_mb': 3200, 'per_core_mb': 13.1, 'max_cores': num_cores,
           'target_cores': num_cores}
    s._detect_capacity = classmethod(lambda cls, **kw: cap).__get__(type(s))
    s._autoscale_buffers = staticmethod(lambda bm, **kw: {
        'cores': num_cores, 'spark_kb': 8, 'domain_kb': 16,
        'per_core_kb': 312, 'per_core_mb': 0.305,
        'total_mb': num_cores * 0.305, 'budget_mb': bm,
        'target_pct': 0.8, 'utilization': 80.0})
    alloc = s._scale_allocation(num_cores)
    s.ALLOCATION = alloc
    s.build()
    
    signal = np.random.randn(1024).astype(np.float32)
    
    # ── Sequential baseline ──────────────────────────────────────
    print('\n--- Sequential baseline ---')
    import math
    workers = s.get_cores_by_role(CoreRole.WORKER)
    for n_cores in [1, 4, 14, 64, 128]:
        cluster_size = min(n_cores, len(workers))
        step = max(1, len(workers) // cluster_size)
        cluster = workers[::step][:cluster_size]
        
        iters = max(50, 500 // cluster_size)
        t0 = time.perf_counter()
        for _ in range(iters):
            for core in cluster:
                core.process_signal(signal)
        dt = time.perf_counter() - t0
        fires = iters * cluster_size
        fps = fires / dt
        print(f'  {cluster_size:>4d} cores: {fps:>10,.0f} fires/s ({fps/cluster_size:,.0f}/core)')
    
    # ── Parallel dispatch ────────────────────────────────────────
    print(f'\n--- Parallel dispatch ({num_workers} workers) ---')
    
    dispatcher = create_parallel_substrate(s, num_workers=num_workers)
    
    # Build dispatch lists per worker for different cluster sizes
    for n_cores in [1, 4, 14, 64, 128]:
        cluster_size = min(n_cores, len(workers))
        step = max(1, len(workers) // cluster_size)
        cluster = workers[::step][:cluster_size]
        gids = [c.global_id for c in cluster]
        
        # Split across workers
        per_worker = {i: [] for i in range(num_workers)}
        for gid in gids:
            wid = gid % num_workers
            per_worker[wid].append(gid)
        indices_per_worker = [per_worker[i] for i in range(len(dispatcher._pipes))]
        
        # Warmup
        for _ in range(3):
            dispatcher.fire(signal, indices_per_worker)
        
        iters = max(50, 500 // cluster_size)
        t0 = time.perf_counter()
        for _ in range(iters):
            dispatcher.fire(signal, indices_per_worker)
        dt = time.perf_counter() - t0
        fires = iters * cluster_size
        fps = fires / dt
        print(f'  {cluster_size:>4d} cores: {fps:>10,.0f} fires/s ({fps/cluster_size:,.0f}/core)')
    
    dispatcher.stop()
    print('\nDone.')
