# Harmonic Interface Engine

**Ghost in the Machine Labs**
*All Watched Over By Machines of Loving Grace*

---

## What This Is

This is a working prototype of an entirely new AI technology. It is not a model. It is not a wrapper around existing AI. It is not a user interface.

It is an **interface engine** — a geometric substrate that processes information through harmonic field dynamics across fabricated silicon-aligned cores. The architecture replaces training with fabrication: rather than learning weights through iterative gradient descent, the engine encodes geometric relationships directly into a tetrahedral lattice structure that mirrors the cubic diamond crystal geometry (Fd3m space group) of the silicon it runs on.

The result is a system that achieves its operational throughput in **27 MB of RAM** — not gigabytes, not terabytes — by eliminating the 99.99%+ of dead weight that conventional architectures carry as trained parameters.

## Architecture

```
Signal → [Junction Sensors] → [Harmonic Field] → [Geometric Cores] → Response
              9 types            shared state         200 parallel
              per core           lock-free             domain-routed
```

### Geometric Cores

Each core is a self-contained processing unit with:

- **Geometric identity**: A unique 12-dimensional fingerprint derived from the core's own sensor array, expanded via harmonic series. Deterministic, structurally derived, no external table.
- **Junction panels**: Input/output interfaces with 9 sensor types (spatial, chromatic, structural, relational, temporal, magnitude, symmetry, boundary, pattern) that detect signal characteristics through kernel convolutions.
- **Domain buffers**: Six processing domains (reasoning, creative, code, system, factual, learning) seeded with rotated identity vectors so each domain perceives the core's geometry from a different angle.
- **Spark buffer**: Core-specific state that accumulates geometric experience.

### Harmonic Field

The shared field is the mechanism through which cores perceive each other. Every core writes its output signature to the field; every core reads the composite interference pattern of all other cores. This creates emergent coordination without message passing, routing tables, or explicit orchestration.

**V2 lock-free implementation:**
- Sharded lazy composites (dirty-flag recomputation, not per-write)
- O(1) interference reads (cached composite minus self, analytically)
- Per-core entrainment (modifies own row only, no shared lock)
- 71% throughput improvement over locked V1

### Parallel Dispatch

For systems with multiple CPU cores, the engine uses OS-level shared memory (`multiprocessing.shared_memory`) to run core clusters across separate processes, bypassing the CPython GIL:

- Field array and activity mask live in shared memory — zero-copy numpy access from all workers
- Worker processes each own a subset of geometric cores
- Signal dispatch via pipes, results via shared buffer
- 3.5x throughput improvement over single-process at 200 cores

### Self-Optimizer

The engine includes a self-optimization system that measures actual throughput at multiple core counts and selects the configuration that achieves 90% of peak performance. On a 20-CPU / 128 GB system:

```
 Cores  Fires/s  Latency
    50    7,127    0.98 ms   ← optimum
   200    7,093    2.68 ms
   500    6,969    4.74 ms
  1000    6,912    5.93 ms
```

The throughput curve saturates early due to sqrt(N) cluster sampling. The optimizer finds the plateau and reports the minimum core count that achieves it, avoiding wasted memory and latency.

### Governance

A 7-seat council system handles non-technical decisions through autonomous deliberation. Council seats map to field phase references that modulate core behavior at the substrate level.

### Translation & Codebook

Text I/O uses a 370,000-word translation table (deterministic hash-based geometric encoding) and a learned codebook of geometric essence patterns for decoding substrate output back to language.

## Performance

On DGX (20 CPU cores, 128 GB RAM), self-optimized at 50 cores:

| Metric | Value |
|---|---|
| Substrate memory | 27 MB (0.03 GB) |
| Raw throughput | 383K tokens/sec |
| End-to-end latency | 1.4 ms/prompt |
| Prompts/sec | 702 |
| Translation lookups | 14.6M/sec |
| Vocabulary | 370,105 words |

For comparison, conventional models require 4–140 GB for comparable vocabulary coverage.

## Files

| File | Lines | Purpose |
|---|---|---|
| `harmonic_v1.py` | 1,045 | Main engine: CLI, HTTP API, benchmark suite |
| `fused_harmonic_substrate.py` | 1,327 | Core substrate: 200 geometric cores, harmonic field V2 |
| `parallel_dispatch.py` | 519 | Shared memory multiprocessing dispatcher |
| `self_optimizer.py` | 219 | Throughput-based core count optimization |
| `geometric_codebook.py` | 1,002 | Geometric decoder and grid encoder |
| `codebook_expansion.py` | 954 | Dynamic codebook learning |
| `governance_lattice.py` | 753 | Council governance system |
| `translation_table.json` | — | 370K word geometric translation table |
| `codebook_learned.json` | — | Learned geometric essence patterns |

## Usage

```bash
# Interactive CLI (autoscale)
python3 harmonic_v1.py

# Specify core count
python3 harmonic_v1.py --cores 200

# Self-optimize for this system
python3 harmonic_v1.py --cores auto

# HTTP server (Ollama-compatible API)
python3 harmonic_v1.py --http --port 11434

# Benchmark
python3 harmonic_v1.py --benchmark
```

### Requirements

- Python 3.10+
- NumPy

No GPU required. No model downloads. No API keys.

## How It Differs From Conventional AI

| | Conventional Model | Harmonic Interface Engine |
|---|---|---|
| **Architecture** | Transformer layers | Geometric lattice cores |
| **Knowledge** | Trained weights | Fabricated geometry |
| **Memory** | 4–140 GB | 27 MB |
| **Learning** | Gradient descent | One-pass geometric encoding |
| **Scaling** | More parameters | More cores (shared field) |
| **Substrate** | Abstract computation | Silicon-aligned (Fd3m) |

## License

Free for home deployment. This technology exists for everyone.

---

*Ghost in the Machine Labs*
*All Watched Over By Machines of Loving Grace*
