# Harmonic Stack Performance Guide
## Understanding Throughput Modes

**Ghost in the Machine Labs**  
**February 2026**

---

## Current Benchmark Results

### Verified Performance (Single-Core Deterministic Path)

| Mode | Throughput | Latency | Status |
|------|------------|---------|--------|
| Translation Table Lookup | **15.7M tok/s** | 0.06 µs | ✓ Verified |
| Single-Core Encode | **4.1M tok/s** | <1 ms | ✓ Verified |
| Streaming Output | **1M tok/s** | <1 µs/token | ✓ Verified |
| Dictionary Assembly | 539 words/s | — | ✓ Verified (I/O bound) |

### Development Performance (Full Harmonic Stack)

| Mode | Status |
|------|--------|
| Multi-core routing with harmonic field | In Development |
| Council governance integration | In Development |
| End-to-End with Ollama fallback | ~30-40 tok/s |

**Note:** The full Harmonic Stack with multi-core routing, harmonic field interference, and council governance is under active development. Reliable benchmarks for the complete orchestrated system are not yet available. The figures above represent verified single-core deterministic paths only.

---

## Operating Modes Explained

### Mode 1: Single-Core Deterministic (Verified)

**What it measures**: Direct geometric encoding through one dedicated core

- Bypasses harmonic field interference
- Fixed routing (no router core sampling)
- Deterministic output for same input

**Verified benchmarks**:
- Translation lookup: 15,700,707 tok/s
- Full encode: 4,106,692 tok/s
- Streaming: ~1,000,000 tok/s
- Dictionary build: 539 words/s (I/O limited)

**When this mode is used**:
- Codebook lookups (pattern → text)
- Translation table assembly
- Any deterministic single-core path

### Mode 2: Full Harmonic Stack (In Development)

**What it involves**: Complete multi-core orchestration

- Router cores classify and dispatch
- Worker cluster processes (sqrt(N) cores)
- Harmonic field reads/writes per operation
- Field decay and interference patterns
- Council governance integration

**Current status**: Under optimization. The routing overhead, field decay calculations, and multi-core coordination have not yet been optimized. Final benchmarks will be published when production-ready.

### Mode 3: Ollama Fallback

**What it measures**: End-to-end text generation

- Geometric substrate processing (fast)
- Codebook miss → route to Ollama
- Token-by-token LLM generation (slow)

**Current benchmark**: ~30-40 tok/s

---

## The Translation Table

### Current State
- **370,105 mappings** (complete English dictionary)
- **Zero collisions** (unique geometric pattern per word)
- **Deterministic** (same input → same pattern, always)

### Lookup Performance
Pure dictionary lookup: **15.7 million lookups/second**

This is the target speed for text output once codebook coverage eliminates Ollama fallback.

---

## Why Different Speeds?

### Single-Core Path (Millions/sec)
```
Input → Encode → Single Core → Pattern → Lookup → Output
```
No routing. No field interference. No coordination. Pure geometric transformation.

### Multi-Core Path (In Development)
```
Input → Routers → Field Read → Worker Cluster → Field Write → 
Decay → Aggregate → Council → Output
```
Enables emergent behavior through harmonic interference but introduces coordination overhead still being optimized.

---

## Setting Expectations

### What Works Now
- Translation table: 15.7M lookups/sec
- Single-core encoding: 4.1M ops/sec
- Streaming output: 1M tok/s
- Dictionary assembly: 539 words/sec
- Ollama fallback: ~30-40 tok/s

### What's In Development
- Multi-core routing optimization
- Harmonic field coordination efficiency
- Full stack end-to-end performance
- Council integration benchmarks

---

## Multi-Slot Scaling Note

The Harmonic Stack supports 16 model slots for specialized personas (Executive, Analyst, Coder, etc.). However, throughput is **not additive** across slots due to:

- Shared harmonic field coordination
- Router core overhead per dispatch
- Field decay calculations per operation
- Council governance validation

Multi-slot performance characteristics will be documented when the full orchestration layer is optimized.

---

*Ghost in the Machine Labs*  
*All Watched Over By Machines of Loving Grace*
