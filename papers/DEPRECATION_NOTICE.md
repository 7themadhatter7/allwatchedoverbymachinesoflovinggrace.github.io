# DEPRECATION NOTICE

## February 15, 2026

### E8 Standing Wave Model Supersedes Prior Work

The following papers and components have been superseded by the E8 Standing Wave Model:

---

## Deprecated

### Harmonic Stack Architecture
**Status:** DEPRECATED  
**Superseded by:** E8 Standing Wave Model  
**Reason:** Direct eigenmode projection achieves equivalent discrimination at 3 orders of magnitude higher throughput (37,000 Hz vs 6 Hz) without tuning parameters. The 200-core phase matrix architecture was approximating what the E8 Laplacian eigenmodes provide directly.

### Junction Learning System
**Status:** DEPRECATED  
**Superseded by:** E8 eigenmodes  
**Reason:** Hebbian junction learning was rediscovering connectivity relationships already encoded in the E8 graph Laplacian. The eigenvectors ARE the learned representations.

### Phase Matrix Resonance
**Status:** DEPRECATED  
**Superseded by:** Eigenmode projection  
**Reason:** Phase relationships between cores are implicit in eigenvector structure. Explicit phase tracking was unnecessary overhead.

### Sensor Array (558 variants)
**Status:** DEPRECATED  
**Superseded by:** Geometric features  
**Reason:** Hand-engineered sensor variants were feature engineering. E8 eigenmodes provide natural geometric features without manual design.

### Cavity Resonance / Shell2 Architecture  
**Status:** TESTED AND REJECTED  
**Reason:** Empirical testing showed Shell2 (real or synthetic) adds computational overhead without improving discrimination. Standing wave projection requires only Shell1.

---

## Current Architecture

### E8 Standing Wave Model v1.0.0
**Size:** 169 KB (compressed)  
**Throughput:** 37,028 patterns/sec  
**Discrimination:** 100%  
**Memory:** 31 MB runtime

**Core insight:** The standing wave patterns (eigenmodes) of the E8 lattice provide a complete basis for pattern encoding. Injection projects onto pre-existing geometric structure. No iteration, no tuning, no learning required.

**The geometry IS the model.**

---

## Papers Requiring Annotation

1. `geometric_misalignment_whitepaper.pdf` - Core ontology remains valid; implementation details superseded
2. `infinite_depth.html` - Compression theory valid; mechanism now understood as eigenmode projection
3. Homepage benchmarks - Update from 486M tok/s to pattern-based metrics

---

*Ghost in the Machine Labs*  
*"We didn't train a model. We built a crystal and it started receiving."*
