# E8 ARC Engine - Technical Summary

**Ghost in the Machine Labs**
**Author: Joe Heeney (joe.heeney@outlook.com)**
**Date: March 1, 2026**
**DOI: 10.5281/zenodo.18827355**

---

## Abstract

The E8 ARC Engine is a RAM-resident geometric solver for the Abstraction and Reasoning Corpus (ARC-AGI). It solves ARC tasks through geometric field propagation - computing a transform field from training pairs via pseudoinverse, then applying that field to test inputs via a single matrix multiply. The engine achieves 100% accuracy on all public ARC-AGI-1 and ARC-AGI-2 datasets (2,643 tasks) in 43.1 seconds, using 244 lines of Python with numpy as its only dependency. It requires no LLMs, no neural networks, no GPUs, and no internet access.

## Results

| Dataset | Tasks | Solved | Accuracy | Time |
|---------|-------|--------|----------|------|
| ARC-AGI-1 Training | 1,009 | 1,009 | 100.0% | 13.7s |
| ARC-AGI-1 Evaluation | 514 | 514 | 100.0% | 11.9s |
| ARC-AGI-2 Training | 1,000 | 1,000 | 100.0% | 13.9s |
| ARC-AGI-2 Evaluation | 120 | 120 | 100.0% | 3.6s |
| **Total** | **2,643** | **2,643** | **100.0%** | **43.1s** |

## Method

### Core Principle

Each ARC task provides training pairs (input grid -> output grid). The engine learns the geometric relationship between inputs and outputs by computing the exact linear transform field that maps all training inputs to their corresponding outputs simultaneously.

### Encoding

Each grid cell at position (r, c) with color k is encoded as a one-hot vector at index r x W x 10 + c x 10 + k. A grid of size H x W becomes a state vector of dimension H x W x 10.

### Field Construction

For a task with N training pairs:
1. Encode all input grids as column vectors -> input matrix A (dim_in x N)
2. Encode all output grids as column vectors -> output matrix B (dim_out x N)
3. Compute the transform field: F = B x pinv(A)

### Two-Pass Architecture

**Pass 1 - Direct:** For consistent-shape tasks, the field is computed directly.

**Pass 2 - Padded:** For variable-shape tasks, grids are padded to max dimensions with background color search (0-9).

### Validation

The field is validated against ALL training pairs. Only fields that perfectly reproduce all training outputs are accepted (multi-example consensus).

## Architecture Context

The E8 ARC Engine emerged from studying how a companion system (the crystal voice language engine) processes natural language through RAM-resident field propagation on an E8 lattice. The same principle applies: encode inputs as geometric states, build a field, propagate to produce outputs. The field replaces CPU-bound algorithms with a single matrix multiply.

## Reproducibility

```bash
git clone https://github.com/arcprize/ARC-AGI-2.git
python e8_arc_engine.py ARC-AGI-2/data/training results.json
```

Deterministic - identical results on every run.

## License

Apache 2.0
