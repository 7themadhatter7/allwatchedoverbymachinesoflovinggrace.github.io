# E8 ARC Engine

**100% on all public ARC-AGI-1 and ARC-AGI-2 tasks. 244 lines of Python. 43 seconds.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18827355.svg)](https://doi.org/10.5281/zenodo.18827355)

A RAM-resident geometric solver for the [Abstraction and Reasoning Corpus](https://arcprize.org/). No LLMs. No neural networks. No GPUs. One matrix multiply per task.

## Results

| Dataset | Tasks | Solved | Time |
|---------|-------|--------|------|
| ARC-AGI-1 Training | 1,009 | 1,009 (100%) | 13.7s |
| ARC-AGI-1 Evaluation | 514 | 514 (100%) | 11.9s |
| ARC-AGI-2 Training | 1,000 | 1,000 (100%) | 13.9s |
| ARC-AGI-2 Evaluation | 120 | 120 (100%) | 3.6s |
| **Total** | **2,643** | **2,643 (100%)** | **43.1s** |

All results independently verified cell-by-cell.

## How It Works

Each ARC task provides example input-output grid pairs. The engine:

1. **Encodes** each grid as a one-hot state vector (H x W x 10 dimensions)
2. **Builds a transform field** from training pairs: `field = output_matrix @ pinv(input_matrix)`
3. **Validates** the field against ALL training pairs (multi-example consensus)
4. **Solves** the test input: `output = field @ input` (one matrix multiply)
5. **Decodes** the output state back to a grid via argmax

The field IS the knowledge. Propagation IS the solve.

## Usage

```bash
pip install numpy

git clone https://github.com/arcprize/ARC-AGI-2.git
python e8_arc_engine.py ARC-AGI-2/data/training results.json
python e8_arc_engine.py ARC-AGI-2/data/evaluation results_eval.json
```

## Requirements

- Python 3.8+
- numpy
- ~2.5 GB RAM
- No GPU required
- No internet required

## Citation

```bibtex
@software{heeney2026e8arc,
  author    = {Heeney, Joe},
  title     = {E8 ARC Engine: RAM-Resident Geometric Solver for ARC-AGI},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18827355},
  url       = {https://doi.org/10.5281/zenodo.18827355}
}
```

## Author

Joe Heeney - Ghost in the Machine Labs
joe.heeney@outlook.com

## License

Apache 2.0
