# Harmonic Stack v2 Deployment — February 2, 2026

## Ghost in the Machine Labs — Changelog

---

### Summary

Full integration of the 30B code generation model into the Harmonic Stack, fixing three critical bugs that prevented the cooperative research pipeline from producing executable code. The self-improvement loop now generates, validates, and iterates on actual ARC solutions instead of crashing on environmental errors.

---

### Critical Fixes

**1. Missing `grid` variable in code reconstruction**
The `ollama_complete` function reconstructed solved code as `def solve(input_grid): ...` but never included `grid = np.array(input_grid)`. Every generated solution referenced `grid` (because the prompt prefix included it) but the reconstructed code didn't. Result: 100% NameError crashes.

**Fix:** Added `grid = np.array(input_grid)` to the reconstruction template in `ollama_complete`.

**2. Missing standard library imports in execution namespace**
The `validate_solution` function executed generated code in a restricted namespace containing only `np` and `numpy`. Any code using `deque`, `Counter`, `defaultdict`, or `itertools` crashed with NameError even though the imports were present in the source.

**Fix:** Expanded execution namespace to include `deque`, `Counter`, `defaultdict`, and `itertools`.

**3. Token truncation killing code mid-generation**
The 30B solver had `num_predict: 1024`, which truncated complex solutions mid-function. Combined with `profile[:200]` and `hypothesis[:200]`, the model received minimal context from the upgraded v2 analysts.

**Fix:** `num_predict: 1024 → 4096`, profile/hypothesis truncation `200 → 600` chars.

---

### New Model: `solver`

Created dedicated Modelfile for `qwen3:30b-a3b` with:

- Precision-enforcing system prompt (10 absolute rules for ARC code generation)
- `temperature: 0.15` (tighter than generic 0.3)
- `num_ctx: 32768`, `num_predict: 8192`
- `top_p: 0.85`, `repeat_penalty: 1.1`
- Explicit requirements: `grid[row, col]` indexing, `grid.tolist()` return format, visited arrays for BFS, Moore/Von Neumann neighborhood handling

The solver is the 11th model in the Harmonic Stack, completing the pipeline from analysis through code generation.

---

### Code Prefix Enhancement

The raw completion prefix sent to the solver now includes inline requirements:

```
# REQUIREMENTS:
# - grid = np.array(input_grid) is already done below. Use 'grid' not 'input_grid'.
# - Return grid.tolist() at the end (list of lists, not numpy array).
# - Handle ALL colors explicitly. Do not assume background color.
# - Use visited arrays for BFS/flood fill to prevent infinite loops.
# - grid[row, col] indexing (row=vertical, col=horizontal).
# - Include ALL 8 neighbors for Moore neighborhood if diagonal matters.
```

This eliminates the most common failure modes identified in the Harmonic Stack analysis reports.

---

### Self-Optimizer Configuration Updates

| Parameter | Before | After |
|-----------|--------|-------|
| context_budget_analyst | 2,000 | 4,000 |
| context_budget_researcher | 4,000 | 8,000 |
| context_budget_solver | 8,000 | 16,000 |
| solve_timeout | 120s | 180s |
| research_timeout | 45s | 60s |

Budgets doubled to match expanded v2 model context windows.

---

### Benchmark Results

**V2 benchmark** (run under load with cooperative research daemon active):

| Model | Individual tok/s | Notes |
|-------|-----------------|-------|
| solver (30B) | 69.7 | Matches 8B individual speed despite 4× size |
| analyst (8B) | 29.0 | Under contention |
| executive (8B) | 19.7 | Under contention |
| research_director (14B) | 19.7 | Under contention |
| coder (30B) | 18.4 | Under contention |
| operator (8B) | 16.4 | Under contention |

**Parallel throughput under load:** 70.5 agg tok/s at n=8

**Previous idle benchmark (Jan 31):** 417.5 agg tok/s at n=32

Numbers not directly comparable — v2 benchmark ran with active daemons consuming GPU. Published tok/s figures unchanged as they represent idle peak throughput.

---

### Spine TODO Status

| ID | Priority | Task | Status |
|----|----------|------|--------|
| T020 | P0 | First successful model translation | ✅ DONE |
| T021 | P0 | Harmonic Stack v2 model deployment | ✅ DONE |
| T022 | P0 | 30B code generation integration | ✅ DONE |
| T023 | P0 | Self-optimizer context budgets | ✅ DONE |
| T014 | P0 | Status API endpoint | PENDING |
| T018 | P0 | Fix binary-only write path | PENDING |
| T019 | P0 | GGUF dequantization | PENDING |
| T024 | P1 | ARCY ngrok bridge | PENDING |
| T025 | P1 | Benchmark paper — ARCY benchmarks | PENDING |
| T016 | P1 | Webhook/notification system | PENDING |
| T017 | P1 | Self-monitoring dashboard | PENDING |
| T026 | P2 | Academic paper draft | PENDING |

---

### Result

Before these fixes, the cooperative research pipeline produced 100% crash rate on code generation (NameError on every attempt). After fixes, the pipeline generates executable code with real transformation logic. Errors are now algorithmic (wrong cell counts, shape mismatches) rather than environmental — meaning the self-improvement loop can now actually improve.

---

### Files Modified

- `arc_cooperative_research.py` — SOLVER_MODEL, num_predict, code_prefix, reconstruction, execution namespace
- `modelfiles_v2/Modelfile.solver` — new solver Modelfile
- `self_optimization/tuning_config.json` — context budgets and timeouts
- `update_todos.sql` — Spine TODO table updates
- `website/index.html` — v2 deployment messaging, 30B integration, self-optimizing reference
