# openseespy_solvers.nvmath

NVIDIA nvMath sparse direct solver factory.

This module provides a CUDA direct solver for OpenSeesPy `PythonSparse` linear systems.
It wraps `nvmath.sparse.advanced.DirectSolver` and returns a solver object that can be
registered with `ops.system("PythonSparse", ...)`.

## Solving Linear Problems

| Factory | nvMath analogue | Description |
|---------|-----------------|-------------|
| [`direct_solver`](#openseespy_solvers.nvmath.direct_solver) | `nvmath.sparse.advanced.DirectSolver` | Sparse direct solver on CUDA |

## Installation

Install the CUDA extra matching your driver:

```bash
python -m pip install "openseespy-solvers[cuda13]"   # or [cuda12]
```

## OpenSeesPy Usage

```python
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
```

## Notes

`direct_solver()` runs on the GPU by default. OpenSeesPy supplies CPU buffers; the solver
copies them to the device, factors the sparse matrix with nvMath/cuDSS, and writes the
solution back through the OpenSeesPy callback.

Optional `execution=` and `plan_algorithm=` arguments are forwarded to
`nvmath.sparse.advanced.DirectSolver`.

## Function Reference

::: openseespy_solvers.nvmath
    options:
      members:
        - direct_solver
      show_root_heading: false
      heading_level: 3
