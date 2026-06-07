# openseespy_solvers.cupy.precond

CUDA preconditioner factories for iterative `cupyx.scipy.sparse.linalg` solvers.

Pass these factories through the `M=` keyword of `cg`, `gmres`, or `lobpcg`. When `M` is
callable, the solver calls it as `M(A)` after OpenSeesPy supplies the assembled matrix.

## Preconditioners

| Factory | Description |
|---------|-------------|
| [`jacobi`](#openseespy_solvers.cupy.precond.jacobi) | Diagonal/Jacobi preconditioner on CUDA |
| [`ilu`](#openseespy_solvers.cupy.precond.ilu) | Incomplete LU preconditioner |
| [`direct`](#openseespy_solvers.cupy.precond.direct) | Direct-solver preconditioner |

## Usage

```python
from openseespy_solvers.cupy import cg
from openseespy_solvers.cupy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_openseespy())
```

## Function Reference

::: openseespy_solvers.cupy.precond
    options:
      members:
        - jacobi
        - ilu
        - direct
      show_root_heading: false
      heading_level: 3
