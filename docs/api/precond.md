# openseespy_solvers.scipy.precond

CPU built-in preconditioners for iterative `scipy.sparse.linalg` solvers.

Pass these callables through the `M=` keyword of `cg`, `gmres`, or `lobpcg`. When `M` is
callable, the solver calls it as `M(A)` after OpenSeesPy supplies the assembled matrix.

## Preconditioners

| Preconditioner | Description |
|---------|-------------|
| [`jacobi`](#openseespy_solvers.scipy.precond.jacobi) | Diagonal/Jacobi preconditioner |
| [`ilu`](#openseespy_solvers.scipy.precond.ilu) | Incomplete LU preconditioner |
| [`direct`](#openseespy_solvers.scipy.precond.direct) | Direct-solver preconditioner |

## Usage

```python
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_openseespy())
```

## Function Reference

::: openseespy_solvers.scipy.precond
    options:
      members:
        - jacobi
        - ilu
        - direct
      show_root_heading: false
      heading_level: 3
