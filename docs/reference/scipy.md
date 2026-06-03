# openseespy_solvers.scipy

Sparse linear algebra solvers implemented with SciPy.

This module mirrors [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).
Each function configures a solver object; pass `solver.to_opensees()` to OpenSeesPy.

## Functions

| Function | SciPy routine | Problem |
|----------|---------------|---------|
| [`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve) | `splu` / `spsolve` | `Ax = b` (direct) |
| [`cg`](../api/scipy.md#openseespy_solvers.scipy.cg) | `cg` | `Ax = b` (SPD, iterative) |
| [`gmres`](../api/scipy.md#openseespy_solvers.scipy.gmres) | `gmres` | `Ax = b` (iterative) |
| [`eigsh`](../api/scipy.md#openseespy_solvers.scipy.eigsh) | `eigsh` | `K x = λ M x` |
| [`lobpcg`](../api/scipy.md#openseespy_solvers.scipy.lobpcg) | `lobpcg` | `K x = λ M x` |

## Submodule

[`precond`](../api/precond.md) — Jacobi and ILU preconditioner factories.

## Notes

Parameters documented in the [API reference](../api/scipy.md) follow SciPy naming.
Additional parameters (`scheme`, `writable`, `debug`) control the OpenSees
interface; they are described in each function's docstring.

## See Also

[`openseespy_solvers.cupy`](cupy.md)
