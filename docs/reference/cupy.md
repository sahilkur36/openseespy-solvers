# openseespy_solvers.cupy

Sparse linear algebra solvers implemented with CuPy.

This module mirrors [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html).
Importing it requires CuPy (`pip install openseespy-solvers[gpu]`).

## Functions

| Function | CuPy routine | Problem |
|----------|--------------|---------|
| [`spsolve`](../api/cupy.md#openseespy_solvers.cupy.spsolve) | `spsolve` | `Ax = b` (direct) |
| [`cg`](../api/cupy.md#openseespy_solvers.cupy.cg) | `cg` | `Ax = b` (iterative) |
| [`gmres`](../api/cupy.md#openseespy_solvers.cupy.gmres) | `gmres` | `Ax = b` (iterative) |
| [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg) | `lobpcg` | `K x = λ M x` |

## Submodule

[`precond`](../api/precond_cupy.md) — Jacobi and GPU ILU (`fill_factor=1`) factories.

## Notes `cupyx.scipy.sparse.linalg.eigsh` does not support the
generalized eigenproblem with a mass matrix. Use [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg)
for GPU modal analysis.

On older CuPy releases, relative tolerance may be passed as `tol` rather than
`rtol`; the backend selects the correct keyword automatically.

## See Also

[`openseespy_solvers.scipy`](scipy.md)

[`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)

[`cupyx.scipy.sparse.linalg.cg`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html)
