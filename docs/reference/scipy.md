# openseespy_solvers.scipy

Sparse linear algebra solvers implemented with SciPy.

This module mirrors [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).
Each function configures a solver object; pass `solver.to_openseespy()` to OpenSeesPy.

**Recommended on CPU (no GPU):** [`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve) or
[`umfpack`](../api/scipy.md#openseespy_solvers.scipy.umfpack) for `PythonSparse` linear steps;
[`eigsh`](../api/scipy.md#openseespy_solvers.scipy.eigsh) for `K x = λ M x`. See
[Recommended solvers](../recommended-solvers.md).

## Functions

| Function | SciPy routine | Problem |
|----------|---------------|---------|
| [`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve) | `splu` / `spsolve` | `Ax = b` (direct, SuperLU) |
| [`umfpack`](../api/scipy.md#openseespy_solvers.scipy.umfpack) | `scikits.umfpack` | `Ax = b` (direct, 64-bit UMFPACK) |
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

[`umfpack`](../api/scipy.md#openseespy_solvers.scipy.umfpack) is a 64-bit direct
solver (`scikits.umfpack.UmfpackContext("dl")`) distinct from the SuperLU-based
[`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve). It requires the
optional `scikit-umfpack` package:
`python -m pip install "openseespy-solvers[umfpack]"`.
The import is deferred until `umfpack()` is called, so the SciPy namespace
imports without it.

## See Also

[`openseespy_solvers.cupy`](cupy.md)
