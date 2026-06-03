# Solver objects

Backend factories return solver objects that implement the OpenSees
`PythonSparse` protocol. Application code configures the solver, passes
`solver.to_opensees()` to OpenSeesPy, and inspects cached data after analysis.

## Common methods

`to_opensees(scheme=None, writable=None)`
: Return the OpenSees configuration dict. See [to_opensees](to-opensees.md).

`solve(**kwargs)`
: Called by OpenSees; not normally invoked from application code.

`formAp(**kwargs)` *(linear solvers only)*
: Matrix-vector product `Ap = A @ p`.

## Linear solver attributes

`A`
: Cached sparse system matrix (`scipy.sparse.spmatrix` or
  `cupyx.scipy.sparse.spmatrix`).

`b`
: Right-hand side from the last solve.

`x`
: Solution from the last solve.

`stats`
: Runtime statistics (`num_solves`, `last_solve_time`, `last_info`,
  `last_num_iterations`, `last_residual_norm`, `last_error`).

## Eigen solver attributes

`K`, `M`
: Cached stiffness and mass matrices from the last solve.

`stats`
: Runtime statistics (`num_solves`, `last_solve_time`, `last_num_modes`,
  `last_eigenvalues`, `last_info`, `last_error`).

## copy.copy

`copy.copy(solver)` returns a new instance with the same keyword configuration
but reset internal cache and statistics. OpenSees uses this when cloning a
system of equations.

## Notes

Set `debug=True` at construction to propagate exceptions from `solve` and
`formAp` instead of returning a negative status code (linear solvers only).

## Compute precision (`dtype`)

Every factory accepts ``dtype`` (default ``numpy.float64``). Supported values are
``float32`` and ``float64``. The internal sparse solve, matrix cache (``A``,
``K``, ``M``), and right-hand sides use that precision.

OpenSees ``PythonSparse`` buffers are always double precision: matrix entries,
``b``, ``x``, eigenvalues, and eigenvectors are read and written as
``float64``. The solver casts at the boundary so you can trade accuracy for
speed or memory without changing the OpenSees side.

```python
from openseespy_solvers.scipy import cg

solver = cg(rtol=1e-6, dtype="float32")
```

On GPU, ``openseespy_solvers.cupy`` factories accept the same ``dtype`` keyword.

## See Also

[PythonSparse interface](pythonsparse-interface.md)

[Tutorial](../getting-started.md)
