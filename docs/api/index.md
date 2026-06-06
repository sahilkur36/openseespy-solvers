# API Reference

This section documents the public Python API for `openseespy-solvers`.

The public modules are:

| Module | Purpose |
|--------|---------|
| [`openseespy_solvers.scipy`](scipy.md) | CPU linear and eigen solver factories |
| [`openseespy_solvers.scipy.precond`](precond.md) | CPU preconditioner factories |
| [`openseespy_solvers.cupy`](cupy.md) | CUDA linear and eigen solver factories |
| [`openseespy_solvers.cupy.precond`](precond_cupy.md) | CUDA preconditioner factories |
| [`openseespy_solvers.nvmath`](nvmath.md) | NVIDIA nvMath direct sparse solver factory |
| [`openseespy_solvers.hybrid`](hybrid.md) | Hybrid direct-factorization/GMRES solver factory |
| [`openseespy_solvers.exceptions`](exceptions.md) | Public exception types |

Private modules and names beginning with `_` are implementation details.

## Importing

Use the backend namespace that matches the solver you want:

```python
from openseespy_solvers.scipy import spsolve, eigsh
from openseespy_solvers.scipy import precond
```

For optional GPU backends:

```python
from openseespy_solvers.cupy import eigsh
from openseespy_solvers.nvmath import direct_solver
```

All solver factories return solver objects. Register those objects with OpenSeesPy by
passing `solver.to_openseespy()` to `ops.system("PythonSparse", ...)` or
`ops.eigen("PythonSparse", ...)`.

## Linear Solvers

| Factory | Backend | Purpose |
|---------|---------|---------|
| [`scipy.spsolve`](scipy.md#openseespy_solvers.scipy.spsolve) | SciPy / SuperLU | CPU sparse direct solve |
| [`scipy.umfpack`](scipy.md#openseespy_solvers.scipy.umfpack) | scikit-umfpack | CPU sparse direct solve using UMFPACK |
| [`scipy.cg`](scipy.md#openseespy_solvers.scipy.cg) | SciPy | CPU Conjugate Gradient |
| [`scipy.gmres`](scipy.md#openseespy_solvers.scipy.gmres) | SciPy | CPU GMRES |
| [`cupy.spsolve`](cupy.md#openseespy_solvers.cupy.spsolve) | CuPy | CUDA sparse direct solve |
| [`cupy.cg`](cupy.md#openseespy_solvers.cupy.cg) | CuPy | CUDA Conjugate Gradient |
| [`cupy.gmres`](cupy.md#openseespy_solvers.cupy.gmres) | CuPy | CUDA GMRES |
| [`nvmath.direct_solver`](nvmath.md#openseespy_solvers.nvmath.direct_solver) | NVIDIA nvMath | CUDA sparse direct solve |
| [`hybrid`](hybrid.md#openseespy_solvers.hybrid.hybrid) | SciPy-compatible direct solver + GMRES | Reuse a direct factorization as a GMRES preconditioner |

## Eigen Solvers

| Factory | Backend | Purpose |
|---------|---------|---------|
| [`scipy.eigsh`](scipy.md#openseespy_solvers.scipy.eigsh) | SciPy / ARPACK | CPU generalized symmetric eigen solve |
| [`scipy.lobpcg`](scipy.md#openseespy_solvers.scipy.lobpcg) | SciPy | CPU LOBPCG eigen solve |
| [`cupy.eigsh`](cupy.md#openseespy_solvers.cupy.eigsh) | CuPy + SciPy ARPACK | CUDA-assisted generalized symmetric eigen solve |
| [`cupy.lobpcg`](cupy.md#openseespy_solvers.cupy.lobpcg) | CuPy | CUDA LOBPCG eigen solve |

## Preconditioners

| Factory | Backend | Purpose |
|---------|---------|---------|
| [`scipy.precond.jacobi`](precond.md#openseespy_solvers.scipy.precond.jacobi) | SciPy | Diagonal/Jacobi preconditioner |
| [`scipy.precond.ilu`](precond.md#openseespy_solvers.scipy.precond.ilu) | SciPy | Incomplete LU preconditioner |
| [`scipy.precond.direct`](precond.md#openseespy_solvers.scipy.precond.direct) | SciPy | Direct-solver preconditioner |
| [`cupy.precond.jacobi`](precond_cupy.md#openseespy_solvers.cupy.precond.jacobi) | CuPy | Diagonal/Jacobi preconditioner |
| [`cupy.precond.ilu`](precond_cupy.md#openseespy_solvers.cupy.precond.ilu) | CuPy | Incomplete LU preconditioner |
| [`cupy.precond.direct`](precond_cupy.md#openseespy_solvers.cupy.precond.direct) | CuPy | Direct-solver preconditioner |

## OpenSeesPy Difference

These factories intentionally resemble `scipy.sparse.linalg` and
`cupyx.scipy.sparse.linalg`, but they do not accept `A` and `b` when constructed.
OpenSeesPy supplies the assembled matrix and right-hand side at solve time.

```python
solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

See [Solver objects](../user-guide/solver-objects.md) and
[PythonSparse interface](../user-guide/pythonsparse-interface.md) for lifecycle details.
