# Preconditioners

Iterative solvers accept an optional preconditioner through the `M` keyword,
consistent with [`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
and [`cupyx.scipy.sparse.linalg.cg`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html).

## Callable factories

The system matrix `A` is not available when the solver object is constructed.
To build a preconditioner from `A`, pass a callable:

```python
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(M=precond.jacobi)
```

The factory is invoked as `M(A)` during `solve` when the matrix structure or
coefficients change. When `matrix_status='UNCHANGED'`, the cached
preconditioner is reused.

## Accepted forms of `M`

`None`
: No preconditioning.

sparse matrix or `LinearOperator`
: A fixed preconditioner that does not depend on the current `A`.

callable
: A factory `M(A)` returning a sparse matrix or `LinearOperator`.

## Built-in factories (scipy)

[`openseespy_solvers.scipy.precond`](../api/precond.md) provides:

[`jacobi`](../api/precond.md#openseespy_solvers.scipy.precond.jacobi)
: `M = diag(1 / diag(A))`.

[`ilu`](../api/precond.md#openseespy_solvers.scipy.precond.ilu)
: Incomplete LU as a `LinearOperator`; extra keywords are forwarded to
  `scipy.sparse.linalg.spilu`.

[`direct`](../api/precond.md#openseespy_solvers.scipy.precond.direct)
: Full sparse factorization as a `LinearOperator`, primarily for
  [`lobpcg`](../api/scipy.md#openseespy_solvers.scipy.lobpcg) ``M=``.
  Pass a direct solver, e.g. ``M=precond.direct(spsolve())``.

## Built-in factories (cupy)

[`openseespy_solvers.cupy.precond`](../api/precond_cupy.md) provides the same
pattern on GPU:

[`jacobi`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.jacobi)
: Diagonal preconditioner as a device ``LinearOperator`` (element-wise apply).

[`ilu`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.ilu)
: Incomplete LU via [`cupyx.scipy.sparse.linalg.spilu`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spilu.html).
Defaults to ``fill_factor=1`` so factorization runs on the GPU with **no
fill-in** (ILU(0)-like). Other ``fill_factor`` values use `scipy.sparse.linalg` on CPU for the
factorization; only ``solve`` stays on GPU.

[`direct`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.direct)
: Full sparse factorization as a ``LinearOperator``, primarily for
  [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg) ``M=``.
  Pass a direct solver, e.g. ``M=precond.direct(direct_solver())``.

## Example (scipy)

```python
from openseespy_solvers.scipy import cg, precond

solver = cg(
    rtol=1e-8,
    M=lambda A: precond.ilu(A, drop_tol=1e-4, fill_factor=10),
)
```

## Example (cupy)

```python
from openseespy_solvers.cupy import cg, precond

solver = cg(rtol=1e-8, M=precond.ilu)  # fill_factor=1, GPU ILU(0)-like
```

## Notes

Pass a `cupyx.scipy.sparse.linalg.LinearOperator` or a custom factory when you
need a preconditioner not covered by `precond`.

## See Also

[`scipy.cg`](../api/scipy.md#openseespy_solvers.scipy.cg)

[`scipy.gmres`](../api/scipy.md#openseespy_solvers.scipy.gmres)

[scipy tutorial: Preconditioned conjugate gradient](https://docs.scipy.org/doc/scipy/tutorial/sparse.html)
