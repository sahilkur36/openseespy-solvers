# openseespy_solvers.cupy

Sparse linear algebra solvers implemented with CuPy.

This module mirrors [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html).
Importing it requires CuPy. Install a CUDA-matched extra such as
`openseespy-solvers[cuda13]` or `openseespy-solvers[cuda12]`; see
[GPU install](../installation.md#gpu).

Requires **serial OpenSeesPy** — OpenSees assembles the matrix on the CPU; CuPy runs the
solve on GPU. See [parallelism](../user-guide/pythonsparse-interface.md#parallelism).

**Recommended on GPU:** [`eigsh`](../api/cupy.md#openseespy_solvers.cupy.eigsh) for modal
analysis (default `mass_mode="general"`). Pair with
[`nvmath.direct_solver`](nvmath.md) for linear `PythonSparse` steps. See
[Recommended solvers](../recommended-solvers.md).

## Functions

| Function | CuPy routine | Problem |
|----------|--------------|---------|
| [`spsolve`](../api/cupy.md#openseespy_solvers.cupy.spsolve) | `splu` | `Ax = b` (direct, SuperLU) |
| [`cg`](../api/cupy.md#openseespy_solvers.cupy.cg) | `cg` | `Ax = b` (iterative) |
| [`gmres`](../api/cupy.md#openseespy_solvers.cupy.gmres) | `gmres` | `Ax = b` (iterative) |
| [`eigsh`](../api/cupy.md#openseespy_solvers.cupy.eigsh) | shift-invert + `eigsh` / SciPy ARPACK | `K x = λ M x` |
| [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg) | `lobpcg` | `K x = λ M x` |

## Submodule

[`precond`](../api/precond_cupy.md) — ``jacobi``, ``direct`` (LOBPCG on ``K``), and ``ilu`` factories.

## Notes

[`eigsh`](../api/cupy.md#openseespy_solvers.cupy.eigsh) solves `K x = λ M x` with
shift-invert. ``mass_mode='diagonal'`` / ``'lumped'`` use GPU Lanczos on
``(K - σ diag(m))⁻¹ diag(m)``; ``mass_mode='general'`` uses SciPy ARPACK with GPU
inner solves on ``(K - σ M)⁻¹``. Raw
[`cupyx.scipy.sparse.linalg.eigsh`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html)
does not accept a mass matrix — use [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg)
when you need a different iterative strategy.

On older CuPy releases, relative tolerance may be passed as `tol` rather than
`rtol`; the backend selects the correct keyword automatically.

## See Also

[`openseespy_solvers.scipy`](scipy.md)

[`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)

[`cupyx.scipy.sparse.linalg.cg`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html)
