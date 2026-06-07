# Tutorial

This section shows how to wire solvers into OpenSeesPy. Model-building details are omitted;
see the [OpenSeesPy documentation](https://openseespydoc.readthedocs.io/) and
[examples](examples.md).

For install steps (CPU vs GPU wheels), see [Installation](installation.md). For which
solver to pick first, see [Recommended solvers](recommended-solvers.md).

## Recommended choices (summary)

| Analysis | With NVIDIA GPU | CPU only |
|----------|-----------------|----------|
| Linear `Ax = b` (static, transient, …) | `nvmath.direct_solver` | `scipy.spsolve` or `scipy.umfpack` |
| Eigen `K x = λ M x` | `cupy.eigsh` | `scipy.eigsh` |

These are often faster than built-in OpenSees solvers for many models because they use
mature library implementations (SuperLU, UMFPACK, ARPACK, nvmath.sparse/cuDSS, etc.) with
factorization reuse when the matrix is unchanged.

---

## Static analysis — CPU direct solver

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("Plain")
ops.constraints("Plain")
ops.integrator("LoadControl", 1.0)
ops.algorithm("Linear")
ops.analysis("Static")
ops.analyze(1)
```

The LU factorization is reused when OpenSees reports an unchanged matrix structure and
sparsity pattern between solves.

For large systems on CPU, prefer [`umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack)
after installing UMFPACK (see [installation — UMFPACK](installation.md#umfpack)).

---

## Static analysis — GPU (nvmath.sparse)

Requires `cupy` and nvmath wheels matching your driver ([GPU install](installation.md#gpu)).

```python
import openseespy.opensees as ops
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("Plain")
ops.constraints("Plain")
ops.integrator("LoadControl", 1.0)
ops.algorithm("Linear")
ops.analysis("Static")
ops.analyze(1)
```

---

## Static analysis with CG and a Jacobi preconditioner

Use when a direct factorization is too expensive:

```python
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, maxiter=500, M=precond.jacobi)
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("RCM")
```

The `M` argument accepts a preconditioner object or a callable `M(A)` that receives the
assembled matrix; see [Preconditioners](user-guide/preconditioners.md).

---

## Modal analysis — CPU (`scipy.eigsh`)

Recommended CPU path for `K x = λ M x` with the full mass matrix OpenSees assembles:

```python
from openseespy_solvers.scipy import eigsh

eigsolver = eigsh(tol=1e-8)
ops.numberer("RCM")
lam = ops.eigen("PythonSparse", 5, eigsolver.to_openseespy())
```

Eigenvalues and eigenvectors are written to OpenSees output buffers in place.
`eigsolver.K` and `eigsolver.M` hold the assembled stiffness and mass matrices from the
last call.

---

## Modal analysis — GPU (`cupy.eigsh`)

Recommended GPU path when you have CUDA (default `mass_mode="general"`: full `M`,
`scipy.sparse.linalg.eigsh` / ARPACK on CPU, inner `(K - σ M)⁻¹` solves on GPU):

```python
from openseespy_solvers.cupy import eigsh

eigsolver = eigsh(tol=1e-8)
ops.numberer("RCM")
lam = ops.eigen("PythonSparse", 5, eigsolver.to_openseespy())
```

For row-sum **lumped** mass (faster, different physics), use `mass_mode="lumped"`.

**LOBPCG** is an alternative iterative eigen solver:

```python
from openseespy_solvers.cupy import lobpcg, precond
from openseespy_solvers.nvmath import direct_solver

eigsolver = lobpcg(M=precond.direct(direct_solver()), tol=1e-8)
ops.numberer("RCM")
lam = ops.eigen("PythonSparse", 5, eigsolver.to_openseespy())
```

---

## GPU iterative static (`cupy.cg`)

```python
from openseespy_solvers.cupy import cg

solver = cg(rtol=1e-8)
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("RCM")
```

Requires `cupy`. After a solve, `solver.A` and `solver.x` are `cupy` arrays on device.

---

## See Also

- [Recommended solvers](recommended-solvers.md)
- [`scipy.spsolve` API](api/scipy.md#openseespy_solvers.scipy.spsolve)
- [`scipy.umfpack` API](api/scipy.md#openseespy_solvers.scipy.umfpack)
- [`scipy.eigsh` API](api/scipy.md#openseespy_solvers.scipy.eigsh)
- [`nvmath.direct_solver` API](api/nvmath.md#openseespy_solvers.nvmath.direct_solver)
- [`cupy.eigsh` API](api/cupy.md#openseespy_solvers.cupy.eigsh)
- [PythonSparse interface](user-guide/pythonsparse-interface.md)
