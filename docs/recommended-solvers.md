# Recommended Solvers

These are practical first choices for OpenSeesPy `PythonSparse` analyses. They are not
universal winners: performance depends on model size, sparsity, hardware, and the native
OpenSees solver used for comparison.

## Linear Systems

Use these for static, transient, or other analyses that solve `Ax = b` through
`ops.system("PythonSparse", ...)`.

| Situation | Start with | Notes |
|-----------|------------|-------|
| CPU | [`scipy.spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve) | Good default; uses SciPy's sparse direct solver |
| CPU, larger systems | [`scipy.umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack) | Optional UMFPACK backend; often worth trying for larger sparse systems |
| NVIDIA GPU | [`nvmath.direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) | Recommended GPU direct solver |

CPU example:

```python
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

GPU example:

```python
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
```

Iterative solvers (`cg`, `gmres`) are useful when direct factorization is too expensive or
when you have a good preconditioner. See [Preconditioners](user-guide/preconditioners.md).

For Newton or transient analyses where the tangent changes slowly, consider
[`hybrid`](api/hybrid.md). It reuses a direct factorization as a GMRES preconditioner until
the system size changes or GMRES fails to converge.

## Eigenproblems

Use these for generalized modal problems `K x = lambda M x` through
`ops.eigen("PythonSparse", ...)`.

| Situation | Start with | Notes |
|-----------|------------|-------|
| CPU | [`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) | ARPACK through SciPy |
| NVIDIA GPU | [`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) | Uses GPU work in the shift-invert path |

CPU example:

```python
from openseespy_solvers.scipy import eigsh

eigsolver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, eigsolver.to_openseespy())
```

GPU example:

```python
from openseespy_solvers.cupy import eigsh

eigsolver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, eigsolver.to_openseespy())
```

`lobpcg` is available for experiments and specialized workflows. For routine modal
analysis, start with `eigsh`.

## Install Recap

CPU:

```bash
python -m pip install "openseespy-solvers[opensees]"
```

Optional UMFPACK:

```bash
python -m pip install "openseespy-solvers[umfpack]"
```

NVIDIA GPU, using CUDA 13.x as an example:

```bash
python -m pip install "openseespy-solvers[opensees,cuda13]"
```

See [Installation](installation.md) for CUDA 12.x, Windows UMFPACK, and troubleshooting.

## Parallelism

These solvers target serial OpenSeesPy. GPU backends accelerate the sparse solve, but model
assembly and the `PythonSparse` callback still run in one process. See
[PythonSparse and parallelism](user-guide/pythonsparse-interface.md#parallelism).

## See Also

- [Tutorial](getting-started.md)
- [Examples](examples.md)
- [Reference: scipy](reference/scipy.md)
- [Reference: cupy](reference/cupy.md)
- [Reference: nvmath](reference/nvmath.md)
