# openseespy_solvers.nvmath

GPU direct sparse linear solver implemented with [NVIDIA nvmath-python](https://pypi.org/project/nvmath-python/).

**Recommended for `Ax = b` via `PythonSparse` when you have CUDA** (static, transient, etc.) —
often faster than many native
OpenSees direct solvers on medium-to-large systems. See
[Recommended solvers](../recommended-solvers.md).

Importing this module does not require ``nvmath-python``; the dependency is loaded when
``direct_solver()`` is called. Install a CUDA-matched stack (see
[installation](../installation.md#gpu-install-cuda)), for example:

```bash
pip install cupy-cuda13x
pip install "nvmath-python[cu13]>=0.9.0"
```

## Functions

| Function | nvMath routine | Problem |
|----------|----------------|---------|
| [`direct_solver`](../api/nvmath.md#openseespy_solvers.nvmath.direct_solver) | `nvmath.sparse.advanced.DirectSolver` | `Ax = b` (direct, GPU) |

``direct_solver()`` runs on the GPU by default (requires CuPy and a CUDA GPU).
OpenSees supplies CPU buffers; the solver copies them to the device.

Optional ``execution=`` and ``plan_algorithm=`` arguments are forwarded to
[`DirectSolver`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html).

## See Also

[`openseespy_solvers.cupy.spsolve`](cupy.md) · [`openseespy_solvers.scipy.spsolve`](scipy.md)

[NVIDIA nvmath sparse documentation](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/index.html)
