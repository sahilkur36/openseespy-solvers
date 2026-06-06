# openseespy_solvers.nvmath

GPU direct sparse linear solver implemented with [NVIDIA nvmath-python](https://pypi.org/project/nvmath-python/).

**Recommended for `Ax = b` via `PythonSparse` when you have CUDA** for static, transient,
and similar linear solve steps. See [Recommended solvers](../recommended-solvers.md).

Requires **serial OpenSeesPy** — assembly stays on the CPU; nvMath factorizes on GPU.
See [parallelism](../user-guide/pythonsparse-interface.md#parallelism).

Importing this module does not require ``nvmath-python``; the dependency is loaded when
``direct_solver()`` is called. Install a CUDA-matched stack (see
[installation](../installation.md#gpu)), for example:

```bash
python -m pip install "openseespy-solvers[cuda13]"
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
