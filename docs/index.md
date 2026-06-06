# openseespy_solvers

Sparse linear algebra solvers for [OpenSeesPy](https://openseespydoc.readthedocs.io/)
`PythonSparse` system and eigen commands.

The package wraps existing numerical libraries—primarily
[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
and [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html)—as
solver objects that OpenSeesPy can call directly. Factory signatures match the
underlying library except that the system matrix and right-hand side are assembled
by OpenSees and passed at solve time.

For linear (static, transient, …) and modal analysis, see **[Recommended solvers](recommended-solvers.md)**:
**`nvmath.direct_solver`** or **`cupy.eigsh`** with a CUDA GPU;
**`scipy.spsolve`** / **`scipy.umfpack`** and **`scipy.eigsh`** on CPU. These paths are
often faster than native OpenSees sparse/direct/eigen solvers on medium-to-large models
because they delegate to heavily optimized third-party libraries.

## Submodules

| Module | Description |
|--------|-------------|
| [`scipy`](reference/scipy.md) | CPU solvers (`spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`) |
| [`scipy.precond`](api/precond.md) | Preconditioner factories (`jacobi`, `ilu`, `direct`) |
| [`cupy`](reference/cupy.md) | GPU solvers (`spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`) |
| [`cupy.precond`](api/precond_cupy.md) | GPU preconditioner factories (`jacobi`, `ilu`, `direct`) |
| [`nvmath`](reference/nvmath.md) | GPU direct sparse solver (`direct_solver`) |
| [`hybrid`](api/hybrid.md) | Frozen factorization + GMRES (`hybrid(direct=...)`) |

## Quick reference

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_openseespy())
ops.analyze(1)
```

After a linear solve, `solver.A`, `solver.b`, and `solver.x` refer to the cached
matrix and vectors from the last call. See [LinearSolver attributes](user-guide/solver-objects.md).

## GitHub

Source, bug reports, and feature requests:
[github.com/gaaraujo/openseespy-solvers](https://github.com/gaaraujo/openseespy-solvers).
Open an [issue](https://github.com/gaaraujo/openseespy-solvers/issues) for problems or
questions; contributions (pull requests, documentation fixes, examples) are welcome there
as well.

## See Also

- [Installation](installation.md)
- [Full setup from scratch](installation.md#full-setup)
- [Verify your install](installation.md#verify)
- [Recommended solvers](recommended-solvers.md)
- [PythonSparse documentation](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
- [Tutorial](getting-started.md)
- [API reference](api/scipy.md)
- [Serial OpenSeesPy and parallelism](user-guide/pythonsparse-interface.md#parallelism)
