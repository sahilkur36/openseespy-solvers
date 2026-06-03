# openseespy_solvers

Sparse linear algebra solvers for [OpenSeesPy](https://openseespydoc.readthedocs.io/)
`PythonSparse` system and eigen commands.

The package wraps existing numerical libraries—primarily
[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
and [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html)—as
solver objects that OpenSeesPy can call directly. Factory signatures match the
underlying library except that the system matrix and right-hand side are assembled
by OpenSees and passed at solve time.

## Submodules

| Module | Description |
|--------|-------------|
| [`scipy`](reference/scipy.md) | CPU solvers (`spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`) |
| [`scipy.precond`](api/precond.md) | Preconditioner factories for iterative solvers |
| [`cupy`](reference/cupy.md) | GPU solvers (`spsolve`, `cg`, `gmres`, `lobpcg`) |

## Quick reference

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_opensees())
ops.analyze(1)
```

After a linear solve, `solver.A`, `solver.b`, and `solver.x` refer to the cached
matrix and vectors from the last call. See [LinearSolver attributes](user-guide/solver-objects.md).

## See Also

- [PythonSparse documentation](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
- [Tutorial](getting-started.md)
- [API reference](api/scipy.md)
