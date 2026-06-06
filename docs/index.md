# openseespy-solvers

`openseespy-solvers` provides SciPy-style sparse linear and eigen solvers for
[OpenSeesPy](https://openseespydoc.readthedocs.io/) `PythonSparse` commands.

The package wraps numerical libraries such as
[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html),
[`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html),
and NVIDIA nvMath as solver objects. OpenSeesPy assembles the model matrices; the solver
object receives those arrays, runs the selected backend, and writes results back to
OpenSeesPy.

## Start Here

```bash
python -m pip install openseespy-solvers
```

If OpenSeesPy is not already installed:

```bash
python -m pip install "openseespy-solvers[opensees]"
```

Then wire a solver into OpenSeesPy:

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

For a full model example, continue with the [tutorial](getting-started.md).

## Recommended Defaults

| Analysis | CPU | NVIDIA GPU |
|----------|-----|-------------|
| Static or transient linear solve | `scipy.spsolve`; `scipy.umfpack` for larger CPU systems | `nvmath.direct_solver` |
| Generalized eigen solve | `scipy.eigsh` | `cupy.eigsh` |

See [Recommended solvers](recommended-solvers.md) for the reasoning, install notes, and
alternatives.

## Modules

| Module | Provides |
|--------|----------|
| [`scipy`](reference/scipy.md) | CPU solvers: `spsolve`, `umfpack`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| [`scipy.precond`](api/precond.md) | CPU preconditioners: `jacobi`, `ilu`, `direct` |
| [`cupy`](reference/cupy.md) | GPU solvers: `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| [`cupy.precond`](api/precond_cupy.md) | GPU preconditioners: `jacobi`, `ilu`, `direct` |
| [`nvmath`](reference/nvmath.md) | GPU direct sparse solver: `direct_solver` |
| [`hybrid`](api/hybrid.md) | Direct factorization reused as a GMRES preconditioner |

Factory signatures match the underlying SciPy or CuPy functions where possible. The matrix
and right-hand side are supplied by OpenSeesPy at solve time.

## Helpful Pages

- [Installation](installation.md)
- [Tutorial](getting-started.md)
- [Recommended solvers](recommended-solvers.md)
- [Examples](examples.md)
- [Solver objects](user-guide/solver-objects.md)
- [PythonSparse interface](user-guide/pythonsparse-interface.md)
- [API reference](api/scipy.md)

## Support

Source, issues, and contributions are hosted on
[GitHub](https://github.com/gaaraujo/openseespy-solvers).
