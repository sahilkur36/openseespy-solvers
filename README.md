# openseespy-solvers

Sparse linear algebra solvers for OpenSeesPy
[`PythonSparse`](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
commands.

**Documentation:** [openseespy-solvers.readthedocs.io](https://openseespy-solvers.readthedocs.io/)

## Installation

Requires **Python ≥ 3.12**, **NumPy ≥ 1.26**, and **SciPy ≥ 1.12**.

```bash
pip install openseespy-solvers
pip install openseespy-solvers[gpu]   # CuPy ≥ 13 (CUDA-specific wheel may be required)
```

## Example

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_opensees())
ops.analyze(1)
```

## Submodules

| Module | Functions |
|--------|-----------|
| `openseespy_solvers.scipy` | `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| `openseespy_solvers.scipy.precond` | `jacobi`, `ilu` |
| `openseespy_solvers.cupy` | `spsolve`, `cg`, `gmres`, `lobpcg` |
| `openseespy_solvers.cupy.precond` | `jacobi`, `ilu` |

Factory signatures match [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
and [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html);
`A` and `b` are supplied by OpenSees at solve time.

See the [tutorial](https://openseespy-solvers.readthedocs.io/en/latest/getting-started/) and
[API reference](https://openseespy-solvers.readthedocs.io/en/latest/api/scipy/) for details.

## License

BSD 3-Clause — see [LICENSE](LICENSE).