# openseespy-solvers

Sparse linear algebra solvers for OpenSeesPy
[`PythonSparse`](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
commands.

**Documentation:** [openseespy-solvers.readthedocs.io](https://openseespy-solvers.readthedocs.io/)

## Installation

Requires **Python ≥ 3.12**, **NumPy ≥ 1.26**, **SciPy ≥ 1.12**, and **OpenSeesPy**
(already in your environment, or `pip install openseespy-solvers[opensees]`).

```bash
pip install openseespy-solvers
```

**CPU:** `pip install openseespy-solvers[umfpack]` optional for faster direct solves.

**GPU:** after `nvidia-smi`, install matching wheels, e.g. `cupy-cuda13x` and
`pip install "nvmath-python[cu13]>=0.9.0"`.

Full steps: [installation guide](https://openseespy-solvers.readthedocs.io/en/latest/installation/).

**Verify install** (after clone + editable install): run `pytest`, the
[`examples/solvers/`](examples/solvers/) smoke scripts, and
[`examples/brick_bar*.py`](examples/) benchmarks — see
[Verify your install](https://openseespy-solvers.readthedocs.io/en/latest/installation/#verify-your-install).

## Recommended solvers

| Analysis | GPU (CUDA) | CPU |
|----------|------------|-----|
| Linear (`PythonSparse`, static/transient/…) | `nvmath.direct_solver` | `scipy.spsolve` or `scipy.umfpack` |
| Eigen | `cupy.eigsh` | `scipy.eigsh` |

These `PythonSparse` backends are **often faster** than native OpenSees sparse/direct/eigen
solvers on medium-to-large models because they use optimized library implementations.
Details: [recommended solvers](https://openseespy-solvers.readthedocs.io/en/latest/recommended-solvers/).

## Example

```python
from openseespy_solvers import hybrid
from openseespy_solvers.scipy import spsolve

solver = hybrid(spsolve(), rtol=1e-6, restart=50)
ops.system("PythonSparse", solver.to_openseespy())
ops.analyze(n)
```

For Newton or transient steps where the tangent changes slowly, ``hybrid`` factorizes once
then reuses that factorization as a GMRES preconditioner until the system size changes or
GMRES fails to converge.

## Submodules

| Module | Functions |
|--------|-----------|
| `openseespy_solvers.scipy` | `spsolve`, `umfpack`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| `openseespy_solvers.scipy.precond` | `jacobi`, `ilu`, `direct` |
| `openseespy_solvers.cupy` | `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg` (GPU) |
| `openseespy_solvers.cupy.precond` | `jacobi`, `ilu`, `direct` |
| `openseespy_solvers.nvmath` | `direct_solver` (GPU) |
| `openseespy_solvers.hybrid` | `hybrid(direct=...)` — frozen factorization + GMRES |

Factory signatures match [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
and [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html);
`A` and `b` are supplied by OpenSees at solve time.

See the [tutorial](https://openseespy-solvers.readthedocs.io/en/latest/getting-started/) and
[API reference](https://openseespy-solvers.readthedocs.io/en/latest/api/scipy/) for details.

## GitHub

Repository: [github.com/gaaraujo/openseespy-solvers](https://github.com/gaaraujo/openseespy-solvers).
Report bugs and ask questions via
[issues](https://github.com/gaaraujo/openseespy-solvers/issues); send contributions
(pull requests) there as well.

## License

BSD 3-Clause — see [LICENSE](LICENSE).