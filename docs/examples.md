# Examples

Runnable scripts are in the [`examples/`](https://github.com/gaaraujo/openseespy-solvers/tree/main/examples)
directory. Install OpenSeesPy to run them:

```bash
pip install openseespy-solvers[opensees]
```

## linear_static_spsolve.py

Static analysis of a 2-D truss using [`scipy.spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve).

## linear_static_cg.py

Same model with [`scipy.cg`](api/scipy.md#openseespy_solvers.scipy.cg) and
[`precond.jacobi`](api/precond.md#openseespy_solvers.scipy.precond.jacobi).

## linear_static_cg_gpu.py

GPU counterpart using [`cupy.cg`](api/cupy.md#openseespy_solvers.cupy.cg). Requires
the `[gpu]` extra.

## modal_eigsh.py

2-D frame eigenvalue analysis with [`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh).

## Testing without OpenSeesPy

The test suite constructs synthetic OpenSees-style keyword arguments:

```bash
pip install -e ".[dev]"
pytest
```

Helpers are defined in `tests/conftest.py` (`csr_linear_kwargs`,
`csr_eigen_kwargs`, `form_ap_kwargs`).
