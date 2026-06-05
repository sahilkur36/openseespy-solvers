# Examples

Scripts live in [`examples/`](../examples/) (clone the repo — they are not in the pip
wheel). After [installing](../installation.md) optional backends, use
[Verify your install](../installation.md#verify-your-install) for copy-paste commands:
**`pytest`**, **example scripts**, and **benchmarks**.

There are two kinds of scripts:

1. **Benchmarks** — [`brick_bar.py`](../examples/brick_bar.py) and
   [`brick_bar_eigen.py`](../examples/brick_bar_eigen.py) sweep mesh sizes and compare
   PythonSparse against native OpenSees solvers.
2. **Solver catalog** — [`examples/solvers/`](../examples/solvers/) has **one runnable
   script per solver factory** (small mesh, Passed!/Failed! smoke test).

```bash
pip install -e ".[dev,opensees]"   # from repo root; add GPU/UMFPACK extras as needed
cd examples

# Benchmarks
python brick_bar.py
python brick_bar_eigen.py

# Per-solver smoke tests
python solvers/scipy_spsolve.py
python solvers/scipy_eigsh.py
```

Optional GPU backends (CuPy, nvMath) and UMFPACK: see the
[installation guide](../installation.md). For which factories to try first, see
[Recommended solvers](recommended-solvers.md) (`scipy_spsolve`, `scipy_umfpack`,
`scipy_eigsh`, `nvmath_direct_solver`, `cupy_eigsh`).

## Benchmark scripts

| Script | Analysis | Solvers compared |
|--------|----------|------------------|
| [`brick_bar.py`](../examples/brick_bar.py) | static | [Recommended](../recommended-solvers.md) PythonSparse vs `BandGeneral`, `SuperLU`, `UmfPack` |
| [`brick_bar_eigen.py`](../examples/brick_bar_eigen.py) | eigen | Recommended `scipy.eigsh` / `cupy.eigsh` vs `genBandArpack` (`fullGenLapack` tiebreaker) |

## Solver catalog (one factory per script)

| Backend | Scripts |
|---------|---------|
| SciPy | `scipy_spsolve`, `scipy_umfpack`, `hybrid_spsolve`, `scipy_cg`, `scipy_cg_jacobi`, `scipy_gmres`, `scipy_gmres_ilu`, `scipy_eigsh`, `scipy_lobpcg` (manual) |
| CuPy | `cupy_spsolve`, `cupy_cg`, `cupy_cg_jacobi`, `cupy_gmres`, `cupy_gmres_ilu`, `cupy_eigsh`, `cupy_lobpcg` (manual) |
| nvMath | `nvmath_direct_solver` |

Preconditioners (`scipy.precond` / `cupy.precond`: `jacobi`, `ilu`, `direct`) are demonstrated via
`M=` on `cg`, `gmres`, or `lobpcg` in the `*_jacobi`, `*_ilu`, and LOBPCG scripts.

**LOBPCG** (`scipy_lobpcg`, `cupy_lobpcg`) uses a larger mesh and is **manual-only** — it is
not run by `pytest` on the tiny smoke mesh. Use [`eigsh`](recommended-solvers.md) for normal
eigen work.

Every catalog script follows the same OpenSeesPy style: top-to-bottom flow,
`ops.system("PythonSparse", solver.to_openseespy())`, and a **Passed!** / **Failed!**
footer.

## One-line mesh knob (benchmarks)

```python
MESH_FACTORS = [1.5, 2.0, 2.5, 3.0]
```

## Two-tier eigen verification

Eigen examples compare `PythonSparse` to OpenSees **`genBandArpack`** first (fast banded
reference). If eigenvalues or the mode shape disagree, they fall back to **`fullGenLapack`**
(dense LAPACK tiebreaker). Catalog scripts use the same logic via
[`run_eigen_verified()`](../examples/solvers/_brick_common.py).

## String-based solver loop (benchmarks)

```python
from openseespy_solvers.scipy import spsolve

solver = spsolve()
NATIVE_SOLVERS = ["BandGeneral", "SuperLU", "UmfPack"]

for factor in MESH_FACTORS:
    nx, ny, nz = mesh_counts(factor)
    for name in solvers:
        build_model(nx, ny, nz)
        apply_load()
        if name == "PythonSparse":
            ops.system("PythonSparse", solver.to_openseespy())
        else:
            ops.system(name)
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0 / NUM_STEPS)
        ops.test("NormUnbalance", 1.0e-7, 50)
        ops.algorithm("ModifiedNewton", "-FactorOnce")
        ops.analysis("Static")
        status = ops.analyze(NUM_STEPS)
```

## See also

- [examples/README.md](../examples/README.md) — install extras and full script list
- OpenSees [`benchmark_python_sparse.py`](https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/SolverBenchmark/benchmark_python_sparse.py)
- OpenSees [`benchmark_python_sparse_eigen.py`](https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/SolverBenchmark/benchmark_python_sparse_eigen.py)
- [Tutorial](getting-started.md)
- [PythonSparse interface](user-guide/pythonsparse-interface.md)
