# Examples

## Layout

| Location | Purpose |
|----------|---------|
| [`brick_bar.py`](brick_bar.py) | Benchmark-style static sweep vs native OpenSees solvers |
| [`brick_bar_eigen.py`](brick_bar_eigen.py) | Benchmark-style eigen sweep vs native OpenSees solvers |
| [`solvers/`](solvers/) | **One script per solver factory** (small mesh, fast smoke test) |
| [`solvers/_brick_common.py`](solvers/_brick_common.py) | Shared brick-bar model helpers (not run directly) |

We use **one script per factory** under `solvers/` so each backend stays easy to find,
copy, and run. The benchmark scripts stay separate for timing sweeps.

## Quick start

Clone the repo and follow [Full setup](../docs/installation.md#full-setup) (environment →
install → [verify](../docs/installation.md#verify)). Then from `examples/`:

```bash
# Benchmark comparisons (mesh sweeps)
python brick_bar.py
python brick_bar_eigen.py

# Finer meshes, more equations (much slower)
python brick_bar.py --large-test
python brick_bar_eigen.py --large-test

# Single-solver smoke tests
python solvers/scipy_spsolve.py
python solvers/scipy_eigsh.py
```

Optional backends (UMFPACK, nvMath, CuPy): see the
[installation guide](../docs/installation.md).

## Solver catalog (`solvers/`)

Each script builds a small 3-D brick bar, runs one analysis, and prints **Passed!** /
**Failed!**. All use `solver.to_openseespy()` with `ops.system("PythonSparse", ...)`.

Eigen scripts and `brick_bar_eigen` use **two-tier verification**: compare PythonSparse to
OpenSees **`genBandArpack`** first; if results disagree, fall back to **`fullGenLapack`**
(dense tiebreaker). The benchmark sweep uses only recommended eigen solvers. Experimental
`lobpcg` scripts use a larger mesh and are **manual-only** (not in pytest smoke tests).
Use the default mesh sweep for CI; add `--large-test` only when you want heavier runs.

### SciPy (`openseespy_solvers.scipy`)

| Script | Factory |
|--------|---------|
| [`scipy_spsolve.py`](solvers/scipy_spsolve.py) | `spsolve()` |
| [`scipy_umfpack.py`](solvers/scipy_umfpack.py) | `umfpack()` — needs `[umfpack]` |
| [`hybrid_spsolve.py`](solvers/hybrid_spsolve.py) | `hybrid(spsolve(), ...)` |
| [`scipy_cg.py`](solvers/scipy_cg.py) | `cg()` |
| [`scipy_cg_jacobi.py`](solvers/scipy_cg_jacobi.py) | `cg(M=precond.jacobi)` |
| [`scipy_gmres.py`](solvers/scipy_gmres.py) | `gmres()` |
| [`scipy_gmres_ilu.py`](solvers/scipy_gmres_ilu.py) | `gmres(M=precond.ilu)` |
| [`scipy_eigsh.py`](solvers/scipy_eigsh.py) | `eigsh()` |
| [`scipy_lobpcg.py`](solvers/scipy_lobpcg.py) | `lobpcg()` |

Preconditioner factories live in `openseespy_solvers.scipy.precond`: **`jacobi`**, **`ilu`**,
**`direct`** (used via `M=` on `cg` / `gmres` / `lobpcg`, as in the scripts above).

### CuPy (`openseespy_solvers.cupy`)

| Script | Factory |
|--------|---------|
| [`cupy_spsolve.py`](solvers/cupy_spsolve.py) | `spsolve()` |
| [`cupy_cg.py`](solvers/cupy_cg.py) | `cg()` |
| [`cupy_cg_jacobi.py`](solvers/cupy_cg_jacobi.py) | `cg(M=precond.jacobi)` |
| [`cupy_gmres.py`](solvers/cupy_gmres.py) | `gmres()` |
| [`cupy_gmres_ilu.py`](solvers/cupy_gmres_ilu.py) | `gmres(M=precond.ilu)` |
| [`cupy_eigsh.py`](solvers/cupy_eigsh.py) | `eigsh` (default `mass_mode="general"`) |
| [`cupy_lobpcg.py`](solvers/cupy_lobpcg.py) | `lobpcg()` |

`openseespy_solvers.cupy.precond` exports **`jacobi`**, **`ilu`**, and **`direct`**.
[`eigsh`](solvers/cupy_eigsh.py) uses shift-invert (`diagonal` / `lumped` on GPU;
`general` with full mass).

### nvMath (`openseespy_solvers.nvmath`) — GPU

| Script | Factory |
|--------|---------|
| [`nvmath_direct_solver.py`](solvers/nvmath_direct_solver.py) | `direct_solver()` — needs `cupy-cuda12x`/`cupy-cuda13x` and matching `nvmath-python[cu12]`/`[cu13]` |

## Benchmark scripts

Change the mesh sweep in one line (bigger number = finer mesh):

```python
MESH_FACTORS = [1.5, 2.0, 2.5, 3.0]
```

Benchmarks compare [recommended solvers](../docs/recommended-solvers.md) to native
OpenSees backends. **Static** (`brick_bar.py`): `scipy.spsolve`, optional
`scipy.umfpack`, optional `nvmath.direct_solver` (GPU), vs `BandGeneral`,
`SuperLU`, `UmfPack`. **Eigen** (`brick_bar_eigen.py`): `scipy.eigsh`, optional
`cupy.eigsh`, vs `genBandArpack` (`fullGenLapack` tiebreaker on mismatch). Other
factories remain in `solvers/` as smoke tests only.

String-based solver loop (no lambdas):

```python
from openseespy_solvers.scipy import spsolve

solver = spsolve()
NATIVE_SOLVERS = brick.NATIVE_STATIC_SOLVERS  # BandGeneral, SuperLU, UmfPack

for factor in MESH_FACTORS:
    nx, ny, nz = mesh_counts(factor)
    for name in solvers:
        build_model(nx, ny, nz)
        apply_load()
        if name == "PythonSparse":
            ops.system("PythonSparse", solver.to_openseespy())
        else:
            ops.system(name)
        # numberer / constraints / integrator / test / algorithm / analysis ...
        status = ops.analyze(NUM_STEPS)
```

## See also

- OpenSees [`benchmark_python_sparse.py`](https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/SolverBenchmark/benchmark_python_sparse.py)
- OpenSees [`benchmark_python_sparse_eigen.py`](https://github.com/OpenSees/OpenSees/blob/master/EXAMPLES/SolverBenchmark/benchmark_python_sparse_eigen.py)
- Docs: [Read the Docs](https://openseespy-solvers.readthedocs.io/en/latest/examples/)
