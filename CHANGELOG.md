# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documentation: [Recommended solvers](recommended-solvers.md) (CPU/GPU defaults, performance notes vs native OpenSees); installation quick-start recipes.
- CuPy `eigsh` for `K x = λ M x` with shift-invert (`mass_mode`: `diagonal`, `lumped`, `general`).
- nvMath `direct_solver` factory: NVIDIA `nvmath.sparse.advanced.DirectSolver` with plan/factorize/solve caching; namespace `openseespy_solvers.nvmath`; optional `[nvmath]` extra (`nvmath-python[cpu]>=0.8.0`). CPU execution uses SciPy sparse matrices; GPU execution also requires `[cupy]` and a GPU-capable nvmath-python install (e.g. `nvmath-python[cu12-dx]`).
- SciPy `umfpack` solver: 64-bit UMFPACK direct solver (`scikits.umfpack.UmfpackContext("dl")`) with cached symbolic/numeric factorization; optional `[umfpack]` extra (`scikit-umfpack>=0.3.3`).

### Changed

- CuPy `eigsh` default `mass_mode` is `general` (was `diagonal`).
- `nvmath.direct_solver` default `device` is `'gpu'` (was auto CPU when CuPy missing).
- CuPy `eigsh` `diagonal` / `lumped` use GPU shift-invert with cached diagonal mass (replaces mass-normalized Lanczos).
- nvMath documented and packaged as a **GPU** backend (like CuPy): install `nvmath-python[cu12]` or `[cu13]` manually; removed `[nvmath]` pip extra.
- nvMath factory named `direct_solver` (not `spsolve`) to mirror `nvmath.sparse.advanced.direct_solver` / `DirectSolver`; SciPy and CuPy backends keep `spsolve` for their respective `scipy.sparse.linalg.spsolve` / `cupyx.scipy.sparse.linalg.spsolve` wrappers.
- Rename `.to_opensees()` to `.to_openseespy()` on solver objects (breaking API change).
- Require Python ≥ 3.12, NumPy ≥ 1.26, SciPy ≥ 1.12; optional CuPy ≥ 13 via `[cupy]`.
- Rename optional extra `[gpu]` → `[cupy]` (one extra per backend).
- Simplify examples to two brick-bar solver-comparison scripts (`brick_bar.py` static, `brick_bar_eigen.py` eigen); remove the RC frame and shell examples.

## [0.1.0] - 2026-06-03

### Added

- Per-backend namespaces `openseespy_solvers.scipy` and `openseespy_solvers.cupy`.
- SciPy solvers: `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`.
- CuPy solvers: `spsolve`, `cg`, `gmres`, `lobpcg` (no generalized `eigsh`).
- `.to_opensees()` helper for OpenSeesPy `PythonSparse` system and eigen commands.
- `formAp` matrix-vector product on all linear solvers.
- `copy.copy` support via `__copy__` / `__deepcopy__` for OpenSees SOE cloning.
- Exposed solver arrays: `A`, `b`, `x` (linear) and `K`, `M` (eigen).
- Preconditioner factories: `M=` accepts a ready operator or callable `M(A)`.
- Built-in SciPy preconditioners in `openseespy_solvers.scipy.precond` (`jacobi`, `ilu`).
- Solver statistics via `.stats`.
- Examples and tests that do not require OpenSeesPy.
