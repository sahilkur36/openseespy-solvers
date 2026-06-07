# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-06-08

### Added

- Brick-bar benchmark plot script (`examples/plot_brick_bar_benchmark.py`) with live
  progress, incremental CSV writes, and `--append` resume.
- Published static and eigen benchmark assets under `docs/assets/`.
- Contributor guide: [Adding a solver](adding-a-solver.md).
- Tutorial FEM theory section and MathJax support in the docs site.

### Changed

- Rewrote the [Examples](examples.md) page with mesh tables, analysis snippets,
  and timing plots (mesh sweeps through factor 11, ~91k equations).
- Normalized backend naming in solver docstrings and PyPI description.
- Docs navigation: merged API reference, workflow diagram, removed
  `recommended-solvers.md`; refreshed preconditioner guidance.
- Static benchmark: dropped native OpenSees `UmfPack` from comparisons.
- Examples: `RCM` numberer for direct solvers; benchmark scripts skip completed
  mesh/solver pairs when appending results.

### Fixed

- Example smoke tests: isolate `sys.argv` so pytest flags are not passed to
  `brick_bar.py` / `brick_bar_eigen.py` argparse.

## [0.1.1] - 2026-06-05

### Added

- `hybrid(direct=...)` — frozen direct factorization reused as a GMRES preconditioner.
- CUDA optional extras `[cuda12]` and `[cuda13]` (matching `cupy` + nvmath GPU wheels).
- Documentation: full install walkthrough, verify checklist (`pytest` / examples / benchmarks), serial OpenSeesPy note.

### Changed

- Install hints and error messages recommend `[cuda12]` / `[cuda13]` instead of generic `[cupy]`.
- PyPI metadata: description, classifiers, and project URLs.
- LOBPCG example scripts use a larger mesh and are manual-only (not pytest smoke tests).

## [0.1.0] - 2026-06-03

### Added

- Per-backend namespaces `openseespy_solvers.scipy` and `openseespy_solvers.cupy`.
- scipy solvers: `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`.
- cupy solvers: `spsolve`, `cg`, `gmres`, `lobpcg` (no generalized `eigsh`).
- `.to_opensees()` helper for OpenSeesPy `PythonSparse` system and eigen commands.
- `formAp` matrix-vector product on all linear solvers.
- `copy.copy` support via `__copy__` / `__deepcopy__` for OpenSees SOE cloning.
- Exposed solver arrays: `A`, `b`, `x` (linear) and `K`, `M` (eigen).
- Preconditioner factories: `M=` accepts a ready operator or callable `M(A)`.
- Built-in scipy preconditioners in `openseespy_solvers.scipy.precond` (`jacobi`, `ilu`).
- Solver statistics via `.stats`.
- Examples and tests that do not require OpenSeesPy.
