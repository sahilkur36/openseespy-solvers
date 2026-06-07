# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
