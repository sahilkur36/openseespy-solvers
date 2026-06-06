# Exceptions

Public exception types raised by `openseespy-solvers`.

| Exception | Description |
|-----------|-------------|
| [`OpenSeesSolverError`](#openseespy_solvers.exceptions.OpenSeesSolverError) | Base class for package exceptions |
| [`BackendNotAvailableError`](#openseespy_solvers.exceptions.BackendNotAvailableError) | Optional backend is not installed |
| [`SolverConvergenceError`](#openseespy_solvers.exceptions.SolverConvergenceError) | Iterative or eigen solver failed to converge |
| [`InvalidOpenSeesDataError`](#openseespy_solvers.exceptions.InvalidOpenSeesDataError) | OpenSeesPy supplied missing or malformed buffers |
| [`UnsupportedComputeDtypeError`](#openseespy_solvers.exceptions.UnsupportedComputeDtypeError) | Requested compute dtype is unsupported |
| [`UnsupportedStorageSchemeError`](#openseespy_solvers.exceptions.UnsupportedStorageSchemeError) | Requested sparse storage scheme is unsupported |

## Exception Reference

::: openseespy_solvers.exceptions
    options:
      show_root_heading: false
      heading_level: 3
