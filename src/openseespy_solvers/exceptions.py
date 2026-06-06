"""Exception types raised by openseespy-solvers."""

from __future__ import annotations


class OpenSeesSolverError(Exception):
    """Base class for all openseespy-solvers exceptions."""


class UnsupportedStorageSchemeError(OpenSeesSolverError, NotImplementedError):
    """Sparse storage scheme is not supported.

    Raised when ``storage_scheme`` is not ``'CSR'`` or ``'CSC'``.
    """


class SolverConvergenceError(OpenSeesSolverError):
    """Iterative or eigen solver failed to converge."""


class InvalidOpenSeesDataError(OpenSeesSolverError, ValueError):
    """OpenSees passed missing or malformed buffer arguments."""


class UnsupportedComputeDtypeError(OpenSeesSolverError, ValueError):
    """Requested compute dtype is not supported (only float32 and float64)."""


class BackendNotAvailableError(OpenSeesSolverError, ImportError):
    """Optional backend library is not installed.

    For example, importing :mod:`openseespy_solvers.cupy` without CuPy raises
    this exception. Install the CUDA 13 extra, for example
    ``python -m pip install "openseespy-solvers[cuda13]"``.
    """
