"""Sparse linear algebra solvers for OpenSeesPy (CuPy backend).

This module provides solver objects that wrap :mod:`cupyx.scipy.sparse.linalg`
for OpenSeesPy's ``PythonSparse`` commands. Factory signatures match CuPy's
SciPy-compatible API except that ``A`` and ``b`` (or ``K`` and ``M``) are
supplied by OpenSees at solve time.

Importing this module requires CuPy (``pip install openseespy-solvers[gpu]``).

Notes
-----
``eigsh`` is not provided: :func:`cupyx.scipy.sparse.linalg.eigsh` does not
support the generalized eigenproblem ``K x = \\lambda M x``. Use :func:`lobpcg`
for GPU modal analysis.

Submodules
----------
precond
    GPU preconditioner factories for use with ``M=``.

See Also
--------
cupyx.scipy.sparse.linalg
openseespy_solvers.scipy
    CPU backend with the same interface.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from openseespy_solvers._base import EigenSolver, LinearSolver
from openseespy_solvers._docstrings import (
    _EIGEN_NOTES,
    _EIGEN_RETURNS,
    _LINEAR_NOTES,
    _LINEAR_RETURNS,
    _OPENSEES_EIGEN,
    _OPENSEES_LINEAR,
)
from openseespy_solvers.cupy._base import CupyMixin, _import_cupy

# Fail fast at import time if CuPy is unavailable.
_import_cupy()

__all__ = ["cg", "gmres", "spsolve", "lobpcg"]


class _CG(CupyMixin, LinearSolver):
    def __init__(
        self,
        x0: Any = None,
        *,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
        M: Any = None,
        callback: Any = None,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(
            scheme=scheme, writable=writable, debug=debug, preconditioner=M, dtype=dtype
        )
        self._x0 = x0
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback
        self._params = {
            "x0": x0,
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        count = {"n": 0}
        callback = None
        if self._callback is not None:

            def callback(_xk=None):  # noqa: ANN001
                count["n"] += 1
                self._callback()

        kwargs = self._iterative_kwargs(
            self._cspla.cg,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=M,
            callback=callback,
        )
        if self._x0 is not None:
            kwargs["x0"] = self._to_device(np.asarray(self._x0, dtype=self._compute_dtype))
        result, info = self._cspla.cg(A, b, **kwargs)
        return result, int(info), (count["n"] or None)


class _GMRES(CupyMixin, LinearSolver):
    def __init__(
        self,
        x0: Any = None,
        *,
        rtol: float = 1e-5,
        atol: float = 0.0,
        restart: int = 20,
        maxiter: int | None = None,
        M: Any = None,
        callback: Any = None,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(
            scheme=scheme, writable=writable, debug=debug, preconditioner=M, dtype=dtype
        )
        self._x0 = x0
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._params = {
            "x0": x0,
            "rtol": rtol,
            "atol": atol,
            "restart": restart,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        count = {"n": 0}
        callback = None
        if self._callback is not None:

            def callback(_xk=None):  # noqa: ANN001
                count["n"] += 1
                self._callback()

        kwargs = self._iterative_kwargs(
            self._cspla.gmres,
            rtol=self._rtol,
            atol=self._atol,
            maxiter=self._maxiter,
            M=M,
            callback=callback,
        )
        if "restart" in inspect.signature(self._cspla.gmres).parameters:
            kwargs["restart"] = self._restart
        if self._x0 is not None:
            kwargs["x0"] = self._to_device(np.asarray(self._x0, dtype=self._compute_dtype))
        result, info = self._cspla.gmres(A, b, **kwargs)
        return result, int(info), (count["n"] or None)


class _SpSolve(CupyMixin, LinearSolver):
    def __init__(
        self,
        *,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(
            scheme=scheme, writable=writable, debug=debug, preconditioner=None, dtype=dtype
        )
        self._params = {"scheme": scheme, "writable": writable, "debug": debug, "dtype": dtype}

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        return self._cspla.spsolve(A.tocsr(), b), 0, None


class _Lobpcg(CupyMixin, EigenSolver):
    def __init__(
        self,
        *,
        X: Any = None,
        M: Any = None,
        tol: float | None = None,
        maxiter: int = 20,
        largest: bool = False,
        rng: Any = None,
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
        self._X = X
        self._M = M
        self._tol = tol
        self._maxiter = maxiter
        self._largest = largest
        self._rng = rng
        self._params = {
            "X": X,
            "M": M,
            "tol": tol,
            "maxiter": maxiter,
            "largest": largest,
            "rng": rng,
            "scheme": scheme,
            "debug": debug,
            "dtype": dtype,
        }

    def _solve_eigen(self, K, M, *, num_modes, find_smallest):  # noqa: ANN001
        cp = self._cp
        X = self._X
        if X is None:
            host = np.random.default_rng(self._rng).standard_normal(
                (K.shape[0], num_modes), dtype=self._compute_dtype
            )
            X = cp.asarray(host, dtype=self._cupy_dtype)
        else:
            X = cp.asarray(X, dtype=self._cupy_dtype)
        kwargs: dict[str, Any] = {
            "X": X,
            "B": M,
            "M": self._M,
            "maxiter": self._maxiter,
            "largest": self._largest,
        }
        if self._tol is not None:
            kwargs["tol"] = self._tol
        return self._cspla.lobpcg(K, **kwargs)


def spsolve(
    *,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
) -> _SpSolve:
    r"""Configure a sparse direct solver for OpenSees ``PythonSparse`` (GPU).

    Uses :func:`cupyx.scipy.sparse.linalg.spsolve` internally.

    Parameters
    ----------
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    cupyx.scipy.sparse.linalg.spsolve
    openseespy_solvers.scipy.spsolve

    Examples
    --------
    >>> from openseespy_solvers.cupy import spsolve
    >>> solver = spsolve()
    >>> solver.backend
    'cupy'
    """
    return _SpSolve(scheme=scheme or "CSR", writable=writable, debug=debug, dtype=dtype)


def cg(
    x0: Any = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Any = None,
    callback: Any = None,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
) -> _CG:
    r"""Configure a Conjugate Gradient solver for OpenSees ``PythonSparse`` (GPU).

    Uses :func:`cupyx.scipy.sparse.linalg.cg` internally to solve ``Ax = b``.

    Parameters
    ----------
    x0 : cupy.ndarray, optional
        Starting guess for the solution.
    rtol, atol : float, optional
        Relative and absolute tolerances for convergence. Defaults are
        ``rtol=1e-5`` and ``atol=0.0``.
    maxiter : int, optional
        Maximum number of iterations.
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``A``.
    callback : callable, optional
        User-supplied function called after each iteration.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    cupyx.scipy.sparse.linalg.cg
    scipy.sparse.linalg.cg
    openseespy_solvers.scipy.cg

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg
    >>> solver = cg(rtol=1e-8)
    >>> solver._on_device
    True
    """
    return _CG(
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        callback=callback,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
    )


def gmres(
    x0: Any = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    restart: int = 20,
    maxiter: int | None = None,
    M: Any = None,
    callback: Any = None,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
) -> _GMRES:
    r"""Configure a GMRES solver for OpenSees ``PythonSparse`` (GPU).

    Uses :func:`cupyx.scipy.sparse.linalg.gmres` internally.

    Parameters
    ----------
    x0 : cupy.ndarray, optional
        Starting guess for the solution.
    rtol, atol : float, optional
        Relative and absolute tolerances. Defaults are ``rtol=1e-5`` and
        ``atol=0.0``.
    restart : int, optional
        Number of iterations between restarts. Default is ``20``.
    maxiter : int, optional
        Maximum number of iterations.
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``A``.
    callback : callable, optional
        User-supplied function called after each iteration.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    cupyx.scipy.sparse.linalg.gmres
    openseespy_solvers.scipy.gmres
    """
    return _GMRES(
        x0,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        callback=callback,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
    )


def lobpcg(
    *,
    X: Any = None,
    M: Any = None,
    tol: float | None = None,
    maxiter: int = 20,
    largest: bool = False,
    rng: Any = None,
    scheme: str | None = None,
    debug: bool = False,
    dtype: Any = np.float64,
) -> _Lobpcg:
    r"""Configure a LOBPCG eigenvalue solver for OpenSees ``PythonSparse`` (GPU).

    Uses :func:`cupyx.scipy.sparse.linalg.lobpcg` internally to solve
    ``K x = \lambda M x``.

    Parameters
    ----------
    X : cupy.ndarray, optional
        Initial approximation to the eigenvectors.
    M : {sparse matrix, LinearOperator}, optional
        Optional preconditioner.
    tol : float, optional
        Convergence tolerance.
    maxiter : int, optional
        Maximum number of iterations. Default is ``20``.
    largest : bool, optional
        If ``True``, compute largest eigenvalues. Default is ``False``.
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator used when ``X`` is ``None``.
    """ + _OPENSEES_EIGEN + _EIGEN_RETURNS + _EIGEN_NOTES + """
    See Also
    --------
    cupyx.scipy.sparse.linalg.lobpcg
    openseespy_solvers.scipy.lobpcg

    Notes
    -----
    This is the recommended GPU eigen solver; CuPy does not expose a
    generalized :func:`~cupyx.scipy.sparse.linalg.eigsh`.
    """
    return _Lobpcg(
        X=X,
        M=M,
        tol=tol,
        maxiter=maxiter,
        largest=largest,
        rng=rng,
        scheme=scheme or "CSR",
        debug=debug,
        dtype=dtype,
    )
