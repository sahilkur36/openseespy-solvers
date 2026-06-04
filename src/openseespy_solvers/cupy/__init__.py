"""Sparse linear algebra solvers for OpenSeesPy (CuPy backend).

This module provides solver objects that wrap :mod:`cupyx.scipy.sparse.linalg`
for OpenSeesPy's ``PythonSparse`` commands. Factory signatures match CuPy's
SciPy-compatible API except that ``A`` and ``b`` (or ``K`` and ``M``) are
supplied by OpenSees at solve time.

Importing this module requires CuPy (``pip install openseespy-solvers[cupy]``).

Notes
-----
Generalized :func:`cupyx.scipy.sparse.linalg.eigsh` is not available.
:func:`eigsh` supports:

- ``mass_mode='general'`` (default): shift-invert with full ``M`` via SciPy ARPACK
  plus a pluggable GPU inner :class:`~openseespy_solvers._base.LinearSolver`.
- ``mass_mode='diagonal'`` / ``'lumped'``: shift-invert on GPU with diagonal (exact or
  row-sum lumped) mass; cached ``m`` and ``K - sigma diag(m)``.

Use :func:`lobpcg` when these modes are not appropriate.

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
import scipy.sparse as sp_host
import scipy.sparse.linalg as spla_cpu

from openseespy_solvers._base import EigenSolver, LinearSolver
from openseespy_solvers._sparse import csr_linear_kwargs_from_matrix
from openseespy_solvers.exceptions import SolverConvergenceError
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

__all__ = ["cg", "gmres", "spsolve", "eigsh", "lobpcg"]


def _default_inner_linear_solver() -> LinearSolver:
    try:
        from openseespy_solvers.nvmath._base import _import_nvmath

        _import_nvmath()
        from openseespy_solvers.nvmath import direct_solver

        return direct_solver()
    except ImportError:
        return spsolve()


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
        permc_spec: str = "COLAMD",
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(
            scheme=scheme, writable=writable, debug=debug, preconditioner=None, dtype=dtype
        )
        self._permc_spec = permc_spec
        self._solve_func = None
        self._params = {
            "permc_spec": permc_spec,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        if matrix_status != "UNCHANGED" or self._solve_func is None:
            self._solve_func = self._cspla.splu(A.tocsc(), permc_spec=self._permc_spec).solve
        return self._solve_func(b), 0, None


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
        self._preconditioner = M
        self._preconditioner_cached: Any | None = None
        self._eigen_matrix_status = "STRUCTURE_CHANGED"
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

    def solve(self, **kwargs: Any) -> None:
        status = kwargs["matrix_status"]
        self._eigen_matrix_status = status
        if status != "UNCHANGED":
            self._preconditioner_cached = None
        super().solve(**kwargs)

    def _resolve_preconditioner(self, K: Any) -> Any | None:
        M = self._preconditioner
        if M is None:
            return None
        if self._is_sparse(M) or self._is_linear_operator(M):
            return M
        if callable(M):
            if self._eigen_matrix_status == "UNCHANGED" and self._preconditioner_cached is not None:
                return self._preconditioner_cached
            self._preconditioner_cached = M(K)
            return self._preconditioner_cached
        return M

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
            "B": M,
            "M": self._resolve_preconditioner(K),
            "maxiter": self._maxiter,
            "largest": not find_smallest,
        }
        if self._tol is not None:
            kwargs["tol"] = self._tol
        return self._cspla.lobpcg(K, X, **kwargs)


class _Eigsh(CupyMixin, EigenSolver):
    def __init__(
        self,
        *,
        sigma: float | None = None,
        which: str = "LM",
        mass_mode: str = "general",
        linear_solver: LinearSolver | None = None,
        v0: Any = None,
        ncv: int | None = None,
        maxiter: int | None = None,
        tol: float = 0.0,
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
        self._sigma = sigma
        self._which = which
        self._mass_mode = mass_mode
        self._inner_solver = linear_solver or _default_inner_linear_solver()
        self._v0 = v0
        self._ncv = ncv
        self._maxiter = maxiter
        self._tol = tol
        self._shifted: Any | None = None
        self._shift_sigma: float | None = None
        self._diag_m: Any | None = None
        self._inner_needs_refactor = True
        self._eigen_matrix_status = "STRUCTURE_CHANGED"
        self._params = {
            "sigma": sigma,
            "which": which,
            "mass_mode": mass_mode,
            "linear_solver": linear_solver,
            "v0": v0,
            "ncv": ncv,
            "maxiter": maxiter,
            "tol": tol,
            "scheme": scheme,
            "debug": debug,
            "dtype": dtype,
        }

    def solve(self, **kwargs: Any) -> None:
        status = kwargs["matrix_status"]
        self._eigen_matrix_status = status
        if status == "STRUCTURE_CHANGED":
            self._shifted = None
            self._shift_sigma = None
            self._diag_m = None
            self._inner_needs_refactor = True
        elif status == "COEFFICIENTS_CHANGED":
            self._shifted = None
            self._shift_sigma = None
            self._diag_m = None
            self._inner_needs_refactor = True
        super().solve(**kwargs)

    def _shift_sigma_value(self, find_smallest: bool) -> float:
        if find_smallest:
            return 0.0 if self._sigma is None else self._sigma
        if self._sigma is None:
            raise ValueError(
                "CuPy eigsh shift-invert requires sigma or request smallest modes"
            )
        return self._sigma

    def _shifted_stiffness_diagonal_mass(self, K, m, sigma: float):  # noqa: ANN001
        if (
            self._eigen_matrix_status == "UNCHANGED"
            and self._shifted is not None
            and self._shift_sigma == sigma
        ):
            return self._shifted
        shifted = K.tocsr().copy()
        diag = shifted.diagonal().copy()
        shifted.setdiag(diag - sigma * m)
        self._shifted = shifted.tocsr()
        self._shift_sigma = sigma
        self._inner_needs_refactor = True
        return self._shifted

    def _cupy_eigsh_kwargs(self, num_modes: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"k": num_modes, "which": self._which}
        if self._v0 is not None:
            kwargs["v0"] = self._to_device(np.asarray(self._v0, dtype=self._compute_dtype))
        if self._ncv is not None:
            kwargs["ncv"] = self._ncv
        if self._maxiter is not None:
            kwargs["maxiter"] = self._maxiter
        if self._tol > 0.0:
            kwargs["tol"] = self._tol
        return kwargs

    def _mass_diagonal(self, M):  # noqa: ANN001
        """Diagonal mass vector for ``diagonal`` / ``lumped`` shift-invert (cached)."""
        cp = self._cp
        mode = self._mass_mode.lower()
        if self._eigen_matrix_status == "UNCHANGED" and self._diag_m is not None:
            return self._diag_m
        if mode == "lumped":
            m = cp.asarray(M.sum(axis=1), dtype=self._cupy_dtype).ravel()
        elif mode == "diagonal":
            coo = M.tocoo()
            if bool(cp.any(coo.row != coo.col).item()):
                raise ValueError(
                    "cupy.eigsh mass_mode='diagonal' requires a diagonal mass matrix; "
                    "use mass_mode='lumped' or mass_mode='general'."
                )
            m = cp.zeros(M.shape[0], dtype=self._cupy_dtype)
            if coo.nnz > 0:
                m[coo.row] = coo.data
        else:
            raise ValueError(
                f"Unsupported mass_mode {self._mass_mode!r}. "
                "Use 'diagonal', 'lumped', or 'general'."
            )
        if bool(cp.any(m <= 0).item()):
            raise ValueError(
                "cupy.eigsh requires strictly positive mass diagonal entries "
                f"for mass_mode={mode!r}."
            )
        self._diag_m = m
        return m

    def _solve_eigen_shift_invert_diagonal_mass(
        self, K, M, *, num_modes, find_smallest
    ):  # noqa: ANN001
        """Shift-invert with diagonal mass; OP = (K - sigma diag(m))^{-1} diag(m)."""
        cp = self._cp
        cspla = self._cspla
        sigma = self._shift_sigma_value(find_smallest)
        m = self._mass_diagonal(M)
        shifted = self._shifted_stiffness_diagonal_mass(K, m, sigma)
        n = K.shape[0]

        def op_matvec(v):
            x = cp.asarray(v, dtype=self._cupy_dtype).ravel()
            return self._inner_linear_solve(shifted, m * x)

        op = cspla.LinearOperator((n, n), matvec=op_matvec, dtype=self._cupy_dtype)
        mu, z = cspla.eigsh(op, **self._cupy_eigsh_kwargs(num_modes))
        if z.ndim == 1:
            z = z[:, None]
        eigvals = 1.0 / mu + sigma
        phi = cp.stack(
            [self._inner_linear_solve(shifted, z[:, j]) for j in range(num_modes)],
            axis=1,
        )
        denom = cp.sqrt(cp.sum((phi * phi) * m[:, None], axis=0))
        phi = phi / denom
        return eigvals, phi

    def _inner_linear_solve(self, A, b):  # noqa: ANN001
        cp = self._cp
        rhs = cp.asarray(b, dtype=self._cupy_dtype).ravel()
        n = A.shape[0]
        x_host = np.zeros(n, dtype=np.float64)
        matrix_status = (
            "STRUCTURE_CHANGED" if self._inner_needs_refactor else "UNCHANGED"
        )
        lin_kwargs = csr_linear_kwargs_from_matrix(
            A, self._to_host(rhs), matrix_status=matrix_status, x=x_host
        )
        info = self._inner_solver.solve(**lin_kwargs)
        if info != 0:
            raise SolverConvergenceError(
                f"Shift-invert inner linear solve failed with info={info}"
            )
        self._inner_needs_refactor = False
        return self._to_device(
            np.frombuffer(lin_kwargs["x"], dtype=np.float64, count=n).copy()
        )

    def _host_csr(self, matrix):  # noqa: ANN001
        return sp_host.csr_matrix(
            (
                self._to_host(matrix.data),
                self._to_host(matrix.indices),
                self._to_host(matrix.indptr),
            ),
            shape=matrix.shape,
        )

    def _solve_eigen_general(self, K, M, *, num_modes, find_smallest):  # noqa: ANN001
        sigma = self._shift_sigma_value(find_smallest)

        if self._shifted is None or self._eigen_matrix_status != "UNCHANGED":
            self._shifted = K - sigma * M
            self._shift_sigma = sigma
            self._inner_needs_refactor = True

        n = K.shape[0]

        def opinv_matvec(v):
            v_gpu = self._to_device(np.asarray(v, dtype=np.float64).ravel())
            return self._to_host(self._inner_linear_solve(self._shifted, v_gpu))

        op_inv = spla_cpu.LinearOperator((n, n), matvec=opinv_matvec, dtype=np.float64)
        kwargs: dict[str, Any] = {
            "k": num_modes,
            "which": self._which,
            "sigma": sigma,
            "OPinv": op_inv,
        }
        if self._v0 is not None:
            kwargs["v0"] = np.asarray(self._v0, dtype=np.float64).ravel()
        if self._ncv is not None:
            kwargs["ncv"] = self._ncv
        if self._maxiter is not None:
            kwargs["maxiter"] = self._maxiter
        if self._tol > 0.0:
            kwargs["tol"] = self._tol
        # ARPACK runs on the CPU; inner (K - sigma M)^{-1} solves use the GPU solver.
        eigvals, eigvecs = spla_cpu.eigsh(
            self._host_csr(K), M=self._host_csr(M), **kwargs
        )
        return eigvals, eigvecs

    def _solve_eigen(self, K, M, *, num_modes, find_smallest):  # noqa: ANN001
        mode = self._mass_mode.lower()
        if mode in {"diagonal", "lumped"}:
            return self._solve_eigen_shift_invert_diagonal_mass(
                K, M, num_modes=num_modes, find_smallest=find_smallest
            )
        if mode == "general":
            return self._solve_eigen_general(
                K, M, num_modes=num_modes, find_smallest=find_smallest
            )
        raise ValueError(
            f"Unsupported mass_mode {self._mass_mode!r}. "
            "Use 'diagonal', 'lumped', or 'general'."
        )


def spsolve(
    *,
    permc_spec: str = "COLAMD",
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
) -> _SpSolve:
    r"""Configure a sparse direct solver for OpenSees ``PythonSparse`` (GPU).

    Uses :func:`cupyx.scipy.sparse.linalg.splu` internally. The LU factorization
    is reused when OpenSees reports ``matrix_status='UNCHANGED'``.

    Parameters
    ----------
    permc_spec : str, optional
        Column permutation applied during LU factorization. Passed to
        :func:`cupyx.scipy.sparse.linalg.splu`. Default is ``'COLAMD'``.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    cupyx.scipy.sparse.linalg.splu
    cupyx.scipy.sparse.linalg.spsolve
    openseespy_solvers.scipy.spsolve

    Examples
    --------
    >>> from openseespy_solvers.cupy import spsolve
    >>> solver = spsolve()
    >>> solver.backend
    'cupy'
    """
    return _SpSolve(
        permc_spec=permc_spec,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
    )


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


def eigsh(
    *,
    sigma: float | None = None,
    which: str = "LM",
    mass_mode: str = "general",
    linear_solver: LinearSolver | None = None,
    v0: Any = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    scheme: str | None = None,
    debug: bool = False,
    dtype: Any = np.float64,
) -> _Eigsh:
    r"""Configure a generalized eigen solver for OpenSees ``PythonSparse`` (GPU).

    Solves ``K x = \lambda M x`` using one of three modes:

    - ``mass_mode='general'`` (default):
      shift-invert with :func:`scipy.sparse.linalg.eigsh` and full ``M``; GPU
      inner solves on ``(K - sigma M)^{-1}``.
    - ``mass_mode='diagonal'`` / ``'lumped'``:
      shift-invert on GPU with diagonal (exact or row-sum lumped) mass; caches
      ``m`` and ``K - sigma diag(m)`` when the matrix is unchanged.

    Parameters
    ----------
    sigma : float, optional
        Shift for shift-invert mode. When OpenSees requests the smallest
        eigenvalues (``find_smallest=True``) and ``sigma`` is ``None``, a shift
        of ``0.0`` is used.
    which : {'LM', 'SM', 'LR', 'SR', 'LI', 'SI', 'LA', 'SA', 'BE'}, optional
        Which operator eigenvalues to compute. Default is ``'LM'``.
    mass_mode : {'diagonal', 'lumped', 'general'}, optional
        Eigen mode strategy. Default is ``'general'``.
    linear_solver : LinearSolver, optional
        Inner solver for shift-invert linear systems ``(K - \sigma M) w = v``
        (full ``M`` in ``general``; ``diag(m)`` in ``diagonal`` / ``lumped``).
        Defaults to
        :func:`openseespy_solvers.nvmath.direct_solver` on GPU when nvMath is
        installed, otherwise :func:`spsolve`.
    v0 : cupy.ndarray, optional
        Starting vector for ARPACK.
    ncv : int, optional
        Number of Lanczos vectors.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance. Default is ``0.0``.
    """ + _OPENSEES_EIGEN + _EIGEN_RETURNS + _EIGEN_NOTES + """
    See Also
    --------
    lobpcg
    openseespy_solvers.scipy.eigsh
    openseespy_solvers.nvmath.direct_solver

    Notes
    -----
    All modes use shift-invert. CuPy does not expose generalized
    :func:`~cupyx.scipy.sparse.linalg.eigsh`, so ``diagonal`` / ``lumped`` run
    CuPy Lanczos on ``OP = (K - sigma diag(m))^{-1} diag(m)`` while ``general``
    uses SciPy ARPACK with the same shift-invert idea and a full mass matrix.
    """
    return _Eigsh(
        sigma=sigma,
        which=which,
        mass_mode=mass_mode,
        linear_solver=linear_solver,
        v0=v0,
        ncv=ncv,
        maxiter=maxiter,
        tol=tol,
        scheme=scheme or "CSR",
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
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``K`` (approximate ``K^{-1}``). A callable is invoked
        as ``M(K)``; see :mod:`openseespy_solvers.cupy.precond`.
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
    ``M`` in LOBPCG approximates ``K^{-1}`` (not the mass matrix ``B``). Use the same
    callable preconditioner pattern as :func:`cg` / :func:`gmres``, e.g.
    ``M=precond.jacobi`` (diagonal) or ``M=precond.nvmath`` (sparse direct ``K^{-1}``).
    Use :func:`~openseespy_solvers.cupy.precond.k_inverse` only when selecting a custom
    inner direct solver.
    Prefer :func:`eigsh` for shift-invert modal analysis when that fits the mass model.
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
