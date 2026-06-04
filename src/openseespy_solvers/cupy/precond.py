"""Preconditioner factories for the CuPy backend.

These callables are intended for the ``M`` argument of :func:`cg`,
:func:`gmres`, and :func:`lobpcg`. Each factory accepts the assembled sparse
matrix from OpenSees and returns a preconditioner on device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openseespy_solvers._sparse import csr_linear_kwargs_from_matrix
from openseespy_solvers.cupy._base import _import_cupy


def _linear_operator_with_matmat(
    cspla: Any,
    n: int,
    dtype: Any,
    matvec: Any,
    matmat: Any,
) -> Any:
    """LOBPCG applies preconditioners to block vectors via ``matmat``."""
    return cspla.LinearOperator((n, n), matvec=matvec, matmat=matmat, dtype=dtype)


def jacobi(A: Any) -> Any:
    """Return a Jacobi (diagonal) preconditioner on GPU.

    Computes ``M = diag(1 / diag(A))``. Zero diagonal entries are left as ``1``.

    Parameters
    ----------
    A : cupyx.scipy.sparse.spmatrix
        System matrix assembled by OpenSees on device.

    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        Diagonal preconditioner (``y = inv * x`` on device) for ``M=`` in
        :func:`~openseespy_solvers.cupy.cg`.

    See Also
    --------
    nvmath
    ilu
    openseespy_solvers.cupy.cg
    openseespy_solvers.cupy.lobpcg
    openseespy_solvers.scipy.precond.jacobi

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg, lobpcg, precond
    >>> cg(M=precond.jacobi)
    >>> lobpcg(M=precond.jacobi)
    """
    cp, _csp, cspla = _import_cupy()
    diag = A.diagonal()
    inv = cp.ones_like(diag)
    mask = diag != 0
    inv[mask] = 1.0 / diag[mask]

    dtype = A.dtype
    n = A.shape[0]

    def matvec(x: Any) -> Any:
        arr = cp.asarray(x, dtype=dtype)
        if arr.ndim == 1:
            return inv * arr
        return inv[:, None] * arr

    def matmat(X: Any) -> Any:
        return inv[:, None] * cp.asarray(X, dtype=dtype)

    return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)


def nvmath(A: Any, *, device: str = "gpu") -> Any:
    """Return a preconditioner approximating ``A^{-1}`` via nvMath direct solve.

    Intended for :func:`~openseespy_solvers.cupy.lobpcg` as ``M=`` on stiffness
    ``K`` (same role as :func:`jacobi`, but with a full sparse factorization).
    Requires ``nvmath-python`` and a GPU.

    Parameters
    ----------
    A : cupyx.scipy.sparse.spmatrix
        Matrix assembled by OpenSees on device (typically stiffness ``K``).
    device : {'gpu', 'cpu'}, optional
        Passed to :func:`~openseespy_solvers.nvmath.direct_solver`.

    See Also
    --------
    jacobi
    k_inverse
    openseespy_solvers.nvmath.direct_solver
    openseespy_solvers.cupy.lobpcg

    Examples
    --------
    >>> from openseespy_solvers.cupy import lobpcg, precond
    >>> solver = lobpcg(M=precond.nvmath)
    """
    from openseespy_solvers.nvmath._base import _import_nvmath

    _import_nvmath()
    from openseespy_solvers.nvmath import direct_solver

    return k_inverse(A, linear_solver=direct_solver(device=device))


def ilu(A: Any, **opts: Any) -> Any:
    """Return an incomplete LU preconditioner as a ``LinearOperator`` on GPU.

    By default ``fill_factor=1``, which triggers CuPy's GPU ILU path with no
    fill-in and no pivoting. See
    :func:`cupyx.scipy.sparse.linalg.spilu` — only ``fill_factor=1`` runs the
    factorization on GPU; other settings delegate to SciPy on CPU.

    Parameters
    ----------
    A : cupyx.scipy.sparse.spmatrix
        System matrix assembled by OpenSees on device.
    **opts
        Keyword arguments forwarded to
        :func:`cupyx.scipy.sparse.linalg.spilu`. ``fill_factor`` defaults to
        ``1`` when omitted.

    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        Preconditioner implementing ``M @ x`` via ``solve`` on the factorization.

    See Also
    --------
    cupyx.scipy.sparse.linalg.spilu
    jacobi
    openseespy_solvers.scipy.precond.ilu

    Notes
    -----
    With ``fill_factor=1``, factorization and application stay on the GPU but
    the pattern is restricted to the sparsity structure of ``A`` (ILU(0)-like,
    no fill-in).

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg, precond
    >>> solver = cg(M=precond.ilu)
    """
    cp, _csp, cspla = _import_cupy()
    kwargs = dict(opts)
    kwargs.setdefault("fill_factor", 1)
    ilu_factor = cspla.spilu(A.tocsc(), **kwargs)

    dtype = A.dtype
    n = A.shape[0]

    def matvec(x: Any) -> Any:
        arr = cp.asarray(x, dtype=dtype)
        if arr.ndim == 1:
            return ilu_factor.solve(arr)
        return cp.column_stack([ilu_factor.solve(arr[:, j]) for j in range(arr.shape[1])])

    def matmat(X: Any) -> Any:
        X = cp.asarray(X, dtype=dtype)
        return cp.column_stack([ilu_factor.solve(X[:, j]) for j in range(X.shape[1])])

    return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)


def _default_k_inverse_solver() -> Any:
    try:
        from openseespy_solvers.nvmath._base import _import_nvmath

        _import_nvmath()
        from openseespy_solvers.nvmath import direct_solver

        return direct_solver()
    except ImportError:
        from openseespy_solvers.cupy import spsolve

        return spsolve()


def k_inverse(K: Any, *, linear_solver: Any | None = None) -> Any:
    """Return a preconditioner approximating ``K^{-1}`` via one direct solve.

    Intended for :func:`~openseespy_solvers.cupy.lobpcg` as ``M=`` (preconditioner
    for stiffness ``K``). The first application factors ``K``; later applications
    in the same eigen solve reuse the factorization when the matrix is unchanged.

    Parameters
    ----------
    K : cupyx.scipy.sparse.spmatrix
        Stiffness matrix assembled by OpenSees on device.
    linear_solver : LinearSolver, optional
        Direct solver for ``K x = b``. Defaults to
        :func:`~openseespy_solvers.nvmath.direct_solver` on GPU when nvMath is
        installed, otherwise :func:`~openseespy_solvers.cupy.spsolve`. Pass
        ``linear_solver`` explicitly to pin the backend.

    See Also
    --------
    jacobi
    nvmath
    openseespy_solvers.cupy.lobpcg

    Notes
    -----
    Prefer :func:`nvmath` or :func:`jacobi` when the backend is fixed; use this
    function when you need an explicit ``linear_solver``.
    """
    cp, _csp, cspla = _import_cupy()
    solver = linear_solver or _default_k_inverse_solver()
    needs_refactor = True
    dtype = K.dtype
    n = K.shape[0]

    def _apply_column(rhs: Any) -> Any:
        nonlocal needs_refactor
        vec = cp.asarray(rhs, dtype=dtype).ravel()
        x_host = np.zeros(n, dtype=np.float64)
        matrix_status = "STRUCTURE_CHANGED" if needs_refactor else "UNCHANGED"
        lin_kwargs = csr_linear_kwargs_from_matrix(
            K,
            cp.asnumpy(vec),
            matrix_status=matrix_status,
            x=x_host,
        )
        info = solver.solve(**lin_kwargs)
        if info != 0:
            raise RuntimeError(f"LOBPCG K-inverse preconditioner failed with info={info}")
        needs_refactor = False
        return cp.asarray(x_host, dtype=dtype)

    def matvec(x: Any) -> Any:
        arr = cp.asarray(x, dtype=dtype)
        if arr.ndim == 1:
            return _apply_column(arr)
        return cp.column_stack([_apply_column(arr[:, j]) for j in range(arr.shape[1])])

    def matmat(X: Any) -> Any:
        X = cp.asarray(X, dtype=dtype)
        return cp.column_stack([_apply_column(X[:, j]) for j in range(X.shape[1])])

    return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)
