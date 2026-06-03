"""Preconditioner factories for the CuPy backend.

These callables are intended for the ``M`` argument of :func:`cg` and
:func:`gmres`. Each factory accepts the assembled sparse matrix ``A`` from
OpenSees and returns a preconditioner object on device.
"""

from __future__ import annotations

from typing import Any

from openseespy_solvers.cupy._base import _import_cupy


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
    ilu
    openseespy_solvers.cupy.cg
    openseespy_solvers.scipy.precond.jacobi

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg, precond
    >>> solver = cg(M=precond.jacobi)
    """
    cp, _csp, cspla = _import_cupy()
    diag = A.diagonal()
    inv = cp.ones_like(diag)
    mask = diag != 0
    inv[mask] = 1.0 / diag[mask]

    dtype = A.dtype

    def matvec(x: Any) -> Any:
        return inv * cp.asarray(x, dtype=dtype)

    n = A.shape[0]
    return cspla.LinearOperator((n, n), matvec=matvec, dtype=dtype)


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

    def matvec(x: Any) -> Any:
        return ilu_factor.solve(cp.asarray(x, dtype=dtype))

    n = A.shape[0]
    return cspla.LinearOperator((n, n), matvec=matvec, dtype=dtype)
