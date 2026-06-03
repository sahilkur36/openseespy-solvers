"""Preconditioner factories for the SciPy backend.

These callables are intended for the ``M`` argument of :func:`cg` and
:func:`gmres`. Each factory accepts the assembled sparse matrix ``A`` from
OpenSees and returns a preconditioner object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def jacobi(A: sp.spmatrix) -> sp.spmatrix:
    """Return a Jacobi (diagonal) preconditioner.

    Computes ``M = diag(1 / diag(A))``. Zero diagonal entries are left as ``1``.

    Parameters
    ----------
    A : sparse matrix
        System matrix assembled by OpenSees.

    Returns
    -------
    M : scipy.sparse.spmatrix
        Diagonal preconditioner suitable for ``M=`` in :func:`~openseespy_solvers.scipy.cg`.

    See Also
    --------
    ilu
    openseespy_solvers.scipy.cg

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from openseespy_solvers.scipy import cg, precond
    >>> A = sp.diags([2.0, 3.0, 4.0])
    >>> M = precond.jacobi(A)
    >>> solver = cg(M=precond.jacobi)
    """
    diag = A.diagonal()
    inv = np.ones_like(diag, dtype=np.float64)
    nz = diag != 0.0
    inv[nz] = 1.0 / diag[nz]
    return sp.diags(inv)


def ilu(A: sp.spmatrix, **opts: Any) -> spla.LinearOperator:
    """Return an incomplete LU preconditioner as a ``LinearOperator``.

    Parameters
    ----------
    A : sparse matrix
        System matrix assembled by OpenSees.
    **opts
        Additional keyword arguments forwarded to
        :func:`scipy.sparse.linalg.spilu` (for example ``drop_tol``,
        ``fill_factor``).

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        Preconditioner implementing ``M @ x`` via forward/back substitution on
        the ILU factorization.

    See Also
    --------
    scipy.sparse.linalg.spilu
    jacobi

    Examples
    --------
    >>> from openseespy_solvers.scipy import cg, precond
    >>> solver = cg(M=lambda A: precond.ilu(A, drop_tol=1e-4))
    """
    ilu_factor = spla.spilu(A.tocsc(), **opts)

    def matvec(x: np.ndarray) -> np.ndarray:
        return ilu_factor.solve(x)

    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=matvec, dtype=A.dtype)
