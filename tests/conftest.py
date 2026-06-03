"""Shared helpers for OpenSees-style synthetic kwargs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def csr_linear_kwargs(
    A: sp.spmatrix | np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse linear ``solve``."""
    mat = sp.csr_matrix(A)
    if x is None:
        x = np.zeros(mat.shape[0], dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    return {
        "index_ptr": memoryview(mat.indptr.astype(np.int32)),
        "indices": memoryview(mat.indices.astype(np.int32)),
        "values": memoryview(mat.data.astype(np.float64)),
        "rhs": memoryview(rhs),
        "x": memoryview(x),
        "num_eqn": mat.shape[0],
        "nnz": mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSR",
    }


def csr_eigen_kwargs(
    K: sp.spmatrix | np.ndarray,
    M: sp.spmatrix | np.ndarray,
    *,
    num_modes: int,
    find_smallest: bool = True,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse eigen ``solve``."""
    k_mat = sp.csr_matrix(K)
    m_mat = sp.csr_matrix(M)
    n = k_mat.shape[0]
    eigenvalues = np.zeros(num_modes, dtype=np.float64)
    eigenvectors = np.zeros(num_modes * n, dtype=np.float64)
    return {
        "index_ptr": memoryview(k_mat.indptr.astype(np.int32)),
        "indices": memoryview(k_mat.indices.astype(np.int32)),
        "k_values": memoryview(k_mat.data.astype(np.float64)),
        "m_values": memoryview(m_mat.data.astype(np.float64)),
        "eigenvalues": memoryview(eigenvalues),
        "eigenvectors": memoryview(eigenvectors),
        "num_eqn": n,
        "nnz": k_mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSR",
        "num_modes": num_modes,
        "find_smallest": find_smallest,
    }


def form_ap_kwargs(
    A: sp.spmatrix | np.ndarray,
    p: np.ndarray,
    *,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse ``formAp``."""
    mat = sp.csr_matrix(A)
    ap = np.zeros(mat.shape[0], dtype=np.float64)
    base = csr_linear_kwargs(mat, np.zeros(mat.shape[0]), matrix_status=matrix_status)
    base["p"] = memoryview(np.asarray(p, dtype=np.float64))
    base["Ap"] = memoryview(ap)
    return base
