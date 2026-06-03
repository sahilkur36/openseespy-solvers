"""SciPy backend hooks shared by all SciPy-namespace solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers.exceptions import UnsupportedStorageSchemeError


class ScipyMixin:
    """Implements the backend hooks using SciPy / NumPy (CPU)."""

    backend = "scipy"
    _on_device = False

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> sp.spmatrix:
        if fmt == "CSR":
            return sp.csr_matrix((values.copy(), indices.copy(), indptr.copy()), shape=shape)
        if fmt == "CSC":
            return sp.csc_matrix((values.copy(), indices.copy(), indptr.copy()), shape=shape)
        raise UnsupportedStorageSchemeError(f"SciPy backend does not support scheme {fmt!r}")

    def _update_matrix(self, matrix: sp.spmatrix, values: np.ndarray) -> sp.spmatrix:
        matrix.data[:] = np.asarray(values, dtype=self._compute_dtype)
        return matrix

    def _to_device(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=self._compute_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        return np.asarray(array, dtype=self._compute_dtype)

    def _matvec(self, matrix: sp.spmatrix, vector: Any) -> np.ndarray:
        return matrix @ np.asarray(vector, dtype=self._compute_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return sp.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        return isinstance(obj, spla.LinearOperator)
