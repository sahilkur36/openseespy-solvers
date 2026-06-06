"""CuPy backend hooks shared by all CuPy-namespace solvers."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from openseespy_solvers.exceptions import BackendNotAvailableError, UnsupportedStorageSchemeError


def _import_cupy() -> tuple[Any, Any, Any]:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspla
    except ImportError as exc:  # pragma: no cover - exercised only without CuPy
        raise BackendNotAvailableError(
            "The 'cupy' backend requires CuPy. Install a CUDA-matched wheel, "
            "for example: python -m pip install \"openseespy-solvers[cuda13]\""
        ) from exc
    return cp, csp, cspla


class CupyMixin:
    """Implements the backend hooks using CuPy / cupyx (GPU)."""

    backend = "cupy"
    _on_device = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cp, self._csp, self._cspla = _import_cupy()
        super().__init__(*args, **kwargs)

    @property
    def _cupy_dtype(self) -> Any:
        return self._cp.float32 if self._compute_dtype == np.float32 else self._cp.float64

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> Any:
        cp, csp = self._cp, self._csp
        data = cp.asarray(values, dtype=self._cupy_dtype)
        ind = cp.asarray(indices, dtype=cp.int32)
        ptr = cp.asarray(indptr, dtype=cp.int32)
        if fmt == "CSR":
            return csp.csr_matrix((data, ind, ptr), shape=shape)
        if fmt == "CSC":
            return csp.csc_matrix((data, ind, ptr), shape=shape)
        raise UnsupportedStorageSchemeError(f"CuPy backend does not support scheme {fmt!r}")

    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        matrix.data[:] = self._cp.asarray(values, dtype=self._cupy_dtype)
        return matrix

    def _to_device(self, array: np.ndarray) -> Any:
        return self._cp.asarray(array, dtype=self._cupy_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        return self._cp.asnumpy(array)

    def _matvec(self, matrix: Any, vector: Any) -> Any:
        return matrix @ self._cp.asarray(vector, dtype=self._cupy_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return self._csp.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        return isinstance(obj, self._cspla.LinearOperator)

    def _iterative_kwargs(
        self,
        func: Any,
        *,
        rtol: float,
        atol: float,
        maxiter: int | None,
        M: Any | None,
        callback: Any | None,
    ) -> dict[str, Any]:
        """Map SciPy-style rtol/atol onto the installed CuPy iterative API."""
        params = inspect.signature(func).parameters
        kwargs: dict[str, Any] = {"maxiter": maxiter, "M": M}
        if callback is not None:
            kwargs["callback"] = callback
        if "rtol" in params:
            kwargs["rtol"] = rtol
            if "atol" in params:
                kwargs["atol"] = atol
        else:
            kwargs["tol"] = rtol
            if "atol" in params and atol != 0.0:
                kwargs["atol"] = atol
        return kwargs
