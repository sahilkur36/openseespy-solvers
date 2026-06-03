"""Parsing of OpenSees PythonSparse buffers into NumPy arrays.

These helpers are backend-agnostic: they only turn the raw memoryviews OpenSees
passes into NumPy views. Building the actual sparse matrix (SciPy, CuPy, ...) is
the job of each namespace's solver, via its ``_build_matrix`` hook.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from openseespy_solvers.exceptions import InvalidOpenSeesDataError, UnsupportedStorageSchemeError


class SparseArrays(NamedTuple):
    """Parsed sparse matrix arrays plus shape/format metadata."""

    values: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]
    fmt: str


def parse_sparse_arrays(
    kwargs: dict[str, Any],
    *,
    values_key: str = "values",
) -> SparseArrays:
    """Parse a single sparse matrix's buffers from OpenSees kwargs."""
    storage_scheme = kwargs.get("storage_scheme", "CSR")
    num_eqn = int(kwargs["num_eqn"])
    nnz = int(kwargs["nnz"])

    if values_key not in kwargs:
        raise InvalidOpenSeesDataError(f"Missing buffer: {values_key}")
    values = np.frombuffer(kwargs[values_key], dtype=np.float64, count=nnz)

    if storage_scheme in ("CSR", "CSC"):
        if "index_ptr" not in kwargs or "indices" not in kwargs:
            raise InvalidOpenSeesDataError("CSR/CSC requires index_ptr and indices buffers")
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        indices = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        return SparseArrays(values, indices, indptr, (num_eqn, num_eqn), storage_scheme)

    raise UnsupportedStorageSchemeError(
        f"Storage scheme {storage_scheme!r} is not supported in this version."
    )


def parse_eigen_values(kwargs: dict[str, Any]) -> np.ndarray:
    """Parse the mass-matrix coefficient buffer for an eigen solve."""
    nnz = int(kwargs["nnz"])
    if "m_values" not in kwargs:
        raise InvalidOpenSeesDataError("Missing buffer: m_values")
    return np.frombuffer(kwargs["m_values"], dtype=np.float64, count=nnz)
