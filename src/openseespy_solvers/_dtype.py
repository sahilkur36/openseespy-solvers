"""Compute precision for solver numerics (OpenSees buffers remain float64)."""

from __future__ import annotations

from typing import Any

import numpy as np

from openseespy_solvers.exceptions import UnsupportedComputeDtypeError

# OpenSees PythonSparse memoryviews are double precision.
OPENSEES_BUFFER_DTYPE = np.dtype(np.float64)

_ALLOWED_COMPUTE_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})


def resolve_compute_dtype(dtype: Any = np.float64) -> np.dtype:
    """Normalize ``dtype`` to ``float32`` or ``float64`` for internal solves.

    Parameters
    ----------
    dtype : dtype or str, optional
        ``numpy.float32``, ``numpy.float64``, ``'float32'``, ``'f32'``, etc.
        Default is ``float64``.

    Returns
    -------
    numpy.dtype
        Resolved compute dtype.

    Raises
    ------
    UnsupportedComputeDtypeError
        If *dtype* is not single- or double-precision float.
    """
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise UnsupportedComputeDtypeError(f"Invalid dtype {dtype!r}") from exc
    if resolved not in _ALLOWED_COMPUTE_DTYPES:
        raise UnsupportedComputeDtypeError(
            f"Compute dtype must be float32 or float64, got {resolved!r}. "
            "OpenSees buffers are always float64; lower precision applies inside the solver."
        )
    return resolved
