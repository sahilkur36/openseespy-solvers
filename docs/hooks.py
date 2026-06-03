"""MkDocs hooks — build docs without optional GPU dependencies."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _mock_cupy_modules() -> None:
    """Register minimal cupy/cupyx stubs so autodoc can import the cupy namespace."""
    cp = MagicMock(name="cupy")
    cp.float64 = float
    cp.int32 = int
    cp.asarray = lambda arr, **_: arr
    cp.asnumpy = lambda arr: arr

    csp = MagicMock(name="cupyx.scipy.sparse")
    csp.csr_matrix = MagicMock()
    csp.csc_matrix = MagicMock()
    csp.issparse = lambda _: False

    cspla = MagicMock(name="cupyx.scipy.sparse.linalg")
    cspla.LinearOperator = type("LinearOperator", (), {})
    cspla.cg = MagicMock()
    cspla.gmres = MagicMock()
    cspla.spsolve = MagicMock()
    cspla.lobpcg = MagicMock()

    cupyx = MagicMock(name="cupyx")
    cupyx_scipy = MagicMock(name="cupyx.scipy")
    cupyx_scipy.sparse = csp
    cupyx_scipy.sparse.linalg = cspla

    sys.modules["cupy"] = cp
    sys.modules["cupyx"] = cupyx
    sys.modules["cupyx.scipy"] = cupyx_scipy
    sys.modules["cupyx.scipy.sparse"] = csp
    sys.modules["cupyx.scipy.sparse.linalg"] = cspla


def on_startup(**_kwargs) -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        _mock_cupy_modules()
