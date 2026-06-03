"""Import and namespace tests."""

from __future__ import annotations

import importlib

import pytest

import openseespy_solvers
from openseespy_solvers.exceptions import BackendNotAvailableError


def test_version() -> None:
    assert openseespy_solvers.__version__ == "0.1.0"


def test_scipy_namespace_exports() -> None:
    scipy_ns = importlib.import_module("openseespy_solvers.scipy")
    assert set(scipy_ns.__all__) == {"cg", "gmres", "spsolve", "eigsh", "lobpcg"}


def test_lazy_scipy_attribute() -> None:
    assert openseespy_solvers.scipy is not None


def test_cupy_namespace_import() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(BackendNotAvailableError):
            importlib.import_module("openseespy_solvers.cupy")
    else:
        importlib.import_module("openseespy_solvers.cupy")


def test_cupy_has_no_eigsh() -> None:
    pytest.importorskip("cupy")
    cupy_ns = importlib.import_module("openseespy_solvers.cupy")
    assert "eigsh" not in cupy_ns.__all__
    assert not hasattr(cupy_ns, "eigsh")
