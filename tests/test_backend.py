"""Tests for backend namespaces (self-contained per library)."""

from __future__ import annotations

import importlib

import pytest

import openseespy_solvers
from openseespy_solvers import scipy as scipy_ns
from openseespy_solvers.exceptions import BackendNotAvailableError


def test_scipy_solver_backend_attr() -> None:
    assert scipy_ns.cg().backend == "scipy"
    assert scipy_ns.eigsh().backend == "scipy"
    assert scipy_ns.cg()._on_device is False


def test_unknown_namespace_raises() -> None:
    with pytest.raises(AttributeError):
        openseespy_solvers.unknown_backend  # noqa: B018


def test_cupy_namespace_requires_cupy() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(BackendNotAvailableError):
            importlib.import_module("openseespy_solvers.cupy")
    else:
        cupy_ns = importlib.import_module("openseespy_solvers.cupy")
        assert cupy_ns.cg().backend == "cupy"
        assert cupy_ns.cg()._on_device is True
        assert "eigsh" not in cupy_ns.__all__
