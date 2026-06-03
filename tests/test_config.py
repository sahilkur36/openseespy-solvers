"""Tests for ``to_opensees()`` configuration."""

from __future__ import annotations

import copy

from openseespy_solvers.scipy import cg, spsolve


def test_to_opensees_linear_defaults() -> None:
    solver = spsolve()
    cfg = solver.to_opensees()
    assert cfg["solver"] is solver
    assert cfg["scheme"] == "CSR"
    assert cfg["writable"] == "none"


def test_to_opensees_overrides() -> None:
    solver = cg()
    cfg = solver.to_opensees(scheme="CSC", writable="values")
    assert cfg["scheme"] == "CSC"
    assert cfg["writable"] == "values"


def test_copy_produces_empty_clone() -> None:
    solver = cg(rtol=1e-8)
    solver.stats.num_solves = 5
    clone = copy.copy(solver)
    assert clone is not solver
    assert clone._params["rtol"] == 1e-8
    assert clone.stats.num_solves == 0
    assert clone.A is None
