"""Eigen solver tests."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from openseespy_solvers.scipy import eigsh, lobpcg

from conftest import csr_eigen_kwargs


def test_eigsh_diagonal() -> None:
    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = eigsh(tol=1e-10)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 3.0, 4.0], rtol=1e-6)
    assert sp.issparse(solver.K)
    assert sp.issparse(solver.M)


def test_eigsh_to_opensees() -> None:
    solver = eigsh()
    cfg = solver.to_opensees()
    assert cfg["solver"] is solver
    assert "writable" not in cfg


def test_lobpcg_diagonal() -> None:
    n = 4
    K = sp.diags([2.0, 4.0, 6.0, 8.0])
    M = sp.eye(n)
    num_modes = 2
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = lobpcg(maxiter=40, tol=1e-8)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 4.0], rtol=1e-5)
