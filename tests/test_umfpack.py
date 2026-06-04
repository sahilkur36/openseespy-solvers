"""UMFPACK direct solver tests (SciPy backend, scikit-umfpack)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers.scipy import umfpack

from conftest import csr_linear_kwargs

pytest.importorskip("scikits.umfpack")


def test_umfpack_spd_system() -> None:
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = umfpack()
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-10)
    assert sp.issparse(solver.A)


def test_umfpack_nonsymmetric_system() -> None:
    A = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 5.0]])
    b = np.array([1.0, 2.0, 3.0])
    x_expected = spla.spsolve(sp.csc_matrix(A), b)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = umfpack()
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-10)


def test_umfpack_coefficients_changed() -> None:
    A1 = sp.diags([4.0, 5.0, 6.0])
    b = np.ones(3)
    x_buf = np.zeros(3, dtype=np.float64)
    solver = umfpack()

    kwargs = csr_linear_kwargs(A1, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, b / np.array([4.0, 5.0, 6.0]), rtol=1e-10)

    A2 = sp.diags([8.0, 10.0, 12.0])
    kwargs2 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kwargs2) == 0
    np.testing.assert_allclose(x_buf, b / np.array([8.0, 10.0, 12.0]), rtol=1e-10)


def test_umfpack_unchanged_reuses_factorization() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(2, dtype=np.float64)
    solver = umfpack()

    assert solver.solve(**csr_linear_kwargs(A, b, x=x_buf)) == 0
    csc_first = solver._csc

    x_buf2 = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf2, matrix_status="UNCHANGED")
    assert solver.solve(**kwargs) == 0
    assert solver._csc is csc_first
    np.testing.assert_allclose(x_buf2, x_expected, rtol=1e-10)
