"""nvMath direct solver tests (optional nvmath-python backend)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from conftest import csr_linear_kwargs

pytest.importorskip("nvmath")

from openseespy_solvers.nvmath import direct_solver


def test_nvmath_find_multithreading_lib_env(tmp_path, monkeypatch) -> None:
    from openseespy_solvers.nvmath._base import _find_multithreading_lib

    lib = tmp_path / "cudss_mtlayer_vcomp140.dll"
    lib.write_bytes(b"")
    monkeypatch.setenv("CUDSS_THREADING_LIB", str(lib))
    assert _find_multithreading_lib() == str(lib.resolve())


def test_nvmath_cpu_spd_system() -> None:
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = direct_solver(sp_module=sp)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-8)
    assert solver.backend == "nvmath"
    assert sp.issparse(solver.A)


def test_nvmath_cpu_nonsymmetric_system() -> None:
    A = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 5.0]])
    b = np.array([1.0, 2.0, 3.0])
    x_expected = spla.spsolve(sp.csc_matrix(A), b)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = direct_solver(sp_module=sp)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-8)


def test_nvmath_coefficients_changed() -> None:
    A1 = sp.diags([4.0, 5.0, 6.0])
    b = np.ones(3)
    x_buf = np.zeros(3, dtype=np.float64)
    solver = direct_solver(sp_module=sp)

    kwargs = csr_linear_kwargs(A1, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, b / np.array([4.0, 5.0, 6.0]), rtol=1e-8)

    A2 = sp.diags([8.0, 10.0, 12.0])
    kwargs2 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kwargs2) == 0
    np.testing.assert_allclose(x_buf, b / np.array([8.0, 10.0, 12.0]), rtol=1e-8)


def test_nvmath_structure_changed_different_size() -> None:
    """Replan after OpenSees mesh changes (new num_eqn / sparsity pattern)."""
    b3 = np.ones(3)
    b4 = np.ones(4)
    x3 = np.zeros(3, dtype=np.float64)
    x4 = np.zeros(4, dtype=np.float64)
    solver = direct_solver(sp_module=sp)

    A1 = sp.diags([4.0, 5.0, 6.0])
    assert solver.solve(**csr_linear_kwargs(A1, b3, x=x3)) == 0
    np.testing.assert_allclose(x3, b3 / np.array([4.0, 5.0, 6.0]), rtol=1e-8)

    A2 = sp.diags([2.0, 3.0, 4.0, 5.0])
    assert (
        solver.solve(**csr_linear_kwargs(A2, b4, x=x4, matrix_status="STRUCTURE_CHANGED"))
        == 0
    )
    np.testing.assert_allclose(x4, b4 / np.array([2.0, 3.0, 4.0, 5.0]), rtol=1e-8)


def test_nvmath_unchanged_reuses_factorization() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(2, dtype=np.float64)
    solver = direct_solver(sp_module=sp)

    assert solver.solve(**csr_linear_kwargs(A, b, x=x_buf)) == 0
    assert solver._is_factorized is True

    x_buf2 = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf2, matrix_status="UNCHANGED")
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf2, x_expected, rtol=1e-8)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cupy"),
    reason="CuPy not installed",
)
def test_nvmath_gpu_spd_system() -> None:
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA device available")

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = direct_solver()
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-6)
    assert solver._on_device is True
