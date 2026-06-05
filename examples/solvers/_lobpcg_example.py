"""Shared settings for ``scipy_lobpcg.py`` and ``cupy_lobpcg.py``.

SciPy and CuPy LOBPCG examples use identical solver kwargs; each script compares
against its backend's ``eigsh`` reference on the same mesh.
"""

NUM_MODES = 2
MESH = (8, 2, 4)

# Same LOBPCG kwargs for SciPy and CuPy (``M=precond.jacobi`` is added in each script).
LOBPCG_KWARGS = {
    "tol": 1e-3,
    "maxiter": 300,
    "rng": 0,
}

# Reference eigsh settings (same style as ``scipy_eigsh.py`` / ``cupy_eigsh.py``).
EIGSH_KWARGS = {
    "tol": 0.0,
}

# Pass/fail vs the backend ``eigsh`` reference (LOBPCG is approximate).
EV_REL_TOL = 1e-3
VEC_REL_TOL = 0.05
