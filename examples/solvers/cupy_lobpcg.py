"""Brick bar eigen analysis — CuPy ``lobpcg`` (experimental, GPU eigen).

Not included in the automated solver smoke suite: LOBPCG is unreliable on tiny
meshes. Run manually from ``examples/``:

    python solvers/cupy_lobpcg.py

Install: ``pip install cupy-cuda13x`` (see docs/installation.md)
"""

import os
import sys

print("==========================")
print("Start cupy.lobpcg Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from openseespy_solvers.cupy import lobpcg, precond
except Exception as exc:
    print("CuPy backend not available:", exc)
    print("Install: pip install cupy-cuda13x")
    raise SystemExit(1) from exc

import openseespy.opensees as ops

import _brick_common as brick

NUM_MODES = 2
MESH = (8, 2, 4)
solver = lobpcg(M=precond.jacobi, tol=1e-3, maxiter=300, rng=0)


def rebuild():
    brick.build_model(ops, *MESH)


brick.build_model(ops, *MESH)
far_node = brick.far_corner_node(ops)
status, eigenvalues = brick.run_eigen_verified(
    ops, solver, NUM_MODES, far_node, rebuild, ev_rel_tol=1e-3, vec_rel_tol=0.05
)

print()
print("Equations:", ops.systemSize())
if eigenvalues:
    print("Smallest eigenvalue:", eigenvalues[0])
if far_node is not None and eigenvalues:
    print("Mode 1 eigenvector at far corner:", ops.nodeEigenvector(far_node, 1))

print()
print("Passed!" if status == 0 else "Failed!")
print("==========================")
