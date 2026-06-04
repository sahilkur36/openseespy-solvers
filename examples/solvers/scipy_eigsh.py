"""Brick bar eigen analysis — SciPy ``eigsh`` (PythonSparse eigen)."""

import os
import sys

print("==========================")
print("Start scipy.eigsh Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import openseespy.opensees as ops
from openseespy_solvers.scipy import eigsh

import _brick_common as brick

NUM_MODES = 3
MESH = (3, 1, 2)
solver = eigsh(tol=0.0)


def rebuild():
    brick.build_model(ops, *MESH)


brick.build_model(ops, *MESH)
far_node = brick.far_corner_node(ops)
status, eigenvalues = brick.run_eigen_verified(
    ops, solver, NUM_MODES, far_node, rebuild, ev_rel_tol=1e-4, vec_rel_tol=0.02
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
