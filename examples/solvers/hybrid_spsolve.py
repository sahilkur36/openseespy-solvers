"""Brick bar static analysis — hybrid ``spsolve`` + GMRES (PythonSparse)."""

import os
import sys

print("==========================")
print("Start hybrid(spsolve) Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import openseespy.opensees as ops
from openseespy_solvers import hybrid
from openseespy_solvers.scipy import spsolve

import _brick_common as brick

solver = hybrid(spsolve(), rtol=1e-6, restart=50)
brick.build_model(ops, nx=4, ny=1, nz=2)
far_node = brick.far_corner_node(ops)
brick.apply_load(ops)
status = brick.run_static(ops, solver, steps=2)

print()
print("Equations:", ops.systemSize())
if far_node is not None:
    print("Far-corner z displacement:", ops.nodeDisp(far_node, 3))

print()
print("Passed!" if status == 0 else "Failed!")
print("==========================")
