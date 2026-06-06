"""Brick bar static analysis — SciPy ``umfpack`` (optional scikit-umfpack).

Install: ``python -m pip install "openseespy-solvers[umfpack]"``
"""

import os
import sys

print("==========================")
print("Start scipy.umfpack Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from openseespy_solvers.scipy import umfpack
except ImportError as exc:
    print("umfpack backend not available:", exc)
    print('Install: python -m pip install "openseespy-solvers[umfpack]"')
    raise SystemExit(1) from exc

import openseespy.opensees as ops

import _brick_common as brick

solver = umfpack()
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
