"""Brick bar static analysis — nvMath ``direct_solver`` (GPU).

Install: ``pip install cupy-cuda13x`` and ``pip install "nvmath-python[cu13]"``
(see docs/installation.md; use ``cu12`` / ``cupy-cuda12x`` for CUDA 12.x).
"""

import os
import sys

print("==========================")
print("Start nvmath.direct_solver Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from openseespy_solvers.nvmath import direct_solver
except ImportError as exc:
    print("nvMath backend not available:", exc)
    print('Install: pip install "nvmath-python[cu13]" and cupy-cuda13x')
    raise SystemExit(1) from exc

import openseespy.opensees as ops

import _brick_common as brick

solver = direct_solver()
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
