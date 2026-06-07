"""Brick bar eigen analysis — cupy ``lobpcg`` (experimental, GPU eigen).

Not included in the automated solver smoke suite: LOBPCG is unreliable on tiny
meshes. Run manually from ``examples/``:

    python solvers/cupy_lobpcg.py

Verification compares ``lobpcg`` against ``eigsh`` on the same mesh (see
``_lobpcg_example.py`` for shared settings).

Install: ``python -m pip install "openseespy-solvers[cuda13]"`` (or ``[cuda12]``)
"""

import os
import sys

print("==========================")
print("Start cupy.lobpcg Example")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from openseespy_solvers.cupy import eigsh, lobpcg, precond
except Exception as exc:
    print("cupy backend not available:", exc)
    print('Install: python -m pip install "openseespy-solvers[cuda13]"  # or [cuda12]')
    raise SystemExit(1) from exc

import openseespy.opensees as ops

import _brick_common as brick
import _lobpcg_example as cfg

solver = lobpcg(M=precond.jacobi, **cfg.LOBPCG_KWARGS)
reference = eigsh(**cfg.EIGSH_KWARGS)


def rebuild():
    brick.build_model(ops, *cfg.MESH)


brick.build_model(ops, *cfg.MESH)
far_node = brick.far_corner_node(ops)
status, eigenvalues, _ref_eigenvalues = brick.run_eigen_vs_reference(
    ops,
    solver,
    reference,
    cfg.NUM_MODES,
    far_node,
    rebuild,
    ev_rel_tol=cfg.EV_REL_TOL,
    vec_rel_tol=cfg.VEC_REL_TOL,
    reference_label="cupy.eigsh",
    test_label="cupy.lobpcg",
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
