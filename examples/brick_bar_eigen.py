"""3-D brick bar — compare PythonSparse against native OpenSees eigen solvers.

Sweeps a few mesh sizes and times an eigenvalue analysis with each solver, in
the style of the OpenSees SolverBenchmark examples. Verification uses a two-tier
reference: ``genBandArpack`` first; ``fullGenLapack`` only when they disagree
(except ``cupy.eigsh.lumped``, which uses a different mass matrix).

Run with ``--large-test`` for a finer mesh sweep (more equations, slower).
"""

import argparse
import math
import os
import sys
import time

print("==========================")
print("Start BrickBar Eigen Example")

_SOLVERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solvers")
if _SOLVERS not in sys.path:
    sys.path.insert(0, _SOLVERS)

import openseespy.opensees as ops

import _brick_common as brick

NUM_MODES = 5
DEFAULT_TIME_LIMIT = 120.0
BUDGET_SKIP_FRACTION = brick.BUDGET_SKIP_FRACTION

FAST_REFERENCE = brick.FAST_EIGEN_SOLVER
TRUSTED_REFERENCE = brick.TRUSTED_EIGEN_SOLVER

# --- model (kip, in, sec) ---
BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
FAR_CORNER = (BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0)


def mesh_counts(factor):
    mesh_size = BAR_THICKNESS / factor
    nx = max(1, int(math.ceil(BAR_LENGTH / mesh_size)))
    ny = max(1, int(math.ceil(BAR_THICKNESS / mesh_size)))
    nz = max(1, int(math.ceil(BAR_HEIGHT / mesh_size)))
    return nx, ny, nz


def build_model(nx, ny, nz):
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, ELASTIC_MODULUS, POISSON_RATIO, STEEL_DENSITY)
    ops.block3D(
        nx,
        ny,
        nz,
        1,
        1,
        "stdBrick",
        1,
        1, 0.0, -BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        2, BAR_LENGTH, -BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        3, BAR_LENGTH, BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        4, 0.0, BAR_THICKNESS / 2.0, -BAR_HEIGHT / 2.0,
        5, 0.0, -BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        6, BAR_LENGTH, -BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        7, BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
        8, 0.0, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0,
    )
    ops.fixX(0.0, 1, 1, 1)


def far_corner_node():
    for node in ops.getNodeTags():
        x = ops.nodeCoord(node, 1)
        y = ops.nodeCoord(node, 2)
        z = ops.nodeCoord(node, 3)
        if (
            math.isclose(x, FAR_CORNER[0], abs_tol=1e-9)
            and math.isclose(y, FAR_CORNER[1], abs_tol=1e-9)
            and math.isclose(z, FAR_CORNER[2], abs_tol=1e-9)
        ):
            return node
    return None


def run_benchmark(mesh_factors, *, time_limit=DEFAULT_TIME_LIMIT, table=None):
    pythonsparse_solvers = brick.pythonsparse_eigen_solvers()
    results = []
    far_node = None
    mode_shape = None
    skip_remaining: set[str] = set()

    def record(row):
        results.append(row)
        if table is not None:
            table.add_row(*row)

    for factor in mesh_factors:
        nx, ny, nz = mesh_counts(factor)

        if FAST_REFERENCE in skip_remaining:
            build_model(nx, ny, nz)
            equations = ops.systemSize()
            far_node = far_corner_node()
            ref_ev = None
            ref_mode = None
            ref_mode_last = None
            record((factor, equations, FAST_REFERENCE, -2, 0.0, float("nan")))
        else:
            build_model(nx, ny, nz)
            far_node = far_corner_node()
            start = time.perf_counter()
            ref_ev = ops.eigen(FAST_REFERENCE, NUM_MODES)
            ref_seconds = time.perf_counter() - start
            ref_status = 0 if ref_ev else -1
            if ref_status == 0:
                ref_status = brick.budget_record_status(
                    FAST_REFERENCE,
                    ref_seconds,
                    time_limit,
                    factor=factor,
                    skip_remaining=skip_remaining,
                )
            equations = ops.systemSize()
            ref_mode = ops.nodeEigenvector(far_node, 1) if far_node is not None and ref_ev else None
            ref_mode_last = (
                ops.nodeEigenvector(far_node, NUM_MODES)
                if far_node is not None and ref_ev
                else None
            )
            ref_first = ref_ev[0] if ref_ev else float("nan")
            record((factor, equations, FAST_REFERENCE, ref_status, ref_seconds, ref_first))

        for label, solver in pythonsparse_solvers:
            if label in skip_remaining:
                record((factor, equations, label, -2, 0.0, float("nan")))
                continue

            ev_rel_tol, vec_rel_tol = brick.eigen_verify_tolerances(solver)

            build_model(nx, ny, nz)
            far_node = far_corner_node()
            start = time.perf_counter()
            try:
                py_ev = ops.eigen("PythonSparse", NUM_MODES, solver.to_openseespy())
            except Exception as exc:
                py_seconds = time.perf_counter() - start
                print(f"  Mesh {factor}: {label} failed: {exc}")
                record((factor, equations, label, -1, py_seconds, float("nan")))
                continue
            py_seconds = time.perf_counter() - start
            py_status = 0 if py_ev else -1
            if py_status == 0:
                py_status = brick.budget_record_status(
                    label,
                    py_seconds,
                    time_limit,
                    factor=factor,
                    skip_remaining=skip_remaining,
                )
            if py_status == -2:
                record((factor, equations, label, py_status, py_seconds, float("nan")))
                continue

            mode_shape = ops.nodeEigenvector(far_node, 1) if far_node is not None and py_ev else None
            py_first = py_ev[0] if py_ev else float("nan")

            if ref_ev is None:
                record((factor, equations, label, py_status, py_seconds, py_first))
                continue

            match_fast = (
                py_ev
                and brick.eigenvalues_close(ref_ev, py_ev, rel_tol=ev_rel_tol)
                and brick.eigenvector_close(ref_mode, mode_shape, rel_tol=vec_rel_tol)
            )
            if match_fast:
                pass
            elif brick.eigen_skips_trusted_reference(label):
                py_mode_last = (
                    ops.nodeEigenvector(far_node, NUM_MODES)
                    if far_node is not None and py_ev
                    else None
                )
                brick.print_eigen_lumped_approx_report(
                    factor,
                    label,
                    ref_ev,
                    py_ev,
                    ref_mode_first=ref_mode,
                    ref_mode_last=ref_mode_last,
                    py_mode_first=mode_shape,
                    py_mode_last=py_mode_last,
                    reference=FAST_REFERENCE,
                )
            elif TRUSTED_REFERENCE in skip_remaining:
                print(
                    f"  Mesh {factor}: {label} mismatch vs {FAST_REFERENCE}; "
                    f"{TRUSTED_REFERENCE} skipped on finer meshes"
                )
            else:
                build_model(nx, ny, nz)
                far_node = far_corner_node()
                start = time.perf_counter()
                trusted_ev = ops.eigen(TRUSTED_REFERENCE, NUM_MODES)
                trusted_seconds = time.perf_counter() - start
                trusted_status = 0 if trusted_ev else -1
                if trusted_status == 0:
                    trusted_status = brick.budget_record_status(
                        TRUSTED_REFERENCE,
                        trusted_seconds,
                        time_limit,
                        factor=factor,
                        skip_remaining=skip_remaining,
                    )
                trusted_mode = (
                    ops.nodeEigenvector(far_node, 1) if far_node is not None and trusted_ev else None
                )
                trusted_first = trusted_ev[0] if trusted_ev else float("nan")
                record(
                    (
                        factor,
                        ops.systemSize(),
                        TRUSTED_REFERENCE,
                        trusted_status,
                        trusted_seconds,
                        trusted_first,
                    )
                )
                match_trusted = (
                    py_ev
                    and trusted_ev
                    and brick.eigenvalues_close(trusted_ev, py_ev, rel_tol=ev_rel_tol)
                    and brick.eigenvector_close(trusted_mode, mode_shape, rel_tol=vec_rel_tol)
                )
                if match_trusted:
                    print(
                        f"  Mesh {factor}: {label} matched {TRUSTED_REFERENCE} "
                        f"(tiebreaker; differed from {FAST_REFERENCE})"
                    )
                else:
                    print(
                        f"  Mesh {factor}: {label} mismatch vs {FAST_REFERENCE} "
                        f"and {TRUSTED_REFERENCE}"
                    )

            record((factor, equations, label, py_status, py_seconds, py_first))

    return pythonsparse_solvers, results, far_node, mode_shape


def main():
    parser = argparse.ArgumentParser(description="Brick bar eigen solver benchmark.")
    parser.add_argument(
        "--large-test",
        action="store_true",
        help="use a finer mesh sweep (more equations; much slower)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=(
            "per-run time budget in seconds (default: 120); does not interrupt "
            "OpenSees—marks status -2 after a slow run and skips finer meshes "
            f"when a run exceeds {BUDGET_SKIP_FRACTION:.0%} of the budget"
        ),
    )
    args = parser.parse_args()

    if args.large_test:
        mesh_factors = brick.LARGE_EIGEN_MESH_FACTORS
        sweep_label = "large mesh sweep (--large-test)"
    else:
        mesh_factors = brick.DEFAULT_EIGEN_MESH_FACTORS
        sweep_label = "default mesh sweep"

    pythonsparse_solvers = brick.pythonsparse_eigen_solvers()
    solver_names = (
        [FAST_REFERENCE]
        + [label for label, _ in pythonsparse_solvers]
        + [TRUSTED_REFERENCE]
    )

    print()
    print(f"Eigen analysis — {sweep_label}; mesh factors: {mesh_factors}")
    print(
        f"Primary reference: {FAST_REFERENCE}; tiebreaker: {TRUSTED_REFERENCE} "
        "(on mismatch only, except cupy.eigsh.lumped)."
    )
    print(
        f"Per-run time budget: {args.time_limit:.1f}s (not a hard timeout; "
        f"skip finer meshes after >{BUDGET_SKIP_FRACTION:.0%} of budget)"
    )
    print("PythonSparse solvers:", ", ".join(label for label, _ in pythonsparse_solvers))
    print()
    table = brick.EigenBenchmarkTable(solver_names)
    _, results, far_node, mode_shape = run_benchmark(
        mesh_factors, time_limit=args.time_limit, table=table
    )

    if far_node is not None and mode_shape is not None:
        print()
        print("Mode 1 eigenvector at far corner:", mode_shape)

    print()
    print("Passed!" if all(status == 0 for _, _, _, status, _, _ in results) else "Failed!")
    print("==========================")


if __name__ == "__main__":
    main()
