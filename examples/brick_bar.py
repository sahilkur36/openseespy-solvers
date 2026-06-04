"""3-D brick bar — compare PythonSparse against native OpenSees solvers (static).

Sweeps a few mesh sizes and times a small static analysis with each solver, in
the style of the OpenSees SolverBenchmark examples.

Run with ``--large-test`` for a finer mesh sweep (more equations, slower).
"""

import argparse
import math
import os
import sys
import time

print("==========================")
print("Start BrickBar Example")

_SOLVERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solvers")
if _SOLVERS not in sys.path:
    sys.path.insert(0, _SOLVERS)

import openseespy.opensees as ops

import _brick_common as brick

# Other PythonSparse solvers: see examples/solvers/ (one script per factory).

NUM_STEPS = 10
DEFAULT_TIME_LIMIT = 120.0

# Native OpenSees solvers to compare against. You can also add "ProfileSPD",
# "SparseGeneral", etc.
NATIVE_SOLVERS = ["BandGeneral", "SuperLU", "UmfPack"]

# --- model (kip, in, sec) ---
BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
YIELD_STRESS = 50.0
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
        1,
        0.0,
        -BAR_THICKNESS / 2.0,
        -BAR_HEIGHT / 2.0,
        2,
        BAR_LENGTH,
        -BAR_THICKNESS / 2.0,
        -BAR_HEIGHT / 2.0,
        3,
        BAR_LENGTH,
        BAR_THICKNESS / 2.0,
        -BAR_HEIGHT / 2.0,
        4,
        0.0,
        BAR_THICKNESS / 2.0,
        -BAR_HEIGHT / 2.0,
        5,
        0.0,
        -BAR_THICKNESS / 2.0,
        BAR_HEIGHT / 2.0,
        6,
        BAR_LENGTH,
        -BAR_THICKNESS / 2.0,
        BAR_HEIGHT / 2.0,
        7,
        BAR_LENGTH,
        BAR_THICKNESS / 2.0,
        BAR_HEIGHT / 2.0,
        8,
        0.0,
        BAR_THICKNESS / 2.0,
        BAR_HEIGHT / 2.0,
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


def apply_load():
    total = 1.25 * YIELD_STRESS * (BAR_THICKNESS * BAR_HEIGHT**2) / (6 * BAR_LENGTH)
    ops.timeSeries("Trig", 1, 0.0, 6.0, 4.0, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    far_nodes = []
    for node in ops.getNodeTags():
        if math.isclose(ops.nodeCoord(node, 1), BAR_LENGTH, abs_tol=1e-9):
            far_nodes.append(node)
    load = total / len(far_nodes)
    for node in far_nodes:
        ops.load(node, 0.0, 0.0, -load)


def run_benchmark(mesh_factors, *, time_limit=DEFAULT_TIME_LIMIT, table=None):
    pythonsparse_solvers = brick.pythonsparse_static_solvers()
    results = []
    far_node = None

    for factor in mesh_factors:
        nx, ny, nz = mesh_counts(factor)
        for label, solver in pythonsparse_solvers:
            build_model(nx, ny, nz)
            far_node = far_corner_node()
            apply_load()
            ops.system("PythonSparse", solver.to_openseespy())
            ops.numberer("RCM")
            ops.constraints("Plain")
            ops.integrator("LoadControl", 1.0 / NUM_STEPS)
            ops.test("NormUnbalance", 1.0e-7, 50)
            ops.algorithm("ModifiedNewton", "-FactorOnce")
            ops.analysis("Static")
            start = time.perf_counter()
            status = ops.analyze(NUM_STEPS)
            seconds = time.perf_counter() - start
            if seconds > time_limit:
                status = -2
                print(
                    f"  Mesh {factor}: {label} exceeded time limit "
                    f"({seconds:.3f}s > {time_limit:.1f}s)"
                )
            equations = ops.systemSize()
            row = (factor, equations, label, status, seconds)
            results.append(row)
            if table is not None:
                table.add_row(*row)
        for name in NATIVE_SOLVERS:
            build_model(nx, ny, nz)
            far_node = far_corner_node()
            apply_load()
            ops.system(name)
            ops.numberer("RCM")
            ops.constraints("Plain")
            ops.integrator("LoadControl", 1.0 / NUM_STEPS)
            ops.test("NormUnbalance", 1.0e-7, 50)
            ops.algorithm("ModifiedNewton", "-FactorOnce")
            ops.analysis("Static")
            start = time.perf_counter()
            status = ops.analyze(NUM_STEPS)
            seconds = time.perf_counter() - start
            if seconds > time_limit:
                status = -2
                print(
                    f"  Mesh {factor}: {name} exceeded time limit "
                    f"({seconds:.3f}s > {time_limit:.1f}s)"
                )
            equations = ops.systemSize()
            row = (factor, equations, name, status, seconds)
            results.append(row)
            if table is not None:
                table.add_row(*row)

    return pythonsparse_solvers, results, far_node


def main():
    parser = argparse.ArgumentParser(description="Brick bar static solver benchmark.")
    parser.add_argument(
        "--large-test",
        action="store_true",
        help="use a finer mesh sweep (more equations; much slower)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="per-run time limit in seconds (default: 120)",
    )
    args = parser.parse_args()

    if args.large_test:
        mesh_factors = brick.LARGE_STATIC_MESH_FACTORS
        sweep_label = "large mesh sweep (--large-test)"
    else:
        mesh_factors = brick.DEFAULT_STATIC_MESH_FACTORS
        sweep_label = "default mesh sweep"

    pythonsparse_solvers = brick.pythonsparse_static_solvers()
    solver_names = [label for label, _ in pythonsparse_solvers] + NATIVE_SOLVERS

    print()
    print(f"Static analysis — {sweep_label}; mesh factors: {mesh_factors}")
    print(
        f"Running {NUM_STEPS} steps of LoadControl with dLambda = 1 / {NUM_STEPS}"
    )
    print(f"Per-run time limit: {args.time_limit:.1f}s (override with --time-limit)")
    print("PythonSparse solvers:", ", ".join(label for label, _ in pythonsparse_solvers))
    print()
    table = brick.StaticBenchmarkTable(solver_names)
    _, results, far_node = run_benchmark(mesh_factors, time_limit=args.time_limit, table=table)

    if far_node is not None:
        print()
        print("Far-corner displacement:", ops.nodeDisp(far_node))

    print()
    print("Passed!" if all(status == 0 for _, _, _, status, _ in results) else "Failed!")
    print("==========================")


if __name__ == "__main__":
    main()
