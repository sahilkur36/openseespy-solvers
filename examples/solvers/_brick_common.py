"""Shared 3-D brick bar model for solver examples (kip, in, sec)."""

import math

BAR_LENGTH = 10.0
BAR_HEIGHT = 2.0
BAR_THICKNESS = 1.0
ELASTIC_MODULUS = 29_000.0
POISSON_RATIO = 0.3
STEEL_DENSITY = 0.284e-3 / 386.4
YIELD_STRESS = 50.0
FAR_CORNER = (BAR_LENGTH, BAR_THICKNESS / 2.0, BAR_HEIGHT / 2.0)

# Fast banded ARPACK reference (default for verification timing).
FAST_EIGEN_SOLVER = "genBandArpack"
# Dense LAPACK reference (accurate tiebreaker when fast reference disagrees).
TRUSTED_EIGEN_SOLVER = "fullGenLapack"
REFERENCE_EIGEN_SOLVER = TRUSTED_EIGEN_SOLVER

# Mesh-size knobs for benchmark scripts (mesh_size = BAR_THICKNESS / factor).
DEFAULT_STATIC_MESH_FACTORS = [1.5, 2.0, 2.5, 3.0]
LARGE_STATIC_MESH_FACTORS = [4.0, 5.0, 6.0, 8.0]
DEFAULT_EIGEN_MESH_FACTORS = [1.0, 2.0, 3.0, 4.0]
LARGE_EIGEN_MESH_FACTORS = [4.0, 6.0, 8.0, 10.0]

# Native OpenSees solvers compared in ``brick_bar.py`` (static benchmark).
NATIVE_STATIC_SOLVERS = ["BandGeneral", "SuperLU", "UmfPack"]

# Benchmark scripts: skip a solver on finer meshes after it uses this fraction of --time-limit.
BUDGET_SKIP_FRACTION = 0.85


def equation_count_for_mesh() -> int:
    """Return ``ops.systemSize()`` after ``build_model`` (requires a linear system)."""
    import openseespy.opensees as ops

    ops.system("BandGeneral")
    return ops.systemSize()


def budget_record_status(
    name: str,
    seconds: float,
    time_limit: float,
    *,
    factor: float,
    skip_remaining: set[str],
) -> int:
    """Update ``skip_remaining`` when over budget; return OpenSees-style status (0 or -2)."""
    if seconds > time_limit:
        skip_remaining.add(name)
        print(
            f"  Mesh {factor}: {name} exceeded time budget "
            f"({seconds:.3f}s > {time_limit:.1f}s); finer meshes skipped"
        )
        return -2
    if seconds > time_limit * BUDGET_SKIP_FRACTION:
        skip_remaining.add(name)
        print(
            f"  Mesh {factor}: {name} used {seconds:.3f}s "
            f"(>{time_limit * BUDGET_SKIP_FRACTION:.1f}s); finer meshes skipped"
        )
    return 0


# OpenSees-style statuses that do not fail brick_bar / brick_bar_eigen for PythonSparse rows.
_PYTHONSPARSE_BENCHMARK_OK = frozenset({0, -2})


def benchmark_pythonsparse_passed(
    results: list[tuple],
    pythonsparse_labels: set[str],
    *,
    status_index: int = 3,
) -> bool:
    """True when every PythonSparse row succeeded or was budget-skipped (native refs ignored)."""
    for row in results:
        label = row[2]
        if label not in pythonsparse_labels:
            continue
        if row[status_index] not in _PYTHONSPARSE_BENCHMARK_OK:
            return False
    return True


_FACTORY_ALIASES = {
    "SpSolve": "spsolve",
    "Umfpack": "umfpack",
    "DirectSolver": "direct_solver",
}


def pythonsparse_label(solver) -> str:
    """Display name for benchmark tables, e.g. ``PythonSparse (scipy.spsolve)``."""
    cls = type(solver).__name__.removeprefix("_")
    factory = _FACTORY_ALIASES.get(cls, cls.lower())
    return f"PythonSparse ({solver.backend}.{factory})"


def gpu_available() -> bool:
    """True when CuPy sees at least one CUDA device."""
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def eigen_verify_tolerances(solver) -> tuple[float, float]:
    """Relaxed tolerances for iterative LOBPCG eigen solvers."""
    if type(solver).__name__ == "_Lobpcg":
        return 1e-3, 0.05
    return 1e-4, 0.02


def eigen_skips_trusted_reference(label: str) -> bool:
    """Eigen solvers that solve a different problem (no ``fullGenLapack`` tiebreaker)."""
    return "cupy.eigsh.lumped" in label


def pythonsparse_static_solvers() -> list[tuple[str, object]]:
    """Recommended PythonSparse static solvers (see docs/recommended-solvers.md)."""
    from openseespy_solvers.scipy import spsolve as scipy_spsolve

    solvers: list[tuple[str, object]] = []
    scipy_solver = scipy_spsolve()
    solvers.append((pythonsparse_label(scipy_solver), scipy_solver))

    try:
        from openseespy_solvers.scipy import umfpack as scipy_umfpack

        umfpack_solver = scipy_umfpack()
        solvers.append((pythonsparse_label(umfpack_solver), umfpack_solver))
    except ImportError:
        pass

    if not gpu_available():
        return solvers

    try:
        from openseespy_solvers.nvmath import direct_solver

        nvmath_solver = direct_solver()
        solvers.append((pythonsparse_label(nvmath_solver), nvmath_solver))
    except ImportError:
        pass

    return solvers


def pythonsparse_eigen_solvers() -> list[tuple[str, object]]:
    """Recommended PythonSparse eigen solvers (see docs/recommended-solvers.md)."""
    from openseespy_solvers.scipy import eigsh

    solvers: list[tuple[str, object]] = []
    scipy_solver = eigsh(tol=0.0)
    solvers.append((pythonsparse_label(scipy_solver), scipy_solver))

    if not gpu_available():
        return solvers

    try:
        from openseespy_solvers.cupy import eigsh as cupy_eigsh

        cupy_solver = cupy_eigsh(tol=0.0)
        solvers.append((pythonsparse_label(cupy_solver), cupy_solver))
    except ImportError:
        pass

    return solvers


def _static_solver_width(solver_names) -> int:
    return max(len("Solver"), *(len(name) for name in solver_names))


def _print_static_header(solver_w: int) -> None:
    header = (
        f"{'Mesh':>6} {'Eqns':>8} {'Solver':<{solver_w}} {'Status':>8} {'Time (s)':>10}"
    )
    print(header)
    print("-" * len(header))


def _print_static_row(solver_w: int, factor, equations, name, status, seconds) -> None:
    print(
        f"{factor:>6} {equations:>8} {name:<{solver_w}} {status:>8} {seconds:>10.3f}",
        flush=True,
    )


class StaticBenchmarkTable:
    """Print static benchmark rows as each solver finishes."""

    def __init__(self, solver_names) -> None:
        self._solver_w = _static_solver_width(solver_names)
        _print_static_header(self._solver_w)

    def add_row(self, factor, equations, name, status, seconds) -> None:
        _print_static_row(self._solver_w, factor, equations, name, status, seconds)


def print_static_benchmark_table(results) -> None:
    """Print aligned timing table for ``brick_bar.py``."""
    solver_w = max(len("Solver"), *(len(name) for _, _, name, _, _ in results))
    _print_static_header(solver_w)
    for factor, equations, name, status, seconds in results:
        _print_static_row(solver_w, factor, equations, name, status, seconds)


def _eigen_solver_width(solver_names) -> int:
    return max(len("Solver"), *(len(name) for name in solver_names))


def _print_eigen_header(solver_w: int) -> None:
    header = (
        f"{'Mesh':>6} {'Eqns':>8} {'Solver':<{solver_w}} {'Status':>8} "
        f"{'Time (s)':>10} {'lambda_1':>14}"
    )
    print(header)
    print("-" * len(header))


def _print_eigen_row(
    solver_w: int, factor, equations, name, status, seconds, first
) -> None:
    print(
        f"{factor:>6} {equations:>8} {name:<{solver_w}} {status:>8} "
        f"{seconds:>10.3f} {first:>14.6g}",
        flush=True,
    )


class EigenBenchmarkTable:
    """Print eigen benchmark rows as each solver finishes."""

    def __init__(self, solver_names) -> None:
        self._solver_w = _eigen_solver_width(solver_names)
        _print_eigen_header(self._solver_w)

    def add_row(self, factor, equations, name, status, seconds, first) -> None:
        _print_eigen_row(self._solver_w, factor, equations, name, status, seconds, first)


def print_eigen_benchmark_table(results) -> None:
    """Print aligned timing table for ``brick_bar_eigen.py``."""
    solver_w = max(len("Solver"), *(len(name) for _, _, name, _, _, _ in results))
    _print_eigen_header(solver_w)
    for factor, equations, name, status, seconds, first in results:
        _print_eigen_row(solver_w, factor, equations, name, status, seconds, first)


def build_model(ops, nx, ny, nz):
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


def far_corner_node(ops):
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


def apply_load(ops):
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


def run_static(ops, solver, steps=2):
    ops.system("PythonSparse", solver.to_openseespy())
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0 / steps)
    ops.test("NormUnbalance", 1.0e-7, 50)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static")
    return ops.analyze(steps)


def run_eigen_native(ops, num_modes, solver=FAST_EIGEN_SOLVER):
    """Native OpenSees eigenvalues (default: ``genBandArpack``)."""
    eigenvalues = ops.eigen(solver, num_modes)
    if eigenvalues:
        return eigenvalues
    return None


def node_mode_shape(ops, node, mode=1):
    if node is None:
        return None
    return ops.nodeEigenvector(node, mode)


def eigenvalue_rel_diff(reference: float, value: float) -> float | None:
    """Relative eigenvalue difference ``|value - reference| / |reference|``."""
    if reference == 0.0:
        return abs(value - reference)
    return abs(value - reference) / abs(reference)


def mode_shape_cosine(reference, values) -> float | None:
    """Sign-invariant cosine similarity between mode shapes; ``None`` if undefined."""
    if reference is None or values is None:
        return None
    ref = list(reference)
    val = list(values)
    if len(ref) != len(val):
        return None
    dot = sum(a * b for a, b in zip(ref, val, strict=True))
    ref_norm = math.sqrt(sum(a * a for a in ref))
    val_norm = math.sqrt(sum(a * a for a in val))
    if ref_norm < 1e-15 or val_norm < 1e-15:
        return None
    return abs(dot / (ref_norm * val_norm))


def print_eigen_lumped_approx_report(
    factor,
    label,
    ref_ev,
    py_ev,
    *,
    ref_mode_first=None,
    ref_mode_last=None,
    py_mode_first=None,
    py_mode_last=None,
    reference=FAST_EIGEN_SOLVER,
) -> None:
    """Brief lumped-mass comparison vs the fast reference (first/last modes + worst lambda)."""
    print(
        f"  Mesh {factor}: {label} — row-sum lumped mass (approximate; "
        f"not expected to match {reference} exactly)"
    )
    if not ref_ev or not py_ev:
        print("    (missing eigenvalues)")
        return
    n = min(len(ref_ev), len(py_ev))
    if n == 0:
        return

    def _mode_line(mode_num, ref_lam, py_lam, ref_vec, py_vec) -> None:
        rel = eigenvalue_rel_diff(ref_lam, py_lam)
        cos = mode_shape_cosine(ref_vec, py_vec)
        rel_str = f"{rel * 100:.2g}%" if rel is not None else "n/a"
        cos_str = f"{cos:.4f}" if cos is not None else "n/a"
        print(
            f"    mode {mode_num}: lambda {reference}={ref_lam:.6g}  "
            f"lumped={py_lam:.6g}  rel_err={rel_str};  far-corner cosine={cos_str}"
        )

    _mode_line(1, ref_ev[0], py_ev[0], ref_mode_first, py_mode_first)
    if n > 1:
        _mode_line(n, ref_ev[n - 1], py_ev[n - 1], ref_mode_last, py_mode_last)

    worst_idx = 0
    worst_rel = -1.0
    for i in range(n):
        rel = eigenvalue_rel_diff(ref_ev[i], py_ev[i])
        if rel is not None and rel > worst_rel:
            worst_rel = rel
            worst_idx = i
    if worst_rel >= 0 and worst_idx not in (0, n - 1):
        print(
            f"    largest |lambda| rel_err: mode {worst_idx + 1} ({worst_rel * 100:.2g}%)"
        )


def eigenvalues_close(reference, values, rel_tol=1e-4):
    if reference is None or values is None:
        return False
    if len(reference) != len(values):
        return False
    for ref, val in zip(reference, values, strict=True):
        if not math.isclose(ref, val, rel_tol=rel_tol, abs_tol=1e-9):
            return False
    return True


def eigenvector_close(reference, values, rel_tol=0.02):
    """Compare mode shapes at a node (sign-invariant)."""
    if reference is None or values is None:
        return False
    ref = list(reference)
    val = list(values)
    if len(ref) != len(val):
        return False
    dot = sum(a * b for a, b in zip(ref, val, strict=True))
    ref_norm = math.sqrt(sum(a * a for a in ref))
    val_norm = math.sqrt(sum(a * a for a in val))
    if ref_norm < 1e-15 or val_norm < 1e-15:
        return False
    cosine = abs(dot / (ref_norm * val_norm))
    return cosine >= (1.0 - rel_tol)


def run_eigen(ops, solver, num_modes=3):
    eigenvalues = ops.eigen("PythonSparse", num_modes, solver.to_openseespy())
    if eigenvalues:
        return 0, eigenvalues
    return -1, None


def _eigen_results_match(ref_ev, ev, ref_mode, test_mode, *, ev_rel_tol, vec_rel_tol):
    return (
        ref_ev is not None
        and ev is not None
        and eigenvalues_close(ref_ev, ev, rel_tol=ev_rel_tol)
        and eigenvector_close(ref_mode, test_mode, rel_tol=vec_rel_tol)
    )


def print_eigen_solver_comparison(
    ref_ev,
    test_ev,
    *,
    ref_mode_first=None,
    test_mode_first=None,
    reference_label="eigsh",
    test_label="lobpcg",
) -> None:
    """Per-mode eigenvalue comparison and mode-1 shape cosine at the far corner."""
    print(f"  Comparison: {test_label} vs {reference_label}")
    if not ref_ev or not test_ev:
        print("    (missing eigenvalues)")
        return
    n = min(len(ref_ev), len(test_ev))
    for i in range(n):
        rel = eigenvalue_rel_diff(ref_ev[i], test_ev[i])
        rel_str = f"{rel * 100:.2g}%" if rel is not None else "n/a"
        print(
            f"    mode {i + 1}: {reference_label}={ref_ev[i]:.6g}  "
            f"{test_label}={test_ev[i]:.6g}  rel_err={rel_str}"
        )
    if ref_mode_first is not None and test_mode_first is not None:
        cos = mode_shape_cosine(ref_mode_first, test_mode_first)
        cos_str = f"{cos:.4f}" if cos is not None else "n/a"
        print(f"    mode 1 far-corner cosine={cos_str}")


def run_eigen_vs_reference(
    ops,
    solver,
    reference_solver,
    num_modes,
    node,
    rebuild,
    *,
    ev_rel_tol=1e-3,
    vec_rel_tol=0.05,
    reference_label="eigsh",
    test_label="lobpcg",
):
    """Verify a PythonSparse eigen solver against another on the same model."""
    rebuild()
    node = far_corner_node(ops)
    ref_status, ref_ev = run_eigen(ops, reference_solver, num_modes)
    ref_mode = node_mode_shape(ops, node, 1)
    if ref_status != 0:
        print_eigen_solver_comparison(
            None,
            None,
            reference_label=reference_label,
            test_label=test_label,
        )
        return -1, None, None

    rebuild()
    node = far_corner_node(ops)
    status, ev = run_eigen(ops, solver, num_modes)
    test_mode = node_mode_shape(ops, node, 1)
    print_eigen_solver_comparison(
        ref_ev,
        ev,
        ref_mode_first=ref_mode,
        test_mode_first=test_mode,
        reference_label=reference_label,
        test_label=test_label,
    )
    if status != 0:
        return -1, ev, ref_ev

    if _eigen_results_match(
        ref_ev, ev, ref_mode, test_mode, ev_rel_tol=ev_rel_tol, vec_rel_tol=vec_rel_tol
    ):
        return 0, ev, ref_ev

    return -1, ev, ref_ev


def run_eigen_verified(
    ops,
    solver,
    num_modes,
    node,
    rebuild,
    *,
    ev_rel_tol=1e-4,
    vec_rel_tol=0.02,
):
    """Two-tier eigen verification: ``genBandArpack`` first, ``fullGenLapack`` tiebreaker."""
    fast_ev = run_eigen_native(ops, num_modes, FAST_EIGEN_SOLVER)
    fast_mode = node_mode_shape(ops, node, 1)
    rebuild()
    node = far_corner_node(ops)
    status, ev = run_eigen(ops, solver, num_modes)
    test_mode = node_mode_shape(ops, node, 1)
    if status != 0:
        return -1, ev

    if _eigen_results_match(
        fast_ev, ev, fast_mode, test_mode, ev_rel_tol=ev_rel_tol, vec_rel_tol=vec_rel_tol
    ):
        return 0, ev

    if getattr(solver, "_mass_mode", "").lower() == "lumped":
        return 0, ev

    rebuild()
    node = far_corner_node(ops)
    trusted_ev = run_eigen_native(ops, num_modes, TRUSTED_EIGEN_SOLVER)
    trusted_mode = node_mode_shape(ops, node, 1)
    if _eigen_results_match(
        trusted_ev, ev, trusted_mode, test_mode, ev_rel_tol=ev_rel_tol, vec_rel_tol=vec_rel_tol
    ):
        rebuild()
        node = far_corner_node(ops)
        status, ev = run_eigen(ops, solver, num_modes)
        return status, ev

    return -1, ev
