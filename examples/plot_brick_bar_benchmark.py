"""Run brick-bar benchmarks and write figures to ``docs/assets/``."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

_EXAMPLES = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES.parent
_ASSETS = _REPO_ROOT / "docs" / "assets"
_SOLVERS = _EXAMPLES / "solvers"

if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

import _brick_common as brick  # noqa: E402

STATIC_CSV = "brick-bar-static-benchmark.csv"
STATIC_PNG = "brick-bar-static-benchmark.png"
EIGEN_CSV = "brick-bar-eigen-benchmark.csv"
EIGEN_PNG = "brick-bar-eigen-benchmark.png"
STATIC_META = "brick-bar-static-benchmark.json"
EIGEN_META = "brick-bar-eigen-benchmark.json"


def mesh_counts(factor: float) -> tuple[int, int, int]:
    mesh_size = brick.BAR_THICKNESS / factor
    nx = max(1, int(math.ceil(brick.BAR_LENGTH / mesh_size)))
    ny = max(1, int(math.ceil(brick.BAR_THICKNESS / mesh_size)))
    nz = max(1, int(math.ceil(brick.BAR_HEIGHT / mesh_size)))
    return nx, ny, nz


def estimate_equation_count(nx: int, ny: int, nz: int, *, ndf: int = 3) -> int:
    total_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    fixed_nodes = (ny + 1) * (nz + 1)
    return ndf * (total_nodes - fixed_nodes)


def mesh_table(mesh_factors: list[float]) -> list[dict[str, object]]:
    rows = []
    for factor in mesh_factors:
        nx, ny, nz = mesh_counts(factor)
        rows.append(
            {
                "factor": factor,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "equations": estimate_equation_count(nx, ny, nz),
            }
        )
    return rows


def host_metadata() -> dict[str, str]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }


class IncrementalCsvWriter:
    """Write benchmark rows to CSV as each solver finishes."""

    def __init__(
        self, path: Path, fieldnames: list[str], *, append: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        if append and path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(fieldnames)

    def __call__(self, row: tuple) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
            fh.flush()


def _row_keys(results: list[tuple]) -> set[tuple[float, str]]:
    return {(row[0], row[2]) for row in results}


def merge_results(existing: list[tuple], new: list[tuple]) -> list[tuple]:
    by_key = {(row[0], row[2]): row for row in existing}
    for row in new:
        by_key[(row[0], row[2])] = row
    return sorted(by_key.values(), key=lambda row: (row[0], row[2]))


def run_static_benchmark(
    mesh_factors: list[float],
    *,
    time_limit: float,
    csv_path: Path | None = None,
    append: bool = False,
) -> list[tuple]:
    import brick_bar  # noqa: E402

    pythonsparse_solvers = brick.pythonsparse_static_solvers()
    solver_names = [label for label, _ in pythonsparse_solvers] + brick_bar.NATIVE_SOLVERS

    print()
    print(f"Static analysis — mesh factors: {mesh_factors}")
    print(
        f"Running {brick_bar.NUM_STEPS} steps of LoadControl with "
        f"dLambda = 1 / {brick_bar.NUM_STEPS}"
    )
    print(
        f"Per-run time budget: {time_limit:.1f}s (not a hard timeout; "
        f"skip finer meshes after >{brick.BUDGET_SKIP_FRACTION:.0%} of budget)"
    )
    print("PythonSparse solvers:", ", ".join(label for label, _ in pythonsparse_solvers))
    print()
    existing: list[tuple] = []
    skip_rows: set[tuple[float, str]] = set()
    if append and csv_path is not None and csv_path.exists() and csv_path.stat().st_size > 0:
        existing = read_static_csv(csv_path)
        skip_rows = _row_keys(existing)
        print(f"Appending to {csv_path} ({len(existing)} existing rows)")
    elif csv_path is not None:
        print(f"Writing rows to {csv_path}")
    if skip_rows:
        print(f"Skipping {len(skip_rows)} existing mesh/solver pairs")
    print()
    table = brick.StaticBenchmarkTable(solver_names)
    on_row = None
    if csv_path is not None:
        on_row = IncrementalCsvWriter(
            csv_path,
            ["mesh_factor", "equations", "solver", "status", "seconds"],
            append=append,
        )
    _, results, _ = brick_bar.run_benchmark(
        mesh_factors,
        time_limit=time_limit,
        table=table,
        on_row=on_row,
        skip_rows=skip_rows,
    )
    return merge_results(existing, results)


def run_eigen_benchmark(
    mesh_factors: list[float],
    *,
    time_limit: float,
    csv_path: Path | None = None,
    append: bool = False,
) -> list[tuple]:
    import brick_bar_eigen  # noqa: E402

    pythonsparse_solvers = brick.pythonsparse_eigen_solvers()
    solver_names = (
        [brick_bar_eigen.FAST_REFERENCE]
        + [label for label, _ in pythonsparse_solvers]
        + [brick_bar_eigen.TRUSTED_REFERENCE]
    )

    print()
    print(f"Eigen analysis — mesh factors: {mesh_factors}")
    print(
        f"Primary reference: {brick_bar_eigen.FAST_REFERENCE}; "
        f"tiebreaker: {brick_bar_eigen.TRUSTED_REFERENCE} (on mismatch only)."
    )
    print(
        f"Per-run time budget: {time_limit:.1f}s (not a hard timeout; "
        f"skip finer meshes after >{brick.BUDGET_SKIP_FRACTION:.0%} of budget)"
    )
    print("PythonSparse solvers:", ", ".join(label for label, _ in pythonsparse_solvers))
    print()
    existing: list[tuple] = []
    skip_rows: set[tuple[float, str]] = set()
    if append and csv_path is not None and csv_path.exists() and csv_path.stat().st_size > 0:
        existing = read_eigen_csv(csv_path)
        skip_rows = _row_keys(existing)
        print(f"Appending to {csv_path} ({len(existing)} existing rows)")
    elif csv_path is not None:
        print(f"Writing rows to {csv_path}")
    if skip_rows:
        print(f"Skipping {len(skip_rows)} existing mesh/solver pairs")
    print()
    table = brick.EigenBenchmarkTable(solver_names)
    on_row = None
    if csv_path is not None:
        on_row = IncrementalCsvWriter(
            csv_path,
            [
                "mesh_factor",
                "equations",
                "solver",
                "status",
                "seconds",
                "lambda_1",
            ],
            append=append,
        )
    _, results, _, _ = brick_bar_eigen.run_benchmark(
        mesh_factors,
        time_limit=time_limit,
        table=table,
        on_row=on_row,
        skip_rows=skip_rows,
    )
    return merge_results(existing, results)


def write_static_csv(path: Path, results: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mesh_factor", "equations", "solver", "status", "seconds"])
        for factor, equations, solver, status, seconds in results:
            writer.writerow([factor, equations, solver, status, seconds])


def write_eigen_csv(path: Path, results: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["mesh_factor", "equations", "solver", "status", "seconds", "lambda_1"]
        )
        for factor, equations, solver, status, seconds, lambda_1 in results:
            writer.writerow([factor, equations, solver, status, seconds, lambda_1])


def read_static_csv(path: Path) -> list[tuple]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                (
                    float(row["mesh_factor"]),
                    int(row["equations"]),
                    row["solver"],
                    int(row["status"]),
                    float(row["seconds"]),
                )
            )
    return rows


def read_eigen_csv(path: Path) -> list[tuple]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                (
                    float(row["mesh_factor"]),
                    int(row["equations"]),
                    row["solver"],
                    int(row["status"]),
                    float(row["seconds"]),
                    float(row["lambda_1"]),
                )
            )
    return rows


def plot_timing(
    path: Path,
    results: list[tuple],
    *,
    title: str,
    ylabel: str,
    skip_solvers: set[str] | None = None,
) -> None:
    skip = skip_solvers or set()
    ok = [r for r in results if r[3] == 0 and r[4] > 0 and r[2] not in skip]
    by_solver: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in ok:
        equations, solver, seconds = row[1], row[2], row[4]
        by_solver[solver].append((equations, seconds))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for solver, points in sorted(by_solver.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        linestyle = "--" if solver.startswith("PythonSparse") else "-"
        ax.plot(xs, ys, marker="o", linewidth=1.8, linestyle=linestyle, label=solver)

    ax.set_xlabel("Number of equations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-from-csv",
        type=Path,
        help="plot static results from CSV instead of re-running OpenSees",
    )
    parser.add_argument(
        "--eigen-from-csv",
        type=Path,
        help="plot eigen results from CSV instead of re-running OpenSees",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120.0,
        help="per-run budget passed to the benchmark scripts",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="generate only static figures",
    )
    parser.add_argument(
        "--eigen-only",
        action="store_true",
        help="generate only eigen figures",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "keep existing CSV rows and skip mesh/solver pairs already present; "
            "new rows are appended as each run finishes"
        ),
    )
    args = parser.parse_args()

    _ASSETS.mkdir(parents=True, exist_ok=True)
    run_static = not args.eigen_only
    run_eigen = not args.static_only

    if run_static:
        if args.static_from_csv is not None:
            static_results = read_static_csv(args.static_from_csv)
        else:
            static_results = run_static_benchmark(
                brick.STATIC_MESH_FACTORS,
                time_limit=args.time_limit,
                csv_path=_ASSETS / STATIC_CSV,
                append=args.append,
            )

        plot_timing(
            _ASSETS / STATIC_PNG,
            static_results,
            title="Brick-bar static example (10 LoadControl steps)",
            ylabel="Wall time for full static analysis (s)",
        )
        (_ASSETS / STATIC_META).write_text(
            json.dumps(
                {
                    **host_metadata(),
                    "mesh_factors": brick.STATIC_MESH_FACTORS,
                    "mesh_table": mesh_table(brick.STATIC_MESH_FACTORS),
                    "num_steps": 10,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {_ASSETS / STATIC_PNG}")

    if run_eigen:
        if args.eigen_from_csv is not None:
            eigen_results = read_eigen_csv(args.eigen_from_csv)
        else:
            eigen_results = run_eigen_benchmark(
                brick.EIGEN_MESH_FACTORS,
                time_limit=args.time_limit,
                csv_path=_ASSETS / EIGEN_CSV,
                append=args.append,
            )

        plot_timing(
            _ASSETS / EIGEN_PNG,
            eigen_results,
            title="Brick-bar eigen example (5 modes)",
            ylabel="Wall time for full eigen analysis (s)",
            skip_solvers={brick.TRUSTED_EIGEN_SOLVER},
        )
        (_ASSETS / EIGEN_META).write_text(
            json.dumps(
                {
                    **host_metadata(),
                    "mesh_factors": brick.EIGEN_MESH_FACTORS,
                    "mesh_table": mesh_table(brick.EIGEN_MESH_FACTORS),
                    "num_modes": 5,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {_ASSETS / EIGEN_PNG}")


if __name__ == "__main__":
    main()
