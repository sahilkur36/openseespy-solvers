"""Example: modal analysis with SciPy eigsh."""

from openseespy_solvers.scipy import eigsh

try:
    import openseespy.opensees as ops
except ImportError as exc:
    raise SystemExit("Install OpenSeesPy to run this example.") from exc


def main() -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, 3000.0)
    ops.element("Truss", 1, 1, 2, 1.0, 1)
    ops.mass(2, 1.0, 1.0)

    solver = eigsh(tol=1e-8)
    lam = ops.eigen("PythonSparse", 1, solver.to_opensees())
    print("Eigenvalues:", lam)
    print("Stats:", solver.stats)


if __name__ == "__main__":
    main()
