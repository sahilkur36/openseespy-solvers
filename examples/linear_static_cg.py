"""Example: static analysis with SciPy CG and Jacobi preconditioner."""

from openseespy_solvers.scipy import cg, precond

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
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0, 0.0)

    solver = cg(rtol=1e-8, M=precond.jacobi)
    ops.system("PythonSparse", solver.to_opensees())
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    print("Stats:", solver.stats)


if __name__ == "__main__":
    main()
