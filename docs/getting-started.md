# Tutorial

This section demonstrates typical usage patterns. Model-building details are
omitted; see the [OpenSeesPy documentation](https://openseespydoc.readthedocs.io/)
and the [examples](examples.md) directory.

## Static analysis with a direct solver

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_opensees())
ops.numberer("RCM")
ops.constraints("Plain")
ops.integrator("LoadControl", 1.0)
ops.algorithm("Linear")
ops.analysis("Static")
ops.analyze(1)
```

The LU factorization is reused when OpenSees reports an unchanged matrix
structure and sparsity pattern between solves.

## Static analysis with CG and a Jacobi preconditioner

```python
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, maxiter=500, M=precond.jacobi)
ops.system("PythonSparse", solver.to_opensees())
```

The `M` argument accepts a preconditioner object or a callable `M(A)` that
receives the assembled matrix; see [Preconditioners](user-guide/preconditioners.md).

## GPU static analysis

```python
from openseespy_solvers.cupy import cg

solver = cg(rtol=1e-8)
ops.system("PythonSparse", solver.to_opensees())
```

Requires CuPy. After a solve, `solver.A` and `solver.x` are CuPy arrays on device.

## Modal analysis with eigsh

```python
from openseespy_solvers.scipy import eigsh

eigsolver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", 5, eigsolver.to_opensees())
```

Eigenvalues and eigenvectors are written to OpenSees output buffers in place.
`eigsolver.K` and `eigsolver.M` hold the assembled stiffness and mass matrices
from the last call.

## Modal analysis on GPU

CuPy does not provide a generalized `eigsh` for `K x = λ M x`. Use LOBPCG:

```python
from openseespy_solvers.cupy import lobpcg

eigsolver = lobpcg(tol=1e-8)
lam = ops.eigen("PythonSparse", 5, eigsolver.to_opensees())
```

## See Also

- [`scipy.spsolve` API](api/scipy.md#openseespy_solvers.scipy.spsolve)
- [`scipy.cg` API](api/scipy.md#openseespy_solvers.scipy.cg)
- [`scipy.eigsh` API](api/scipy.md#openseespy_solvers.scipy.eigsh)
- [PythonSparse interface](user-guide/pythonsparse-interface.md)
