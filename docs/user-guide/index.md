# User guide

OpenSeesPy can delegate linear and eigen solves to a **Python solver** through the
[`PythonSparse` system and eigen commands](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html).
OpenSees assembles the sparse matrices and right-hand side, then calls methods on
your solver object (`solve`, and for linear solvers `formAp`).

**openseespy-solvers** wraps `scipy`, `cupy`, and `nvmath` as ready-made solver
objects so you do not have to implement that callback protocol yourself. In
application code you typically:

1. Pick a factory (`spsolve`, `eigsh`, `direct_solver`, …).
2. Pass `solver.to_openseespy()` to OpenSeesPy.

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
# … numberer, constraints, integrator, analysis …
ops.analyze(num_steps)
```

```python
from openseespy_solvers.scipy import eigsh

eig_solver = eigsh()
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

That is the whole integration surface for most analyses. The pages below cover
solver configuration, preconditioners, and what you can inspect after a solve.

## to_openseespy()

`solver.to_openseespy()` returns the configuration dict OpenSeesPy expects:

```python
solver = cg(rtol=1e-8)
cfg = solver.to_openseespy()
ops.system("PythonSparse", cfg)
```

```python
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

**Return value**

| Key | Meaning |
|-----|---------|
| `solver` | The solver instance (OpenSees retains a reference and calls its methods). |
| `scheme` | Sparse storage scheme (`'CSR'` by default). |
| `writable` | Writable buffer policy (included when set at construction). |

**Parameters** (optional overrides at call time)

`scheme`
: Override the storage scheme.

`writable`
: Override the writable buffer list (for example `'values'` or `['values', 'rhs']`).
  Default at construction is `'none'`.

Use `copy.copy(solver)` to obtain a fresh instance with the same configuration
but empty internal state. OpenSees may clone solvers when copying a system of
equations.

Factory-level API details: [`BaseOpenSeesSolver.to_openseespy`](../api/scipy.md).

## See also

- [Solver objects](solver-objects.md) — attributes, `stats`, `dtype`, `copy.copy`
- [Preconditioners](preconditioners.md) — `M=` for iterative solvers
- [Tutorial](../getting-started.md) — full static and eigen workflows
- [PythonSparse interface](../development/pythonsparse-interface.md) — buffers,
  `matrix_status`, and parallelism (for contributors and advanced users)
