# openseespy_solvers.hybrid

Hybrid linear solver factory.

`hybrid` reuses a direct factorization as a GMRES preconditioner. It is useful for
analyses where the tangent changes slowly and a full refactorization at every step is
unnecessarily expensive.

## Solving Linear Problems

| Factory | Description |
|---------|-------------|
| [`hybrid`](#openseespy_solvers.hybrid.hybrid) | Direct factorization reused as a GMRES preconditioner |

## Usage

```python
from openseespy_solvers import hybrid
from openseespy_solvers.scipy import spsolve

solver = hybrid(spsolve(), rtol=1e-6, restart=50)
ops.system("PythonSparse", solver.to_openseespy())
```

## Function Reference

::: openseespy_solvers.hybrid
    options:
      show_root_heading: false
      heading_level: 3
