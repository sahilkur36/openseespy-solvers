# Node numbering (Plain vs RCM)

OpenSees **`numberer`** controls how free degrees of freedom are ordered before assembly.
That ordering affects **fill-in and bandwidth** for solvers that rely on the assembled
matrix layout. It is separate from **fill-reducing reorderings inside sparse direct
libraries** (SuperLU column permutations, UMFPACK ordering, nvMath/cuDSS planning, etc.).

## Recommended choices

| OpenSees `system` / PythonSparse backend | `ops.numberer(...)` | Why |
|------------------------------------------|---------------------|-----|
| **`PythonSparse` + direct sparse** — `scipy.spsolve`, `scipy.umfpack`, `nvmath.direct_solver`, `cupy.spsolve` | **`Plain`** | The backend performs its own symbolic ordering / factorization planning on the CSR pattern OpenSees supplies. Extra RCM in OpenSees is usually redundant. |
| **Native sparse direct** — `SuperLU`, `UmfPack` | **`Plain`** | Same idea: reordering is handled inside the sparse direct package. |
| **Native banded** — `BandGeneral`, `genBandArpack` | **`RCM`** | Narrow bandwidth matters; RCM reduces profile before the banded kernel runs. |
| **Iterative PythonSparse** — `cg`, `gmres`, `hybrid`, `lobpcg` | **`RCM`** | No internal fill-reducing factorization; a tighter sparsity profile often helps Krylov methods and preconditioners. |
| **Eigen PythonSparse** — `eigsh`, `lobpcg` | **`RCM`** | Shift-invert and iterative eigen paths benefit from a better assembled matrix layout (same as other iterative use). |

## Examples

Direct sparse (CPU):

```python
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("Plain")
```

GPU direct (nvMath):

```python
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("Plain")
```

Iterative:

```python
from openseespy_solvers.scipy import cg
from openseespy_solvers.scipy import precond

solver = cg(rtol=1e-8, M=precond.jacobi)
ops.system("PythonSparse", solver.to_openseespy())
ops.numberer("RCM")
```

Native banded eigen reference:

```python
ops.numberer("RCM")
lam = ops.eigen("genBandArpack", num_modes)
```

## Notes

- **`constraints("Plain")`** is unrelated — it selects the constraint handler, not node
  numbering.
- Direct solvers in this package may still accept **`permc_spec`** / library options
  (e.g. SuperLU `COLAMD` on `scipy.spsolve`) in addition to OpenSees numbering.
- Using **`RCM` with a direct sparse solver** is not wrong, but it adds an extra
  permutation step that the library often repeats internally; **`Plain`** is the usual
  choice for parity with native `SuperLU` / `UmfPack` workflows.

## See also

- [Tutorial](../getting-started.md)
- [Recommended solvers](../recommended-solvers.md)
- [PythonSparse interface](pythonsparse-interface.md)
