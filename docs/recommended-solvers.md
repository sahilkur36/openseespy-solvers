# Recommended solvers

These are practical defaults for **linear** analysis steps (`ops.system("PythonSparse", ...)`
— static, transient, and any procedure that assembles `Ax = b`) and **modal**
(`ops.eigen("PythonSparse", ...)`). They route work through
[SciPy](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html),
[UMFPACK](https://scikit-umfpack.github.io/scikit-umfpack/),
[ARPACK](https://docs.scipy.org/doc/scipy/tutorial/arpack.html), and (with a GPU)
[NVIDIA nvMath](https://docs.nvidia.com/cuda/nvmath-python/latest/) / [CuPy](https://docs.cupy.dev/)
instead of OpenSees’s built-in sparse/direct/eigen routines.

On many medium-to-large models, the combinations below are **often faster** than native OpenSees
solvers (for example `BandSPD`, `UmfPack`, or bundled ARPACK paths), because they reuse
long‑optimized library implementations and factorization reuse via `matrix_status`. Exact
speedups depend on problem size, sparsity, hardware, and which OpenSees solver you compare
against.

---

## Linear systems (`Ax = b`)

| Situation | Recommended | OpenSees hook |
|-----------|-------------|-----------------|
| **NVIDIA GPU + CUDA** | [`nvmath.direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) | `ops.system("PythonSparse", solver.to_openseespy())` |
| **CPU only** | [`scipy.spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve) | same |
| **CPU, large systems** | [`scipy.umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack) (optional extra) | same |

**GPU example**

```python
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
```

**CPU example**

```python
from openseespy_solvers.scipy import spsolve  # or umfpack

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

Install paths: [GPU install](installation.md#gpu-install-cuda) (CuPy + nvMath wheels) and
[CPU-only install](installation.md#cpu-only-install) (`[umfpack]` on Linux; conda on Windows).

Iterative solvers (`cg`, `gmres`) are available when a direct factorization is too costly;
see [Preconditioners](user-guide/preconditioners.md). For many structural static or transient
steps with a reusable tangent, a **direct** solver above is the first choice.

For Newton or transient analyses where the tangent changes slowly between steps, consider
[`hybrid`](../api/hybrid.md) (`hybrid(direct=spsolve(), ...)`) — it factorizes once, then
reuses that factorization as a GMRES preconditioner until the system size changes or GMRES
fails to converge.

---

## Generalized eigenproblems (`K x = λ M x`)

| Situation | Recommended | Notes |
|-----------|-------------|--------|
| **NVIDIA GPU + CUDA** | [`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) | Default `mass_mode="general"`; SciPy ARPACK + GPU shift-invert inner solves |
| **CPU only** | [`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) | Same physical problem; ARPACK on CPU |

**GPU (consistent mass)**

```python
from openseespy_solvers.cupy import eigsh

eigsolver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, eigsolver.to_openseespy())
```

**CPU**

```python
from openseespy_solvers.scipy import eigsh

eigsolver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, eigsolver.to_openseespy())
```

Additional **experimental** eigen solvers are available (for example
[`cupy.lobpcg`](api/cupy.md#openseespy_solvers.cupy.lobpcg), other
[`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) `mass_mode` values, and
[`scipy.lobpcg`](api/scipy.md#openseespy_solvers.scipy.lobpcg)). For most models, use the
recommended [`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) or
[`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) paths above.

---

## Quick install recap

**CPU (eigen + linear)**

```bash
pip install openseespy-solvers
pip install openseespy-solvers[umfpack]   # optional; see installation.md on Windows
```

**GPU (recommended stack for both tables above)**

```bash
pip install openseespy-solvers
nvidia-smi   # read CUDA Version (12.x or 13.x)
pip install cupy-cuda13x                  # or cupy-cuda12x
pip install "nvmath-python[cu13]>=0.9.0"  # or [cu12]
```

Full steps, driver checks, and troubleshooting: [Installation](installation.md).

---

## See also

- [Tutorial](getting-started.md) — copy-paste OpenSees wiring
- [Examples](examples.md) — brick bar scripts
- [Reference: scipy](reference/scipy.md) · [cupy](reference/cupy.md) · [nvmath](reference/nvmath.md)
