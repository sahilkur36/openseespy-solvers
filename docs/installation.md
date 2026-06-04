# Installation

Follow the **quick install** below, then see [Recommended solvers](recommended-solvers.md) for
which factories to use in OpenSees (`nvmath` / `scipy.spsolve` for `Ax = b`; `cupy.eigsh` /
`scipy.eigsh` for modal).

---

## Quick install

### CPU only (no NVIDIA GPU)

```bash
pip install openseespy-solvers

# Faster direct linear solve on large systems (Linux: usually enough)
pip install openseespy-solvers[umfpack]
```

Use [`scipy.spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve) or
[`scipy.umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack) for linear solves
(static, transient, and any analysis using `ops.system("PythonSparse", ...)`) and
[`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) for eigenanalysis. Details:
[Recommended solvers](recommended-solvers.md).

### NVIDIA GPU (linear + eigen)

```bash
pip install openseespy-solvers
nvidia-smi
```

Read **`CUDA Version`** in the `nvidia-smi` header (top right), then install **matching**
wheels (do not mix 12.x and 13.x):

| `CUDA Version` | Commands |
|----------------|----------|
| **12.x** | `pip install cupy-cuda12x` · `pip install "nvmath-python[cu12]>=0.9.0"` |
| **13.x** | `pip install cupy-cuda13x` · `pip install "nvmath-python[cu13]>=0.9.0"` |

[`nvmath.direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) is recommended
for a GPU-accelerated direct solver in static and transient analysis.
[`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) is recommended for modal analysis
(default `mass_mode="general"`). Step-by-step GPU notes: [GPU install (CUDA)](#gpu-install-cuda).

If CuPy fails at import with missing CUDA libraries:

```bash
pip install "cupy-cuda13x[ctk]"   # or cupy-cuda12x[ctk]
```

---

## Requirements (everyone)

- Python ≥ 3.12
- NumPy ≥ 1.26
- SciPy ≥ 1.12 (needed for `rtol` / `atol` on sparse iterative solvers)
- [OpenSeesPy](https://openseespydoc.readthedocs.io/) (not installed by the base package)

These minimums are enforced in `pyproject.toml`.

This package assumes OpenSeesPy is available in your environment. If it is not installed:

```bash
pip install openseespy-solvers[opensees]
```

**Base install** (SciPy CPU solvers — no GPU required):

```bash
pip install openseespy-solvers
```

That gives you `openseespy_solvers.scipy` (`spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`) and
`openseespy_solvers.scipy.precond` (`jacobi`, `ilu`). For typical OpenSees workflows, start with
the [recommended solvers](recommended-solvers.md) page rather than comparing every factory.

---

## Choose your stack

| You have… | Install path |
|-----------|--------------|
| No NVIDIA GPU (or GPU solvers not needed) | [CPU-only](#cpu-only-install) |
| NVIDIA GPU | [GPU install](#gpu-install-cuda) — CuPy and/or nvMath |

---

## CPU-only install

No CUDA driver or toolkit required.

```bash
pip install openseespy-solvers

pip install openseespy-solvers[dev]       # optional: pytest, ruff
```

### UMFPACK (`scipy.umfpack`)

```bash
pip install openseespy-solvers[umfpack]
```

On **Linux**, that is usually enough (you may need `libsuitesparse-dev` from your distro).

On **Windows**, PyPI ships **source only** for recent scikit-umfpack — `pip` will try to
compile and needs SWIG + SuiteSparse. Use conda-forge instead:

```bash
conda install -c conda-forge scikit-umfpack
```

See the [scikit-umfpack install guide](https://scikit-umfpack.github.io/scikit-umfpack/install.html).

---

## GPU install (CUDA)

Two optional **GPU** backends (like SciPy on CPU, but for CUDA):

| Module | Library | Install (after base package) |
|--------|---------|------------------------------|
| `openseespy_solvers.cupy` | CuPy | `cupy-cuda12x` or `cupy-cuda13x` |
| `openseespy_solvers.nvmath` | nvmath-python | `nvmath-python[cu12]` or `[cu13]` |

SciPy solvers always run on the CPU. CuPy and nvMath require an NVIDIA GPU and a
matching CUDA wheel (see below).

### Step 1 — Check your NVIDIA driver

```bash
nvidia-smi
```

In the header, note **CUDA Version** (top right). That is the **maximum CUDA version your
driver supports**, not necessarily a CUDA Toolkit installed on disk.

Example:

```text
| NVIDIA-SMI 591.74    Driver Version: 591.74    CUDA Version: 13.1 |
```

Use the **CUDA 13** wheels below (`cupy-cuda13x`, `nvmath-python[cu13]`).

### Step 2 — Minimum recommended versions

| Component | Minimum / recommended |
|-----------|------------------------|
| NVIDIA driver | Recent driver for your GPU (`nvidia-smi` must work) |
| CUDA generation | **12.x or 13.x** (match the `CUDA Version` line from step 1) |
| CuPy | ≥ 13 — **`cupy-cuda12x`** or **`cupy-cuda13x`**, not generic `cupy` |
| nvmath-python | ≥ 0.9 — **`[cu12]`** or **`[cu13]`** matching the same generation as CuPy |

You do **not** need a full CUDA Toolkit on disk if the wheels bundle runtimes (CuPy
``[ctk]``). A working NVIDIA **driver** is enough.

### Step 3 — Map `nvidia-smi` to install commands

| `nvidia-smi` reports `CUDA Version` | CuPy | nvMath |
|-------------------------------------|------|--------|
| **12.x** | `pip install cupy-cuda12x` | `pip install "nvmath-python[cu12]>=0.9.0"` |
| **13.x** | `pip install cupy-cuda13x` | `pip install "nvmath-python[cu13]>=0.9.0"` |

If CuPy fails at import with missing CUDA libraries:

```bash
pip install "cupy-cuda13x[ctk]"   # or cupy-cuda12x[ctk]
```

``nvmath.direct_solver()`` uses the GPU by default (install CuPy + nvMath wheels above).

### Step 4 — Full GPU stack (editable install from source)

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers

pip install -e ".[dev,opensees]"

# Replace 13 with 12 if nvidia-smi shows CUDA Version 12.x
pip install cupy-cuda13x
pip install "nvmath-python[cu13]>=0.9.0"

# Optional CPU solver on Windows
conda install -c conda-forge scikit-umfpack

pip check
pytest
```

The nvMath backend auto-locates the cuDSS threading library shipped with
``nvidia-cudss-cu13`` / ``nvidia-cudss-cu12`` (e.g. ``cudss_mtlayer_vcomp140.dll``
under ``site-packages/nvidia/cu13/bin/`` on Windows). To override, set
``CUDSS_THREADING_LIB`` to the full path or pass
``multithreading_lib=...`` to :func:`~openseespy_solvers.nvmath.direct_solver`.

Do **not** use `pip install -e ".[cupy]"` on Windows — it pulls generic `cupy` from
source. Install `cupy-cuda12x` / `cupy-cuda13x` explicitly (step 3).

---

## Optional extras reference

| Extra | Installs | Enables |
|-------|----------|---------|
| *(base)* | NumPy, SciPy | `openseespy_solvers.scipy` |
| `opensees` | openseespy | OpenSeesPy when missing (see Requirements) |
| `umfpack` | scikit-umfpack | `scipy.umfpack` (see Windows note above) |
| `cupy` | generic `cupy>=13` | Prefer `cupy-cuda12x` / `cupy-cuda13x` instead |
| `dev` | pytest, ruff, build | Development |
| `docs` | mkdocs, mkdocstrings | Build documentation |

**CuPy** and **nvMath** are not pip extras with fixed CUDA versions — install the wheels
from [GPU install](#gpu-install-cuda) step 3 so they match your driver.

Further reading:

- [CuPy installation](https://docs.cupy.dev/en/stable/install.html)
- [nvmath-python installation](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html)
- [`nvmath.direct_solver` API](api/nvmath.md#openseespy_solvers.nvmath.direct_solver)

---

## Install from source (minimal)

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers
pip install -e ".[dev]"
pytest
```

Add optional backends using the [CPU-only](#cpu-only-install) or
[GPU](#gpu-install-cuda) sections above.

---

## Build documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site is served at `http://127.0.0.1:8000`.

---

## Troubleshooting

### Mixed CUDA 12 / 13 packages

If `pip check` reports conflicts (e.g. `nvidia-cudss-cu12` vs `cuda-toolkit 13.*`),
remove leftover `*-cu12` packages and reinstall the CUDA 13 stack, or recreate the
environment and follow [GPU install](#gpu-install-cuda) from scratch.

### scikit-umfpack fails on Windows with “Program 'swig' not found”

Use `conda install -c conda-forge scikit-umfpack` instead of
`pip install openseespy-solvers[umfpack]`.
