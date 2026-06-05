# Installation

Set up a Python environment, install **openseespy-solvers** and optional backends, then
verify with **`pytest`**, **example scripts**, and **benchmarks**.

| Goal | Start here |
|------|------------|
| First-time setup (recommended) | [Full setup from scratch](#full-setup) |
| Library only, no clone | [Quick install (pip only)](#quick-install) |
| UMFPACK on your OS | [Step 4 — UMFPACK](#umfpack) |
| NVIDIA GPU (CuPy / nvMath) | [Step 5 — GPU](#gpu) |
| Something failed | [Troubleshooting](#troubleshooting) |

Which solvers to use in OpenSees: [Recommended solvers](recommended-solvers.md).

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | ≥ 3.12 |
| NumPy | ≥ 1.26 |
| SciPy | ≥ 1.12 |
| OpenSeesPy | Required for examples, benchmarks, and integration tests |

OpenSeesPy is **not** installed by `pip install openseespy-solvers` alone. The full setup
below uses the `[opensees]` extra. Minimum versions are enforced in `pyproject.toml`.

---

## Full setup from scratch {#full-setup}

Complete path: **environment → clone → install → optional backends → verify**.

### 1. Create a Python environment

Use a **fresh** environment with Python 3.12 or newer.

**Conda** (Linux, macOS, or Windows — Miniforge, Mambaforge, or Anaconda):

```bash
conda create -n openseespy-solvers python=3.12 -y
conda activate openseespy-solvers
```

**venv** (stdlib; all platforms):

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

To recreate an existing conda environment:

```bash
conda deactivate
conda env remove -n openseespy-solvers -y
conda create -n openseespy-solvers python=3.12 -y
conda activate openseespy-solvers
```

### 2. Clone the repository

Examples, benchmarks, and `pytest` live in the repository — they are **not** in the pip
wheel.

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers
```

### 3. Install the package

Editable install with SciPy solvers, development tools, and OpenSeesPy:

```bash
pip install -e ".[dev,opensees]"
```

### 4. Optional — UMFPACK (faster CPU direct solver) {#umfpack}

Enables [`scipy.umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack). Pick **one**
method for your platform:

| Platform | Install |
|----------|---------|
| **Linux** (pip / venv) | `pip install openseespy-solvers[umfpack]` — may need `libsuitesparse-dev` (Debian/Ubuntu) or your distro’s SuiteSparse package |
| **Linux / macOS / Windows** (conda) | `conda install -c conda-forge scikit-umfpack` |
| **macOS** (pip / venv) | `pip install openseespy-solvers[umfpack]` if wheels are available; otherwise conda-forge |
| **Windows** (pip / venv) | **Do not** use `pip install …[umfpack]` — PyPI ships source only. Use conda-forge |

See the [scikit-umfpack install guide](https://scikit-umfpack.github.io/scikit-umfpack/install.html).

### 5. Optional — GPU backends (NVIDIA) {#gpu}

Skip if you have no NVIDIA GPU or only need CPU solvers.

```bash
nvidia-smi
```

Read **`CUDA Version`** in the header (top right). Install **matching** wheels — do **not**
mix CUDA 12.x and 13.x in the same environment.

| `CUDA Version` | Install |
|----------------|---------|
| **12.x** | `pip install cupy-cuda12x` then `pip install "nvmath-python[cu12]>=0.9.0"` |
| **13.x** | `pip install cupy-cuda13x` then `pip install "nvmath-python[cu13]>=0.9.0"` |

Example (CUDA 13):

```bash
pip install cupy-cuda13x
pip install "nvmath-python[cu13]>=0.9.0"
```

If CuPy fails at import with missing CUDA libraries:

```bash
pip install "cupy-cuda13x[ctk]"   # or cupy-cuda12x[ctk]
```

Do **not** use `pip install -e ".[cupy]"` — it installs generic `cupy` and may build from
source. Always install `cupy-cuda12x` or `cupy-cuda13x` explicitly.

[`nvmath.direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) — recommended
for GPU linear solves. [`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) — recommended
for GPU eigen. Further detail: [GPU reference](#gpu-reference).

### 6. Check dependencies

```bash
pip check
```

Resolve any reported conflicts before verifying (see [Troubleshooting](#troubleshooting)).

### 7. Verify your install {#verify}

Run from the **repository root** with your environment activated. CuPy and nvMath tests in
`pytest` **run when those packages import** and are **skipped** otherwise.

#### 7a. Full test suite

```bash
pytest
```

Expect on the order of **~70 passed** on a full CPU+GPU stack. Skipped tests mean an
optional backend was not installed.

#### 7b. Example scripts

Each script builds a small brick bar, runs one analysis, and prints **Passed!** or
**Failed!**.

```bash
cd examples

# CPU — recommended quick check
python solvers/scipy_spsolve.py
python solvers/scipy_eigsh.py
python solvers/hybrid_spsolve.py

# CPU — optional
python solvers/scipy_umfpack.py      # needs UMFPACK (step 4)
python solvers/scipy_cg.py
python solvers/scipy_cg_jacobi.py
python solvers/scipy_gmres.py
python solvers/scipy_gmres_ilu.py

# GPU — when CuPy / nvMath are installed (step 5)
python solvers/cupy_spsolve.py
python solvers/cupy_eigsh.py
python solvers/nvmath_direct_solver.py

# Experimental — manual only (larger mesh; not run by pytest)
python solvers/scipy_lobpcg.py
python solvers/cupy_lobpcg.py
```

See [Examples](examples.md) for the full catalog.

#### 7c. Benchmarks

Compare PythonSparse timing against native OpenSees solvers on several mesh sizes:

```bash
cd examples    # skip if already there from 7b

python brick_bar.py
python brick_bar_eigen.py

# Finer meshes — slower, more equations
python brick_bar.py --large-test
python brick_bar_eigen.py --large-test
```

**Day-to-day:** `pytest` for exhaustive checks; `scipy_spsolve.py` + `scipy_eigsh.py` (or GPU
equivalents) for a fast smoke test; benchmarks when you want timing comparisons.

---

## Quick install (pip only) {#quick-install}

Use this when you only need the library inside an existing project — no examples,
benchmarks, or `pytest`.

```bash
pip install openseespy-solvers

# OpenSeesPy if not already in your environment
pip install openseespy-solvers[opensees]

# Faster CPU direct solver (Linux / macOS pip; Windows → conda-forge, see UMFPACK table)
pip install openseespy-solvers[umfpack]
```

**GPU** (after `nvidia-smi`, match CUDA generation — see [step 5](#gpu)):

```bash
pip install cupy-cuda13x
pip install "nvmath-python[cu13]>=0.9.0"
```

To run examples and the full [verification checklist](#verify), clone the repo and follow
[Full setup from scratch](#full-setup).

---

## GPU reference {#gpu-reference}

| Module | Library | Wheels |
|--------|---------|--------|
| `openseespy_solvers.cupy` | CuPy | `cupy-cuda12x` or `cupy-cuda13x` |
| `openseespy_solvers.nvmath` | nvmath-python | `nvmath-python[cu12]` or `[cu13]` |

SciPy solvers always run on the CPU. CuPy and nvMath need an NVIDIA GPU and wheels that
match `nvidia-smi`.

| Component | Notes |
|-----------|-------|
| NVIDIA driver | `nvidia-smi` must work |
| CUDA generation | **12.x or 13.x** — match the `CUDA Version` line from `nvidia-smi` |
| CuPy | ≥ 13 — use **`cupy-cuda12x`** / **`cupy-cuda13x`**, not generic `cupy` |
| nvmath-python | ≥ 0.9 — **`[cu12]`** or **`[cu13]`** on the same generation as CuPy |

A full CUDA Toolkit on disk is **not** required if wheels bundle runtimes (CuPy `[ctk]`).

nvMath auto-locates the cuDSS threading library from `nvidia-cudss-cu13` / `nvidia-cudss-cu12`
(e.g. `cudss_mtlayer_vcomp140.dll` under `site-packages/nvidia/cu13/bin/` on Windows). Override
with `CUDSS_THREADING_LIB` or `multithreading_lib=...` on
:func:`~openseespy_solvers.nvmath.direct_solver`.

Further reading:

- [CuPy installation](https://docs.cupy.dev/en/stable/install.html)
- [nvmath-python installation](https://docs.nvidia.com/cuda/nvmath-python/latest/installation.html)
- [`nvmath.direct_solver` API](api/nvmath.md#openseespy_solvers.nvmath.direct_solver)

---

## Optional extras reference

| Extra | Installs | Enables |
|-------|----------|---------|
| *(base)* | NumPy, SciPy | `openseespy_solvers.scipy` |
| `opensees` | openseespy | OpenSeesPy when missing |
| `umfpack` | scikit-umfpack | `scipy.umfpack` — see [UMFPACK](#umfpack) for platform notes |
| `cupy` | generic `cupy>=13` | Prefer explicit `cupy-cuda12x` / `cupy-cuda13x` wheels |
| `dev` | pytest, ruff, build | Development and [verification](#verify) |
| `docs` | mkdocs, mkdocstrings | Build documentation |

CuPy and nvMath are **not** pinned pip extras — install CUDA-matched wheels from [step 5](#gpu).

---

## Build documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000`.

---

## Troubleshooting {#troubleshooting}

### Mixed CUDA 12 / 13 packages

If `pip check` reports conflicts (e.g. `nvidia-cudss-cu12` vs `cuda-toolkit 13.*`), remove
leftover `*-cu12` packages and reinstall the matching stack, or recreate the environment
([step 1](#1-create-a-python-environment)) and repeat [full setup](#full-setup).

### scikit-umfpack fails on Windows with “Program 'swig' not found”

Use `conda install -c conda-forge scikit-umfpack` instead of
`pip install openseespy-solvers[umfpack]`. See the [UMFPACK](#umfpack) table.

### CuPy import errors (missing CUDA libraries)

```bash
pip install "cupy-cuda13x[ctk]"   # or cupy-cuda12x[ctk]
```

Ensure `nvidia-smi` works and wheel CUDA generation matches the driver.
