# Installation

## Requirements

- Python ≥ 3.12
- NumPy ≥ 1.26
- SciPy ≥ 1.12 (needed for `rtol` / `atol` on sparse iterative solvers)

These minimums are enforced in `pyproject.toml`. The project is developed and tested on
Python 3.12 with current NumPy 2.x and SciPy 1.12+ releases.

## Standard install

```bash
pip install openseespy-solvers
```

## Optional dependencies

**GPU backend**

```bash
pip install openseespy-solvers[gpu]
```

Requires [CuPy](https://docs.cupy.dev/) **≥ 13** built for your CUDA version (for example
`pip install cupy-cuda12x` on CUDA 12). The `[gpu]` extra installs `cupy>=13.0`; pick the
wheel that matches your toolkit if the generic package is not right for your machine.

**OpenSeesPy** (for running the repository examples)

```bash
pip install openseespy-solvers[opensees]
```

**Documentation build**

```bash
pip install openseespy-solvers[docs]
```

**Development**

```bash
pip install openseespy-solvers[dev]
```

## Install from source

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers
pip install -e ".[dev]"
pytest
```

## Build documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site is served at `http://127.0.0.1:8000`.
