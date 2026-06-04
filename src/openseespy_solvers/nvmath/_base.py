"""nvMath backend hooks shared by all nvmath-namespace solvers."""

from __future__ import annotations

import importlib.resources
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers.exceptions import UnsupportedStorageSchemeError


def _import_nvmath() -> Any:
    """Import :mod:`nvmath` lazily.

    ``nvmath-python`` is an optional dependency; importing the nvmath namespace
    must never require it. The import is deferred until a solver is constructed.

    Returns
    -------
    module
        The imported :mod:`nvmath` module.

    Raises
    ------
    ImportError
        If ``nvmath-python`` is not installed.
    """
    try:
        import nvmath
    except ImportError as exc:  # pragma: no cover - exercised only without nvmath
        raise ImportError(
            "The 'nvmath' backend requires nvmath-python. "
            "Install with: pip install \"nvmath-python[cu13]\" (or [cu12]; see installation docs)"
        ) from exc
    return nvmath


def _is_cupy_sparse(sp_module: Any) -> bool:
    name = getattr(sp_module, "__name__", "")
    return "cupy" in name.lower()


def _resolve_sp_module(
    sp_module: Any | None,
    *,
    device: str | None = None,
) -> tuple[Any, bool]:
    """Return ``(sparse_module, is_gpu)`` for matrix assembly."""
    if sp_module is not None:
        is_gpu = _is_cupy_sparse(sp_module)
        if is_gpu:
            try:
                import cupy  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "CuPy sparse matrices require CuPy. "
                    "Install with: pip install openseespy-solvers[cupy]"
                ) from exc
        return sp_module, is_gpu

    if device == "cpu":
        return sp, False
    if device == "gpu":
        try:
            import cupyx.scipy.sparse as csp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "device='gpu' requires CuPy. "
                "Install with: pip install openseespy-solvers[cupy]"
            ) from exc
        return csp, True

    try:
        import cupyx.scipy.sparse as csp
    except ImportError:
        return sp, False
    return csp, True


def _find_multithreading_lib() -> str | None:
    """Locate the cuDSS multithreading layer library, if installed.

    nvMath uses this for multi-threaded CPU work during plan/factorize (especially
    with hybrid execution). See ``CUDSS_THREADING_LIB`` and NVIDIA's cuDSS docs.
    """
    env = os.environ.get("CUDSS_THREADING_LIB")
    if env:
        path = Path(env)
        if path.is_file():
            return str(path.resolve())

    if sys.platform == "win32":
        lib_names = (
            "cudss_mtlayer_vcomp140.dll",
            "libcudss_mtlayer_vcomp.dll",
            "cudss_mtlayer_vcomp.dll",
        )
        subdirs = ("bin", "lib")
    else:
        lib_names = (
            "libcudss_mtlayer_gomp.so.0",
            "libcudss_mtlayer_gomp.so",
        )
        subdirs = ("lib",)

    search_roots: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_roots.append(Path(conda_prefix))

    for cuda_pkg in ("nvidia.cu13", "nvidia.cu12"):
        try:
            search_roots.append(Path(str(importlib.resources.files(cuda_pkg))))
        except (ImportError, AttributeError, TypeError, ModuleNotFoundError):
            pass

    for entry in sys.path:
        base = Path(entry)
        if not base.is_dir():
            continue
        for cuda_ver in ("cu13", "cu12"):
            root = base / "nvidia" / cuda_ver
            if root.is_dir():
                search_roots.append(root)

    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for subdir in subdirs:
            for lib_name in lib_names:
                lib_path = root / subdir / lib_name
                if lib_path.is_file():
                    return str(lib_path.resolve())

    if sys.platform != "win32":
        for sys_path in (Path("/usr/local/lib"), Path("/usr/lib"), Path("/lib")):
            for lib_name in lib_names:
                lib_path = sys_path / lib_name
                if lib_path.is_file():
                    return str(lib_path.resolve())

    return None


class NvMathMixin:
    """Implements backend hooks for nvMath direct solvers (CPU or GPU sparse)."""

    backend = "nvmath"

    def __init__(
        self,
        *args: Any,
        sp_module: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._nvmath = _import_nvmath()
        self._sp_module, self._on_device = _resolve_sp_module(sp_module, device=device)
        if self._on_device:
            import cupy as cp

            self._np_module = cp
        else:
            self._np_module = np
        super().__init__(*args, **kwargs)

    @property
    def _cupy_dtype(self) -> Any:
        cp = self._np_module
        return cp.float32 if self._compute_dtype == np.float32 else cp.float64

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> Any:
        spm = self._sp_module
        if self._on_device:
            cp = self._np_module
            data = cp.asarray(values, dtype=self._cupy_dtype)
            ind = cp.asarray(indices, dtype=cp.int32)
            ptr = cp.asarray(indptr, dtype=cp.int32)
            if fmt == "CSR":
                matrix = spm.csr_matrix((data, ind, ptr), shape=shape)
            elif fmt == "CSC":
                matrix = spm.csc_matrix((data, ind, ptr), shape=shape).tocsr()
            else:
                raise UnsupportedStorageSchemeError(
                    f"nvMath backend does not support scheme {fmt!r}"
                )
            return matrix

        if fmt == "CSR":
            matrix = spm.csr_matrix(
                (values.copy(), indices.copy(), indptr.copy()), shape=shape
            )
        elif fmt == "CSC":
            matrix = spm.csc_matrix(
                (values.copy(), indices.copy(), indptr.copy()), shape=shape
            ).tocsr()
        else:
            raise UnsupportedStorageSchemeError(
                f"nvMath backend does not support scheme {fmt!r}"
            )
        return matrix

    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        if self._on_device:
            matrix.data[:] = self._np_module.asarray(values, dtype=self._cupy_dtype)
        else:
            matrix.data[:] = np.asarray(values, dtype=self._compute_dtype)
        return matrix

    def _to_device(self, array: np.ndarray) -> Any:
        if self._on_device:
            return self._np_module.asarray(array, dtype=self._cupy_dtype)
        return np.asarray(array, dtype=self._compute_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        if self._on_device:
            return self._np_module.asnumpy(array)
        return np.asarray(array, dtype=self._compute_dtype)

    def _matvec(self, matrix: Any, vector: Any) -> Any:
        if self._on_device:
            return matrix @ self._np_module.asarray(vector, dtype=self._cupy_dtype)
        return matrix @ np.asarray(vector, dtype=self._compute_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return self._sp_module.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        if self._on_device:
            from cupyx.scipy.sparse import linalg as cspla

            return isinstance(obj, cspla.LinearOperator)
        return isinstance(obj, spla.LinearOperator)
