# to_opensees

Return the configuration dictionary for OpenSeesPy `PythonSparse` commands.

## Usage

```python
solver = cg(rtol=1e-8)
cfg = solver.to_opensees()
ops.system("PythonSparse", cfg)
```

```python
lam = ops.eigen("PythonSparse", num_modes, eigsolver.to_opensees())
```

## Return value

The returned dict contains:

`solver`
: The solver instance (OpenSees retains a reference and calls its methods).

`scheme`
: Sparse storage scheme (`'CSR'` by default).

`writable`
: Writable buffer policy (included when set at construction).

## Parameters

`scheme`
: Override the storage scheme at call time.

`writable`
: Override the writable buffer list (for example `'values'` or
  `['values', 'rhs']`). Default at construction is `'none'`.

## Notes

Use `copy.copy(solver)` to obtain a fresh instance with the same configuration
but empty internal state. OpenSees may clone solvers when copying a system of
equations.

## See Also

[`BaseOpenSeesSolver.to_opensees`](../api/scipy.md) (documented on solver factories)

[solver objects](solver-objects.md)
