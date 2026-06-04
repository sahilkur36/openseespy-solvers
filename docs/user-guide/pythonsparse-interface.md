# PythonSparse interface

OpenSeesPy passes sparse matrix data to solver objects as memoryviews. Solvers
in this package wrap those buffers with NumPy, assemble a backend sparse matrix,
invoke the numerical routine, and write results in place.

## Linear solve

OpenSees calls `solver.solve(**kwargs)`.

**Keyword arguments**

`index_ptr`, `indices`, `values`
: CSR/CSC sparse structure and coefficients.

`rhs`, `x`
: Right-hand side and solution buffers. `x` is overwritten.

`num_eqn`, `nnz`
: Matrix order and number of nonzeros.

`matrix_status`
: One of `'STRUCTURE_CHANGED'`, `'COEFFICIENTS_CHANGED'`, `'UNCHANGED'`.

`storage_scheme`
: `'CSR'` or `'CSC'`. Default is `'CSR'`.

**Returns**

`0` if the solve succeeded; a negative integer otherwise. Set `debug=True` on
the solver to re-raise the underlying exception.

## Eigen solve

OpenSees calls `solver.solve(**kwargs)` for eigen analysis.

Additional keyword arguments: `k_values`, `m_values`, `eigenvalues`,
`eigenvectors`, `num_modes`, `find_smallest`.

Eigen solvers raise on failure; they do not return a status code.

## formAp

Linear solvers implement `formAp(**kwargs)` to compute `Ap = A @ p` without a
full solve. OpenSees supplies read-only `p` and writable `Ap`.

## Matrix status

OpenSees indicates how the system matrix changed since the previous solve:

`STRUCTURE_CHANGED`
: Rebuild the sparse matrix (new index structure).

`COEFFICIENTS_CHANGED`
: Update coefficients in place; sparsity pattern unchanged.

`UNCHANGED`
: Reuse the cached matrix. Direct solvers also reuse the LU factorization.

## See Also

[solver objects](solver-objects.md), [to_openseespy()](to-openseespy.md)

[OpenSees PythonSparse documentation](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
