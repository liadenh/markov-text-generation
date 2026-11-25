"""Matrix powers and two stationary-distribution calculations."""

from collections.abc import Sequence

import numpy as np
import sympy as sp


class NonUniqueStationaryState(ValueError):
    """Raised when the stationary equations admit more than one solution."""


def _as_sympy_matrix(transition: np.ndarray | sp.MatrixBase) -> sp.Matrix:
    """Convert numeric probabilities to simple exact fractions when possible."""
    if isinstance(transition, sp.MatrixBase):
        return sp.Matrix(transition)
    return sp.Matrix(transition).applyfunc(lambda value: sp.nsimplify(float(value)))


def matrix_powers(
    transition: np.ndarray,
    exponents: Sequence[int] = (1, 2, 5, 10, 50),
) -> dict[int, np.ndarray]:
    """Compute selected powers P^k using the row-stochastic convention."""
    if any(exponent < 1 for exponent in exponents):
        raise ValueError("matrix power exponents must be positive")
    return {exponent: np.linalg.matrix_power(transition, exponent) for exponent in exponents}


def distribution_after_steps(
    initial: Sequence[float],
    transition: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Compute x_k = x_0 P^k for a row-vector distribution."""
    initial_array = np.asarray(initial, dtype=float)
    if initial_array.ndim != 1 or initial_array.size != transition.shape[0]:
        raise ValueError("initial distribution must match the matrix dimension")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not np.isclose(initial_array.sum(), 1.0) or np.any(initial_array < 0):
        raise ValueError("initial distribution must be nonnegative and sum to 1")
    return initial_array @ np.linalg.matrix_power(transition, steps)


def distribution_history(
    initial: Sequence[float],
    transition: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return x_0 through x_steps, inclusive, for a row-vector distribution."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = np.asarray(initial, dtype=float)
    if current.ndim != 1 or current.size != transition.shape[0]:
        raise ValueError("initial distribution must match the matrix dimension")
    if not np.isclose(current.sum(), 1.0) or np.any(current < 0):
        raise ValueError("initial distribution must be nonnegative and sum to 1")
    history = [current.copy()]
    for _ in range(steps):
        current = current @ transition
        history.append(current.copy())
    return np.vstack(history)


def stationary_system(
    transition: np.ndarray | sp.MatrixBase,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Build A and b for (P^T - I) pi^T = 0 and sum(pi) = 1."""
    matrix = _as_sympy_matrix(transition)
    if matrix.rows != matrix.cols:
        raise ValueError("transition matrix must be square")
    size = matrix.rows
    equations = matrix.T - sp.eye(size)
    equations = equations.col_join(sp.ones(1, size))
    values = sp.zeros(size, 1).col_join(sp.ones(1, 1))
    return equations, values


def format_stationary_system(
    transition: np.ndarray | sp.MatrixBase,
    words: Sequence[str] | None = None,
) -> str:
    """Render the stationary equations as a labeled augmented matrix."""
    coefficients, values = stationary_system(transition)
    names = [f"pi_{word}" for word in words] if words else [
        f"pi_{index}" for index in range(coefficients.cols)
    ]
    augmented = coefficients.row_join(values)
    return f"A = [P^T - I; 1^T], b = [0; 1]\n{sp.pretty(augmented)}\nVariables: {', '.join(names)}"


def steady_state_by_system(
    transition: np.ndarray | sp.MatrixBase,
) -> sp.Matrix:
    """Solve the normalized stationary equations exactly when possible."""
    coefficients, values = stationary_system(transition)
    solution_set = sp.linsolve((coefficients, values))
    if solution_set is sp.EmptySet or not solution_set:
        raise ValueError("stationary system has no solution")
    solution = next(iter(solution_set))
    if any(value.free_symbols for value in solution):
        raise NonUniqueStationaryState(
            f"stationary distribution is not unique; solution family: {solution}"
        )
    result = sp.Matrix(solution)
    if any(value.is_real is False or value.is_nonnegative is False for value in result):
        raise ValueError("stationary solution is not a probability distribution")
    return result


def steady_state_by_eigenvector(
    transition: np.ndarray | sp.MatrixBase,
) -> sp.Matrix:
    """Find and normalize a right eigenvector of P^T for eigenvalue 1."""
    matrix = _as_sympy_matrix(transition)
    eigenvectors = matrix.T.eigenvects()
    basis = next((vectors for value, _, vectors in eigenvectors if value == 1), [])
    if len(basis) != 1:
        raise NonUniqueStationaryState(
            f"eigenvalue 1 has eigenspace dimension {len(basis)}"
        )
    vector = sp.Matrix(basis[0])
    total = sp.simplify(sum(vector))
    if total == 0:
        raise ValueError("eigenvector cannot be normalized to sum to 1")
    result = vector / total
    if any(value.is_real is False or value.is_nonnegative is False for value in result):
        raise ValueError("normalized eigenvector is not a probability distribution")
    return result