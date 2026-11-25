"""Behavioral checks for the Markov-chain package."""

import numpy as np
import pytest
import sympy as sp

from markov import (
    NonUniqueStationaryState,
    diagnose_chain,
    distribution_after_steps,
    empirical_distribution,
    exact_transition_matrix,
    generate,
    matrix_powers,
    steady_state_by_eigenvector,
    steady_state_by_system,
    text_to_matrix,
)


def test_transition_rows_sum_to_one_for_each_dead_end_policy() -> None:
    for policy in ("wrap", "end_start", "uniform"):
        _, transition, _ = text_to_matrix("red blue green", dead_end_policy=policy)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)


def test_exact_transition_matrix_uses_fractions() -> None:
    counts, _, words = text_to_matrix("a b a c")
    exact = exact_transition_matrix(counts)

    assert words == ["a", "b", "c"]
    assert exact[0, 1] == sp.Rational(1, 2)
    assert exact[0, 2] == sp.Rational(1, 2)


def test_stationary_solvers_agree_and_satisfy_definition() -> None:
    counts, transition, _ = text_to_matrix("the cat sat sat dog")
    exact = exact_transition_matrix(counts)
    by_system = steady_state_by_system(exact)
    by_eigenvector = steady_state_by_eigenvector(exact)

    assert by_system == by_eigenvector
    assert by_system == sp.Matrix([sp.Rational(1, 5), sp.Rational(1, 5), sp.Rational(2, 5), sp.Rational(1, 5)])
    np.testing.assert_allclose(
        np.asarray(by_system, dtype=float).reshape(-1) @ transition,
        np.asarray(by_system, dtype=float).reshape(-1),
    )


def test_non_unique_stationary_distribution_is_explicit() -> None:
    transition = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(NonUniqueStationaryState):
        steady_state_by_system(transition)


def test_generation_is_seeded_and_uses_only_valid_transitions() -> None:
    _, transition, words = text_to_matrix("a b a c")
    sequence = generate(transition, words, "a", 500, seed=19)
    repeated = generate(transition, words, "a", 500, seed=19)
    valid_edges = {
        (words[source], words[target])
        for source in range(len(words))
        for target in np.flatnonzero(transition[source])
    }

    assert sequence == repeated
    assert all(edge in valid_edges for edge in zip(sequence, sequence[1:]))
    assert np.isclose(empirical_distribution(sequence, words).sum(), 1.0)


def test_matrix_powers_and_distribution_evolution() -> None:
    _, transition, _ = text_to_matrix("a b a c")
    powers = matrix_powers(transition)
    initial = np.array([1.0, 0.0, 0.0])

    assert set(powers) == {1, 2, 5, 10, 50}
    np.testing.assert_allclose(distribution_after_steps(initial, transition, 2), initial @ powers[2])


def test_diagnostics_detect_reducibility_and_periodicity() -> None:
    periodic = np.array([[0.0, 1.0], [1.0, 0.0]])
    reducible = np.eye(2)

    assert diagnose_chain(periodic)["periodic"] is True
    assert diagnose_chain(periodic)["reducible"] is False
    assert diagnose_chain(reducible)["reducible"] is True
    assert "multiple stationary distributions" in " ".join(
        diagnose_chain(reducible)["warnings"]
    ).lower()