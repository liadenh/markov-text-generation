"""Run the complete Markov-chain analysis on the bundled sample paragraph."""

from collections import Counter
from pathlib import Path

import numpy as np

from markov import (
    diagnose_chain,
    distribution_history,
    empirical_distribution,
    exact_transition_matrix,
    generate,
    matrix_powers,
    plot_convergence,
    plot_frequency_comparison,
    plot_transition_heatmap,
    steady_state_by_eigenvector,
    steady_state_by_system,
    text_to_matrix,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    text = (ROOT / "data" / "sample.txt").read_text(encoding="utf-8")
    counts, transition, words = text_to_matrix(text, dead_end_policy="wrap")
    exact_transition = exact_transition_matrix(counts)
    diagnostics = diagnose_chain(transition, words)

    print(f"States ({len(words)}): {words}")
    print(f"Transition matrix P: {transition.shape[0]} x {transition.shape[1]}")
    print(f"Maximum row-sum error: {np.max(np.abs(transition.sum(axis=1) - 1)):.2e}")
    print("\nGraph diagnostics:")
    print(
        f"reducible={diagnostics['reducible']}, periodic={diagnostics['periodic']}, "
        f"strongly connected components={len(diagnostics['components'])}, "
        f"closed classes={len(diagnostics['closed_classes'])}, "
        f"recurrent periods={list(diagnostics['recurrent_periods'].values())}"
    )
    for warning in diagnostics["warnings"]:
        print(f"WARNING: {warning}")

    initial = np.zeros(len(words))
    start_index = words.index(tokenize(text)[0])
    initial[start_index] = 1.0
    steps = 50
    history = distribution_history(initial, transition, steps)

    try:
        by_system = steady_state_by_system(exact_transition)
        by_eigenvector = steady_state_by_eigenvector(exact_transition)
        if by_system != by_eigenvector:
            raise ArithmeticError("stationary methods returned different exact answers")
    except ValueError as error:
        print(f"\nStationary state unavailable as a unique vector: {error}")
        return

    stationary = np.array([float(value) for value in by_system])
    print(
        "\nSolved the augmented stationary system (P^T - I) pi^T = 0, "
        f"1^T pi = 1 with {len(words) + 1} equations."
    )
    print("Stationary state (system and eigenvector agree):")
    print({word: round(float(stationary[index]), 4) for index, word in enumerate(words)})

    print("\nConvergence of selected P^k rows toward pi:")
    for exponent, power in matrix_powers(transition).items():
        row = power[start_index]
        distance = np.linalg.norm(row - stationary)
        maximum_distance = np.max(np.linalg.norm(power - stationary, axis=1))
        visible_row = {
            word: round(float(row[index]), 3)
            for index, word in enumerate(words)
            if row[index] >= 0.01
        }
        print(
            f"k={exponent:>2}: start-row distance={distance:.4f}, "
            f"maximum row distance={maximum_distance:.4f}, row={visible_row}"
        )
    print(
        f"\nx_{steps} from the initial state: "
        f"{dict(zip(words, np.round(history[-1], 4).tolist()))}"
    )

    source_tokens = tokenize(text)
    source_counts = Counter(source_tokens)
    actual = np.array([source_counts.get(word, 0) for word in words], dtype=float)
    actual /= actual.sum()
    generated = generate(transition, words, source_tokens[0], 100_000, seed=2026)
    generated_frequencies = empirical_distribution(generated, words)

    print("\nFrequency comparison (stationary / source / generated):")
    for index, word in enumerate(words):
        print(
            f"{word:>12}: {stationary[index]:.4f} / "
            f"{actual[index]:.4f} / {generated_frequencies[index]:.4f}"
        )
    print(
        "\nGenerated frequencies approximate the stationary distribution for an "
        "ergodic chain; source frequencies are finite-sample counts from one paragraph."
    )

    figures = ROOT / "figures"
    plot_transition_heatmap(transition, words, figures / "transition_heatmap.png")
    plot_frequency_comparison(
        words, stationary, actual, figures / "frequency_comparison.png"
    )
    plot_convergence(history, stationary, figures / "convergence.png")
    print(f"\nSaved plots to {figures.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()