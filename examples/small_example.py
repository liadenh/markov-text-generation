"""A compact, exact Markov-chain example suitable for hand calculation."""

from markov import (
    exact_transition_matrix,
    format_matrix,
    format_stationary_system,
    steady_state_by_eigenvector,
    steady_state_by_system,
    text_to_matrix,
)


def main() -> None:
    text = "the cat sat sat dog"
    counts, transition, words = text_to_matrix(text)
    exact_transition = exact_transition_matrix(counts)

    print(f"Text: {text!r}")
    print(f"State order: {words}")
    print("\nBigram counts:")
    print(format_matrix(counts, words, precision=0))
    print("\nExact transition matrix:")
    print(format_matrix(exact_transition, words))
    print("\nAugmented stationary system:")
    print(format_stationary_system(exact_transition, words))

    by_system = steady_state_by_system(exact_transition)
    by_eigenvector = steady_state_by_eigenvector(exact_transition)
    print(f"\nStationary state from system: {by_system.T}")
    print(f"Stationary state from eigenvector: {by_eigenvector.T}")
    print(f"Numeric transition rows sum to: {transition.sum(axis=1)}")


if __name__ == "__main__":
    main()