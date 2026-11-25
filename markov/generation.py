"""Random generation and empirical distributions for a Markov chain."""

from collections.abc import Sequence

import numpy as np


def generate(
    transition: np.ndarray,
    words: Sequence[str],
    start_word: str,
    length: int,
    seed: int | None = None,
) -> list[str]:
    """Generate ``length`` states, including ``start_word`` as the first state."""
    if transition.shape != (len(words), len(words)):
        raise ValueError("transition matrix shape must match the number of words")
    if length < 1:
        raise ValueError("length must be at least 1")
    try:
        current = list(words).index(start_word)
    except ValueError as error:
        raise ValueError(f"unknown start word: {start_word!r}") from error

    rng = np.random.default_rng(seed)
    sequence = [str(words[current])]
    for _ in range(length - 1):
        probabilities = np.asarray(transition[current], dtype=float)
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(f"transition row for {words[current]!r} is not stochastic")
        current = int(rng.choice(len(words), p=probabilities))
        sequence.append(str(words[current]))
    return sequence


def empirical_distribution(sequence: Sequence[str], words: Sequence[str]) -> np.ndarray:
    """Return normalized observed frequencies in the supplied state order."""
    if not sequence:
        raise ValueError("sequence must contain at least one state")
    indices = {word: index for index, word in enumerate(words)}
    unknown = set(sequence) - indices.keys()
    if unknown:
        raise ValueError(f"sequence contains unknown states: {sorted(unknown)}")
    counts = np.zeros(len(words), dtype=float)
    for word in sequence:
        counts[indices[word]] += 1
    return counts / len(sequence)