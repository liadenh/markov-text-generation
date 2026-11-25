"""Build and display row-stochastic transition matrices from text."""

from collections.abc import Sequence

import numpy as np
import sympy as sp

from markov.text import tokenize

DEAD_END_POLICIES = ("wrap", "end_start", "uniform")
END_TOKEN = "<END>"
START_TOKEN = "<START>"


def text_to_matrix(
    text: str,
    dead_end_policy: str = "wrap",
    keep_period: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return counts, transition probabilities, and states in first-seen order.

    The matrix is row-stochastic: P[i, j] is the probability of moving from
    state i to state j, so row distributions evolve as x_(k+1) = x_k P.
    A dead-end row is handled according to ``dead_end_policy``:

    * ``wrap`` (default): transition back to the first token in the text.
    * ``end_start``: transition through synthetic <END> and <START> states.
    * ``uniform``: jump with equal probability to each real word state.
    These choices change the chain and can change its steady state.
    """
    if dead_end_policy not in DEAD_END_POLICIES:
        choices = ", ".join(DEAD_END_POLICIES)
        raise ValueError(f"dead_end_policy must be one of: {choices}")

    tokens = tokenize(text, keep_period=keep_period)
    if not tokens:
        raise ValueError("text must contain at least one token")

    words = list(dict.fromkeys(tokens))
    word_indices = {word: index for index, word in enumerate(words)}
    counts = np.zeros((len(words), len(words)), dtype=int)

    for current, following in zip(tokens, tokens[1:]):
        counts[word_indices[current], word_indices[following]] += 1

    dead_ends = np.flatnonzero(counts.sum(axis=1) == 0)
    if dead_end_policy == "wrap":
        counts[dead_ends, word_indices[tokens[0]]] = 1
    elif dead_end_policy == "uniform":
        counts[dead_ends, :len(words)] = 1
    elif dead_end_policy == "end_start":
        counts = np.pad(counts, ((0, 2), (0, 2)))
        end_index = len(words)
        start_index = len(words) + 1
        counts[dead_ends, end_index] = 1
        counts[end_index, start_index] = 1
        counts[start_index, word_indices[tokens[0]]] = 1
        words.extend((END_TOKEN, START_TOKEN))

    row_totals = counts.sum(axis=1)
    if np.any(row_totals == 0):
        raise ValueError("every state must have at least one outgoing transition")
    transition = counts / row_totals[:, np.newaxis]
    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ArithmeticError("transition matrix rows must sum to 1")

    return counts, transition, words


def exact_transition_matrix(counts: np.ndarray) -> sp.Matrix:
    """Convert integer bigram counts to an exact SymPy probability matrix."""
    rows = []
    for row in counts:
        total = int(row.sum())
        if total == 0:
            raise ValueError("count matrix contains a state with no successors")
        rows.append([sp.Rational(int(value), total) for value in row])
    return sp.Matrix(rows)


def format_matrix(
    matrix: np.ndarray | sp.Matrix,
    words: Sequence[str],
    precision: int = 3,
) -> str:
    """Format a matrix as a labeled table, preserving SymPy fractions."""
    values = matrix.tolist() if isinstance(matrix, sp.MatrixBase) else matrix.tolist()
    labels = [str(word) for word in words]
    rendered = []
    for row in values:
        rendered.append([
            str(value) if isinstance(value, sp.Basic) else f"{float(value):.{precision}f}"
            for value in row
        ])

    row_label_width = max(len("from / to"), *(len(label) for label in labels))
    column_widths = [
        max(len(labels[column]), *(len(row[column]) for row in rendered))
        for column in range(len(labels))
    ]
    header = "from / to".ljust(row_label_width) + "  " + "  ".join(
        label.rjust(column_widths[index]) for index, label in enumerate(labels)
    )
    lines = [header]
    for label, row in zip(labels, rendered):
        lines.append(label.ljust(row_label_width) + "  " + "  ".join(
            value.rjust(column_widths[index]) for index, value in enumerate(row)
        ))
    return "\n".join(lines)