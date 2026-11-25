"""Graph diagnostics for finite Markov chains."""

from collections.abc import Sequence
from math import gcd

import numpy as np


def _strongly_connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    """Return strongly connected components using Tarjan's algorithm."""
    indices = [-1] * len(adjacency)
    low_links = [0] * len(adjacency)
    stack: list[int] = []
    on_stack = [False] * len(adjacency)
    components: list[list[int]] = []
    next_index = 0

    def visit(node: int) -> None:
        nonlocal next_index
        indices[node] = low_links[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack[node] = True

        for neighbor in adjacency[node]:
            if indices[neighbor] == -1:
                visit(neighbor)
                low_links[node] = min(low_links[node], low_links[neighbor])
            elif on_stack[neighbor]:
                low_links[node] = min(low_links[node], indices[neighbor])

        if low_links[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack[member] = False
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in range(len(adjacency)):
        if indices[node] == -1:
            visit(node)
    return components


def _component_period(component: list[int], adjacency: list[list[int]]) -> int:
    """Calculate the gcd cycle length for a strongly connected component."""
    members = set(component)
    root = component[0]
    depths = {root: 0}
    pending = [root]
    while pending:
        node = pending.pop()
        for neighbor in adjacency[node]:
            if neighbor in members and neighbor not in depths:
                depths[neighbor] = depths[node] + 1
                pending.append(neighbor)

    period = 0
    for node in component:
        for neighbor in adjacency[node]:
            if neighbor in members:
                period = gcd(period, abs(depths[node] + 1 - depths[neighbor]))
    return period or 1


def diagnose_chain(
    transition: np.ndarray,
    words: Sequence[str] | None = None,
) -> dict[str, object]:
    """Report reducibility and periodic recurrent classes for a transition matrix."""
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition matrix must be square")
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("transition matrix must be row-stochastic")
    if np.any(matrix < 0):
        raise ValueError("transition probabilities cannot be negative")

    adjacency = [list(np.flatnonzero(row > 0)) for row in matrix]
    components = _strongly_connected_components(adjacency)
    component_of = {
        node: index for index, component in enumerate(components) for node in component
    }
    closed_components = []
    for component_index, component in enumerate(components):
        if all(
            component_of[neighbor] == component_index
            for node in component
            for neighbor in adjacency[node]
        ):
            closed_components.append(component)

    labels = list(words) if words is not None else [str(index) for index in range(len(matrix))]
    if len(labels) != len(matrix):
        raise ValueError("words must match the transition matrix dimension")
    component_labels = [[labels[node] for node in component] for component in components]
    closed_labels = [[labels[node] for node in component] for component in closed_components]
    recurrent_periods = {
        tuple(labels[node] for node in component): _component_period(component, adjacency)
        for component in closed_components
    }
    reducible = len(components) > 1
    periodic = any(period > 1 for period in recurrent_periods.values())
    warnings = []
    if reducible:
        warnings.append("Chain is reducible; stationary distributions may not be unique.")
    if periodic:
        warnings.append("A recurrent class is periodic; P^k may not converge.")
    if len(closed_components) > 1:
        warnings.append("Multiple closed classes imply multiple stationary distributions.")

    return {
        "reducible": reducible,
        "periodic": periodic,
        "components": component_labels,
        "closed_classes": closed_labels,
        "recurrent_periods": recurrent_periods,
        "warnings": warnings,
    }