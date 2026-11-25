"""Matplotlib figures used by the examples and notebook."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_transition_heatmap(
    transition: np.ndarray,
    words: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Save a labeled heatmap of the transition matrix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(6, len(words) * 0.65), max(5, len(words) * 0.55)))
    image = axis.imshow(transition, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(words)), labels=words, rotation=45, ha="right")
    axis.set_yticks(range(len(words)), labels=words)
    axis.set_xlabel("Next state")
    axis.set_ylabel("Current state")
    axis.set_title("Transition probabilities")
    figure.colorbar(image, ax=axis, label="P(i, j)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_frequency_comparison(
    words: Sequence[str],
    stationary: Sequence[float],
    actual: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Save a side-by-side comparison of stationary and source frequencies."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(words))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(7, len(words) * 0.7), 4.5))
    axis.bar(positions - width / 2, stationary, width, label="Stationary")
    axis.bar(positions + width / 2, actual, width, label="Source text")
    axis.set_xticks(positions, labels=words, rotation=45, ha="right")
    axis.set_ylabel("Relative frequency")
    axis.set_title("Stationary probabilities and source-text frequencies")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_convergence(
    history: np.ndarray,
    stationary: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Save ||x_k - pi||_2 against iteration k."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    distances = np.linalg.norm(np.asarray(history) - np.asarray(stationary), axis=1)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(np.arange(len(distances)), distances, color="#176b5b")
    axis.set_xlabel("Steps k")
    axis.set_ylabel(r"$\|x_k - \pi\|_2$")
    axis.set_title("Convergence toward the stationary distribution")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path