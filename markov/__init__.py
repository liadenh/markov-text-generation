"""Linear-algebra demonstrations for text generation with Markov chains."""

from markov.matrix import exact_transition_matrix, format_matrix, text_to_matrix
from markov.text import tokenize
from markov.generation import empirical_distribution, generate
from markov.linear_algebra import (
	NonUniqueStationaryState,
	distribution_after_steps,
	distribution_history,
	format_stationary_system,
	matrix_powers,
	steady_state_by_eigenvector,
	steady_state_by_system,
	stationary_system,
)
from markov.diagnostics import diagnose_chain
from markov.plots import (
	plot_convergence,
	plot_frequency_comparison,
	plot_transition_heatmap,
)

__all__ = [
	"NonUniqueStationaryState",
	"diagnose_chain",
	"distribution_after_steps",
	"distribution_history",
	"empirical_distribution",
	"exact_transition_matrix",
	"format_matrix",
	"format_stationary_system",
	"generate",
	"matrix_powers",
	"plot_convergence",
	"plot_frequency_comparison",
	"plot_transition_heatmap",
	"stationary_system",
	"steady_state_by_eigenvector",
	"steady_state_by_system",
	"text_to_matrix",
	"tokenize",
]