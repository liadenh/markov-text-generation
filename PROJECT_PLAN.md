# Project Plan

## Goal

Demonstrate text generation with a finite-state Markov chain using explicit matrices and linear algebra, without machine-learning libraries.

## Implementation Plan and Status

1. **Project scaffold and dependencies**: create `markov/`, `examples/`, `data/`, `tests/`, `figures/`, and declare NumPy, Matplotlib, SymPy, and pytest. **Complete**
2. **Text and transition matrices**: tokenize consistently, count bigrams, document the row-stochastic convention, support configurable dead-end policies, and verify row sums. **Complete**
3. **Generation and matrix calculations**: add seeded sentence generation, selected matrix powers, and row-distribution evolution. **Complete**
4. **Stationary distributions**: solve the augmented system and the eigenvector problem, preserve exact fractions for small examples, and detect non-unique solutions. **Complete**
5. **Diagnostics and plots**: detect reducible/periodic structure and save the transition heatmap, frequency comparison, and convergence plot. **Complete**
6. **Examples and tests**: add exact small and full-text examples, a public-domain sample, and focused pytest coverage. **Complete**
7. **README creation**: document setup, conventions, commands, project layout, and interpretation of results. **Complete**
8. **Notebook walkthrough**: create an executable report notebook covering README file I/O, project-plan tracking, and the Markov-chain workflow. **In progress**
9. **Final validation**: run the notebook, full test suite, examples, and inspect generated figures. **In progress**

## Plan Review

The order follows the dependency structure: matrix construction precedes generation and stationary-state work, and examples depend on the finished package API. The main mathematical risks are handled explicitly: dead-end policies alter the chain, reducibility may make the stationary distribution non-unique, and periodicity can prevent `P^k` from converging. Exact arithmetic is used for the small example; numeric computations are used for larger text and simulation.
