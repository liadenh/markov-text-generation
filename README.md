# Text Generation with Markov Chains

This is my Linear Algebra project about generating text with Markov chains. The program counts which words follow each other, turns those counts into a transition matrix, and uses the matrix to generate text and study long-run word frequencies. I use NumPy, Matplotlib, and SymPy, but no machine-learning libraries.

## How the model works

Each unique lowercase word is a state. In the transition matrix, `P[i, j]` is the probability that word `j` follows word `i`. The matrix is row-stochastic, so a row distribution changes by `x_(k+1) = x_k P`.

If the last word has no observed next word, the default is to wrap back to the first word. Other dead-end choices are available in the code and can change the results.

## Run the project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.small_example
python -m examples.full_example
```

Run the tests with `python -m pytest -q`. The notebook walks through the calculations, and the full example saves its plots in `figures/`.

## Files

- `markov/` contains the code for the model and calculations.
- `examples/` contains a small worked example and a longer example.
- `data/sample.txt` is the plain-text input story for the full example.
- `tests/` contains checks for the matrix, steady state, and generated text.
- `notebook.ipynb` contains the project walkthrough.
