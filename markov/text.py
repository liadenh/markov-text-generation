"""Text normalization and tokenization for the Markov model."""

import re


def tokenize(text: str, keep_period: bool = False) -> list[str]:
    """Return lowercase word tokens, optionally preserving periods as a token."""
    pattern = r"\w+|\." if keep_period else r"\w+"
    return re.findall(pattern, text.lower())