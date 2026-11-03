"""The tutorial's token features, adapted to sentence pairs."""

import numpy as np
from part1.clslib import Encoder, TokenIndicator


class BagOfWords(Encoder):
    """Count token occurrences in one sentence.
    Source: the Bag-of-Words exercise in part1/tutorial.ipynb.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text):
        vector = np.zeros(len(self), dtype=np.float32)
        for token_id in self.tokenizer.encode(text).ids:
            vector[token_id] += 1
        return vector


class PairEncoder(Encoder):
    """Join separate premise and hypothesis vectors into one feature vector.
    Source: part1.clslib.Encoder, adapted to our MNLI pairs.
    """

    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder

    def __len__(self):
        return 2 * len(self.sentence_encoder)

    def __call__(self, pair):
        premise, hypothesis = pair
        return np.concatenate([
            self.sentence_encoder(premise),
            self.sentence_encoder(hypothesis),
        ]).astype(np.float32)


def make_encoder(tokenizer, representation="indicator"):
    """Choose the tutorial representation and wrap it for sentence pairs."""
    if representation == "indicator":
        return PairEncoder(TokenIndicator(tokenizer))
    if representation == "bow":
        return PairEncoder(BagOfWords(tokenizer))
    raise ValueError(f"Unknown representation: {representation}. Use indicator or bow.")


# TODO: Add TF-IDF as the third representation, fitted only on training data.
# These dense vectors keep the original part1 training interface unchanged.
