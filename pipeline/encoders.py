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
    if representation == "overlap":
        return OverlapEncoder(tokenizer)
    raise ValueError(f"Unknown representation: {representation}. Use indicator, bow or overlap.")


class OverlapEncoder(Encoder):
    """Token indicators for how premise and hypothesis relate:
    tokens in both, tokens only in the premise, tokens only in the hypothesis.
    Our third representation: unlike PairEncoder, it can see that a token
    occurs in both sentences.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def __len__(self):
        return 3 * self.vocab_size

    def __call__(self, pair):
        premise, hypothesis = pair
        P = set(self.tokenizer.encode(premise).ids)
        H = set(self.tokenizer.encode(hypothesis).ids)
        V = self.vocab_size

        vector = np.zeros(len(self), dtype=np.float32)
        vector[list(P & H)] = 1                       # block 1: in both
        vector[[V + i for i in P - H]] = 1            # block 2: only premise
        vector[[2 * V + i for i in H - P]] = 1        # block 3: only hypothesis
        return vector
