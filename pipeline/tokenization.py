"""Turn MNLI sentences into words and GloVe vocabulary IDs.
Tokenizer and BasicTokenizer come from the course's part2/clslib.py and are
demonstrated in part2/tutorial.ipynb.
"""

import re

import nltk

from .embeddings import Vocabulary


class Tokenizer:
    #The tutorial interface for token strings and token IDs.

    def tokens(self, text: str) -> list:
        """Tokenize as a list of string tokens."""
        raise NotImplementedError("Deze nog doen")

    def ids(self, text: str) -> list:
        """Tokenize as a list of token IDs."""
        raise NotImplementedError("Deze nog doen")


class BasicTokenizer(Tokenizer):
    #tutorial's NLTK word tokenization and text filetering options

    def __init__(
        self,
        lowercase: bool = True,  #<<<<<<<< deze 3 nog kijken voor experiment
        remove_punct: bool = False,  #<<<<<<<<
        remove_stopwords: bool = False,  ##<<<<<<<< not is still there
        language="english",
        vocab: Vocabulary = None,
    ):
        #inladen
        self.vocab = vocab
        self.language = language
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:#stopwoord switch
            self.stopwords = set(nltk.corpus.stopwords.words(language))

    def tokens(self, text) -> list[str]:
        """Return token strings before unknown words are mapped to the default ID."""
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)
        tokens = nltk.tokenize.word_tokenize(text, language=self.language)
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def ids(self, text) -> list[int]:
        # ids uit tokens
        ids = [self.vocab.index(t) for t in self.tokens(text)]
        return ids or [self.vocab.default_index]  # als lege lijst uitkomt -> [UNK]


def tokenize_pairs(examples, tokenizer):  # allebij door tokenizer en weer bij elkaar met label 
    return [
        ((tokenizer.ids(premise), tokenizer.ids(hypothesis)), int(label)) 
        for (premise, hypothesis), label in examples
    ]


# TODO measure token/type coverage from tokens(), before mapping to UNK.
