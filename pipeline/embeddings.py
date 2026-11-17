"""Load pretrained word vectors and their vocabulary for MNLI.

The vocabulary and loader come from the course's part2/clslib.py. The download
helper follows part2/tutorial.ipynb, using Python in place of notebook !wget.
Our small changes are marked with inline comments.

Premise and hypothesis share the same vocabulary and word vectors.

Course license: licenses/NLP1-COURSE-LICENSE (Apache-2.0).
"""

import gzip
from io import TextIOWrapper
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


def download_pretrained(option="glove", output_dir=None):
    #Download the tutorial's GloVe file if absent
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "resources" / "embeddings"
    output_dir = Path(output_dir)
    destination = output_dir / "glove.2024.wikigiga.50d.zip"
    dims = 50
    skip_first = False

    if not destination.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        url = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"
        temporary = destination.with_suffix(".zip.part")  # incomplete downloads are not treated as cached files.
        with urlopen(url) as source, temporary.open("wb") as target:  # Python replaces notebook !wget.
            shutil.copyfileobj(source, target)
        temporary.replace(destination)  # publish the file after the download finishes.

    return str(destination), dims, skip_first


class Vocabulary:
    def __init__(self, init: list, default: str, frozen=False):
        self.words = list(init)
        self._str2int = {w: i for i, w in enumerate(self.words)}
        self._default_word = default
        self._default_index = self._str2int[self._default_word]
        self._frozen = frozen

    @property
    def default_word(self):
        return self._default_word

    @property
    def default_index(self):
        return self._default_index

    def freeze(self):
        #als we trained embeddings willen testen, deze false zetten kunnen de GloVe vectors aanpassen
        self._frozen = True #geen nieuw id aanmaken anders heeft het geen row in GloVe

    def add(self, word: str) -> int: #geen duplicates 
        if self._frozen:
            return self.index(word)
        i = self._str2int.get(word, -1)
        if i < 0:
            i = len(self.words)
            self.words.append(word)
            self._str2int[word] = i
        return i

    def index(self, word: str) -> int:
        return self._str2int.get(word, self.default_index)

    def word(self, index: int) -> str:
        if index >= len(self):
            return self.default_word
        return self.words[index]

    def __contains__(self, word: str):
        return word in self._str2int

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def items(self):
        return enumerate(self.words)

    def __str__(self):
        return "\n".join(f"{i} {w}" for i, w in self.items())


def read_lines_from_file(path, encoding="utf-8"):
    #overgenomen van part 2
    path = str(path)
    if path.endswith(".txt"):
        with open(path, "r", encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith(".zip"):
        with ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise ValueError(
                    f"Expected exactly one .txt file in {path}, "
                    f"found {len(members)}: {members}"
                )
            with archive.open(members[0]) as f:
                with TextIOWrapper(f, encoding=encoding) as text:
                    for line in text:
                        yield line.strip()
    else:
        raise ValueError(f"Unknown format: {path}")


class PretrainedEmbeddings:
    #course embedding table with zero pad and UNK

    def __init__(self, path, num_dims, unk_token="<unk>", pad_token="<pad>", encoding="utf-8", skip_first_line=False):
        
        vectors = [
            np.zeros(num_dims, dtype=np.float32),
            np.zeros(num_dims, dtype=np.float32),
        ]
        self.vocab = Vocabulary(init=[pad_token, unk_token], default=unk_token)
        self.num_dims = num_dims
        self.unk_token = unk_token
        self.pad_token = pad_token

        for line_num, line in enumerate(read_lines_from_file(path, encoding=encoding)):
            if skip_first_line and line_num == 0:
                continue
            parts = line.split()
            if not parts:  #geen blanke lijn tussen tokens
                continue
            word = parts[0].lower()
            values = parts[1:]
            if len(values) != num_dims:
                continue
            vector = np.array(values).astype(np.float32)
            if word in self.vocab:
                continue
            self.vocab.add(word)
            vectors.append(vector)

        if len(vectors) == 2:  #faalt geen empty mean
            raise ValueError(f"No {num_dims}-dimensional word vectors found in {path}")
        vectors = np.stack(vectors, axis=0)
        vectors[self.vocab.default_index] = np.mean(vectors[2:, :], 0)
        self.vectors = vectors
        self.vocab.freeze()
        assert self.vectors.shape[0] == len(self.vocab), "Mismatch between vocab and vectors"

    def __contains__(self, word: str):
        return word.lower() in self.vocab

    def index(self, word: str):
        return self.vocab.index(word.lower())

    def word(self, i: int):
        return self.vocab.word(i)

    def __getitem__(self, i: int):
        return self.vectors[i]

    def embed_word(self, word: str):
        i = self.index(word)
        return self.vectors[i]

    def embed_sequence(self, tokens: list):
        return [self.embed_word(t) for t in tokens]
