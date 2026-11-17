"""Keep tokenized MNLI pairs together and pad them into batches.

LabelledCorpus and collate_labelled_texts follow the course's part2/experiment.py.
We keep their names and interfaces, replacing each single text with a premise and
hypothesis. These changes are marked inline.

Course license: licenses/NLP1-COURSE-LICENSE (Apache-2.0).
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LabelledCorpus(Dataset):
    #tutorials dataset class 

    def __init__(self, X, y):
        #X = (premise_ids, hypothesis_ids) y = labels
        self.X = [
            (torch.tensor(premise, dtype=torch.long), torch.tensor(hypothesis, dtype=torch.long))  # MNLI: two sequences per example.
            for premise, hypothesis in X
        ]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_labelled_texts(batched_pairs, padding_value=0):
    
    # right padded met PAD ID as padding_value padded separately
    
    X_rows, y_rows = zip(*batched_pairs)
    premise_rows, hypothesis_rows = zip(*X_rows)  #premise en hypothese scheiden
    premise_batch = pad_sequence(premise_rows, batch_first=True, padding_value=padding_value)  #pad premises.
    hypothesis_batch = pad_sequence(hypothesis_rows, batch_first=True, padding_value=padding_value)  #pad hypotheses.
    y_batch = torch.stack(y_rows)
    return (premise_batch, hypothesis_batch), y_batch
