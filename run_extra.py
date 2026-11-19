"""P2 extra round on fold 0 (D-020): learning curve (Q4), pooling type and learning rate (Q3/Q5)."""

from pathlib import Path

import torch

from pipeline.embeddings import download_pretrained, PretrainedEmbeddings
from pipeline.evaluation import summarize_results
from pipeline.experiments import run_experiments
from pipeline.tokenization import BasicTokenizer

POOL = {"freeze_embeddings": True, "pair_features": "infersent"}
LSTM = {"freeze_embeddings": True, "hidden_size": 50, "word_dropout": 0.0, "pair_features": "infersent"}
OPT = {"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20}

# (name, system, encoder settings, optimiser overrides, max training examples). 40k / lr 1e-3 / mean = final run, fold 0.
RUNS = [
    ("lstm-inf-n5000",   "lstm",           LSTM,                              {},                       5000),
    ("lstm-inf-n10000",  "lstm",           LSTM,                              {},                       10000),
    ("lstm-inf-n20000",  "lstm",           LSTM,                              {},                       20000),
    ("pool-inf-sum",     "embed-and-pool", {**POOL, "average_pool": False},   {},                       None),
    ("lstm-inf-lr3e-4",  "lstm",           LSTM,                              {"learning_rate": 3e-4},  None),
    ("lstm-inf-lr3e-3",  "lstm",           LSTM,                              {"learning_rate": 3e-3},  None),
]


def main():
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path, dims, skip_first = download_pretrained("glove")
    embeddings = PretrainedEmbeddings(path, dims, skip_first_line=skip_first)
    tokenizer = BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False, vocab=embeddings.vocab)

    for name, system, enc, opt, n_train in RUNS:
        print(f"\n===== {name} =====")
        classifiers = run_experiments(
            data_dir=root / "results" / "data" / "mnli-n50000-seed0",
            output_dir=root / "results" / "part2" / "extra-fold0" / name,
            tokenizer=tokenizer,
            emb_model=embeddings,
            systems=(system,),
            enc_kwargs={system: enc},
            regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
            opt_kwargs={**OPT, **opt},
            seed=0,
            device=device,
            fold_ids=[0],
            max_train_examples=n_train,
        )
        print(summarize_results(classifiers))


if __name__ == "__main__":
    main()