"""P2 final runs (D-019): finalists on all 5 folds; extra seeds for the main finalists."""

from pathlib import Path

import torch

from pipeline.embeddings import download_pretrained, PretrainedEmbeddings
from pipeline.evaluation import summarize_results
from pipeline.experiments import run_experiments
from pipeline.tokenization import BasicTokenizer

POOL = {"freeze_embeddings": True, "average_pool": True}
LSTM = {"freeze_embeddings": True, "hidden_size": 50, "word_dropout": 0.0}

# (name, system, encoder settings, seed). Order = priority: the 2x2 on seed 0 first, extra seeds last.
RUNS = [
    ("majority",                  "majority",       {},                                       0),
    ("pool-frozen-concat",        "embed-and-pool", {**POOL, "pair_features": "concat"},      0),
    ("pool-frozen-infersent",     "embed-and-pool", {**POOL, "pair_features": "infersent"},   0),
    ("lstm-frozen-h50-concat",    "lstm",           {**LSTM, "pair_features": "concat"},      0),
    ("lstm-frozen-h50-infersent", "lstm",           {**LSTM, "pair_features": "infersent"},   0),
    ("pool-frozen-infersent",     "embed-and-pool", {**POOL, "pair_features": "infersent"},   100),
    ("lstm-frozen-h50-infersent", "lstm",           {**LSTM, "pair_features": "infersent"},   100),
    ("pool-frozen-infersent",     "embed-and-pool", {**POOL, "pair_features": "infersent"},   200),
    ("lstm-frozen-h50-infersent", "lstm",           {**LSTM, "pair_features": "infersent"},   200),
]


def main():
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path, dims, skip_first = download_pretrained("glove")
    embeddings = PretrainedEmbeddings(path, dims, skip_first_line=skip_first)
    tokenizer = BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False, vocab=embeddings.vocab)

    for name, system, enc, seed in RUNS:
        print(f"\n===== {name} | seed {seed} =====")
        classifiers = run_experiments(
            data_dir=root / "results" / "data" / "mnli-n50000-seed0",
            output_dir=root / "results" / "part2" / "final" / name / f"seed-{seed}",  # own folder per config + seed
            tokenizer=tokenizer,
            emb_model=embeddings,
            systems=(system,),
            enc_kwargs={system: enc} if enc else None,
            regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
            opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20},
            seed=seed,
            device=device,
            # fold_ids / max_examples left out: all 5 folds, full data
        )
        print(summarize_results(classifiers))


if __name__ == "__main__":
    main()