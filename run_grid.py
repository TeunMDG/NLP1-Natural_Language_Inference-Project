"""P2 Q5: small hyperparameter grid, tuned on fold 0 only (decisions-p2)."""

from pathlib import Path

import torch

from pipeline.embeddings import download_pretrained, PretrainedEmbeddings
from pipeline.evaluation import summarize_results
from pipeline.experiments import run_experiments
from pipeline.tokenization import BasicTokenizer

# One entry per run: (name, system, encoder settings). Everything else is shared, so each
# difference between runs has exactly one cause.
# Round 1 (done 18 Nov, results/part2/grid-fold0.log)
GRID_ROUND1 = [
    ("pool-frozen",          "embed-and-pool", {"freeze_embeddings": True,  "average_pool": True}),
    ("lstm-frozen-h50",      "lstm",           {"freeze_embeddings": True,  "hidden_size": 50,  "word_dropout": 0.0}),
    ("lstm-frozen-h100",     "lstm",           {"freeze_embeddings": True,  "hidden_size": 100, "word_dropout": 0.0}),
    ("pool-trainable",       "embed-and-pool", {"freeze_embeddings": False, "average_pool": True}),
    ("lstm-trainable-h50",   "lstm",           {"freeze_embeddings": False, "hidden_size": 50,  "word_dropout": 0.0}),
    ("lstm-trainable-h100",  "lstm",           {"freeze_embeddings": False, "hidden_size": 100, "word_dropout": 0.0}),
]

# Round 2: same frozen configs + InferSent pair features (D-015), directly comparable with round 1
GRID_ROUND2 = [
    ("pool-frozen-infersent",     "embed-and-pool", {"freeze_embeddings": True, "average_pool": True, "pair_features": "infersent"}),
    ("lstm-frozen-h50-infersent", "lstm",           {"freeze_embeddings": True, "hidden_size": 50, "word_dropout": 0.0, "pair_features": "infersent"}),
]

def main():
    root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GloVe + tokenizer once for all runs (slow step).
    path, dims, skip_first = download_pretrained("glove")
    embeddings = PretrainedEmbeddings(path, dims, skip_first_line=skip_first)
    tokenizer = BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False, vocab=embeddings.vocab)

    for name, system, enc in GRID_ROUND2:
        print(f"\n===== {name} =====")
        classifiers = run_experiments(
            data_dir=root / "results" / "data" / "mnli-n50000-seed0",
            output_dir=root / "results" / "part2" / "grid-fold0" / name,  # own folder per config: no overwriting
            tokenizer=tokenizer,
            emb_model=embeddings,
            systems=(system,),
            enc_kwargs={system: enc},
            regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
            opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20},
            seed=0,
            device=device,
            fold_ids=[0],  # tune on fold 0 only; finalists go to all 5 folds later
        )
        print(summarize_results(classifiers))


if __name__ == "__main__":
    main()