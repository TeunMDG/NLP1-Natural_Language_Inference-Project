"""Run the MNLI data preparation and tokenization pipeline."""

from pathlib import Path
from pprint import pprint

from pipeline.data import (
    load_mnli, audit_data, clean_data, select_training_subset,
    make_cv_splits, save_data_manifest,
)
from pipeline.tokenization import (
    train_fold_tokenizers, inspect_tokenization, save_fold_tokenizers,
)


def main():
    """Prepare data and save fold tokenizers using the existing pipeline functions."""
    config = {
        "dataset": "nyu-mll/multi_nli",
        "seed": 0,
        "n_examples": 50000,
        "n_folds": 5,
        "vocab_size": 1000,
        "min_frequency": 2,
    }
    root = Path(__file__).resolve().parent
    output_dir = root / "results" / "data" / f"mnli-n{config['n_examples']}-seed{config['seed']}"

    print("Loading MNLI...")
    mnli = load_mnli(config["dataset"])
    train, config["cleaning"] = clean_data(mnli["train"])
    subset = select_training_subset(train, config["n_examples"], config["seed"])
    pprint(config["cleaning"])
    pprint(audit_data(subset))

    folds = make_cv_splits(subset, config["n_folds"], config["seed"])
    print("Training fold tokenizers...")
    tokenizers = train_fold_tokenizers(
        folds, config["vocab_size"], config["min_frequency"]
    )
    pprint(inspect_tokenization(tokenizers[0], folds[0]["validation"]))

    save_data_manifest(config, subset, folds, output_dir)
    save_fold_tokenizers(tokenizers, output_dir)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
