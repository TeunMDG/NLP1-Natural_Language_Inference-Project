"""Prepare the MNLI subset and grouped folds for Project 2."""

from pathlib import Path
from pprint import pprint

from pipeline.data import (
    load_mnli, audit_data, clean_data, select_training_subset,
    make_cv_splits, save_data_manifest,
)


def main():
    """Audit MNLI and save a manifest without training a tokenizer."""
    config = {
        "dataset": "nyu-mll/multi_nli",
        "seed": 0,
        "n_examples": 50000,
        "n_folds": 5,
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
    save_data_manifest(config, subset, folds, output_dir)
    print("Saved data manifest to:", output_dir)

if __name__ == "__main__":
    main()
