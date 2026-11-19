"""Run the majority baseline while Project 2's neural models are being added."""

from pathlib import Path

import torch

from pipeline.embeddings import download_pretrained, PretrainedEmbeddings
from pipeline.evaluation import summarize_results
from pipeline.experiments import run_experiments
from pipeline.tokenization import BasicTokenizer


def main():
    """Use the saved MNLI folds and write to a separate Part 2 output directory."""
    root = Path(__file__).resolve().parent
    config = {
        "data_dir": root / "results" / "data" / "mnli-n50000-seed0",
        "output_dir": root / "results" / "part2" / "mnli-n50000-seed0",
        "systems": ("majority", "embed-and-pool", "lstm"),
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",  #GPU wanneer beschikbaar
        "enc_kwargs": {
            "embed-and-pool": {"freeze_embeddings": True, "average_pool": True},
            "lstm": {"freeze_embeddings": True, "hidden_size": 50, "word_dropout": 0.0},
        },
        "regressor_kwargs": {"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
        "opt_kwargs": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "num_epochs": 20,
        },
    }
    
    path, dims, skip_first = download_pretrained("glove")  #Reuse the downloaded file if present.
    embeddings = PretrainedEmbeddings(path, dims, skip_first_line=skip_first)
    config["emb_model"] = embeddings
    config["tokenizer"] = BasicTokenizer(
        lowercase=True,
        remove_punct=False,
        remove_stopwords=False,
        vocab=embeddings.vocab,
    )
    
    classifiers = run_experiments(**config)
    print(summarize_results(classifiers))
    print("Saved to:", config["output_dir"])


if __name__ == "__main__":
    main()
