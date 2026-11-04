"""Run model comparisons using the data we already prepared."""

from pathlib import Path
from pipeline.evaluation import summarize_results
from pipeline.experiments import run_experiments


def main():
    """Start with one tutorial combination; add the others in the config."""
    root = Path(__file__).resolve().parent
    config = {
        "data_dir": root / "results" / "data" / "mnli-n50000-seed0",
        "output_dir": root / "results" / "experiments" / "mnli-n50000-seed0",
        "representations": ["indicator"],  # Also available: bow.
        "architectures": ["logistic"],  # Also available: mlp.
        "seed": 0,
        "loglin_args": {"max_iter": 1000},
        "smlp_args": {"hidden_dim": 128, "num_epochs": 20},
    }
    classifiers = run_experiments(**config)
    print(summarize_results(classifiers))
    print("Saved to:", config["output_dir"])


if __name__ == "__main__":
    main()
