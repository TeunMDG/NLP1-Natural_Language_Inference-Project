"""Run the tutorial models on our saved MNLI folds."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from .data import load_saved_folds
from .tokenization import load_fold_tokenizer
from .encoders import make_encoder
from .models import fit_model, predict_probabilities
from .evaluation import evaluate_predictions, save_predictions, make_results_table, summarize_results


def run_experiments(data_dir, output_dir, representations=("indicator",),
                    architectures=("logistic",), seed=0, loglin_args=None, smlp_args=None):
    """Run selected combinations on the saved folds and return their results.
    Source: part1.experiment.cross_validation_experiment, adapted for MNLI pairs.
    """
    # These are the exact examples and splits from our preparation step.
    # Reusing them gives every model the same training/validation comparison.
    subset, folds = load_saved_folds(data_dir)

    # Keep the settings next to the results so we know what produced them.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": str(Path(data_dir).resolve()),
        "representations": list(representations),
        "architectures": list(architectures),
        "seed": seed,
        "loglin_args": loglin_args or {},
        "smlp_args": smlp_args or {},
        "mlp_checkpoint": "final epoch",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # A system is one architecture + one representation, e.g. logistic + bow.
    # The majority baseline doesn't need an encoder, so its representation is None.
    systems = [("majority", None)]
    for representation in representations:
        for architecture in architectures:
            systems.append((architecture, representation))

    # Follow the tutorial's structure: each system has a list of CV runs.
    # We do one run here, and store scores/model paths by fold inside that run.
    classifiers = {}

    # Each fold contains training/validation pairs and their positions in subset.
    for fold_id, fold in folds.items():
        tokenizer = load_fold_tokenizer(data_dir, fold_id)

        # All systems in this fold share the same tokenizer and vocabulary.
        encoders = {}
        for representation in representations:
            encoders[representation] = make_encoder(tokenizer, representation)
        # TODO: When adding TF-IDF, fit its weights on fold["training"] only.

        # Separate the validation inputs from the answers we score against.
        # A pair is (premise, hypothesis); gold contains the integer labels.
        pairs = [pair for pair, label in fold["validation"]]
        gold = [label for pair, label in fold["validation"]]

        # Get IDs and genres in exactly the same order as those validation pairs.
        validation_rows = subset.select(fold["validation_ids"])

        for model_option, representation in systems:
            name = "majority" if model_option == "majority" else f"{model_option}_{representation}"
            if name not in classifiers:
                classifiers[name] = [{
                    "models": {}, "predictions": {}, "accuracies": {}, "macro_f1": {},
                }]

            # Give each combination/fold its own model files and predictions.
            folder = output_dir / name / f"fold-{fold_id}"
            folder.mkdir(parents=True, exist_ok=True)
            model_path = None
            history = {}
            start = perf_counter()

            if model_option == "majority":
                # Learn class frequencies from training labels, then use the
                # same probabilities for every validation example.
                labels = [label for pair, label in fold["training"]]
                prior = np.bincount(labels, minlength=3) / len(labels)
                probabilities = np.tile(prior, (len(gold), 1))
            else:
                model_path = str(folder / "model")
                # These wrappers call the supplied part1 training functions.
                # The MLP uses fixed epochs, without selecting on this outer fold.
                model, history = fit_model(
                    model_option, fold["training"], fold["validation"],
                    encoders[representation], model_path, seed=seed + fold_id,
                    loglin_args=loglin_args, smlp_args=smlp_args,
                )
                probabilities = predict_probabilities(
                    model_option, pairs, model, encoders[representation]
                )
                # The model is already saved, so release it before the next fit.
                del model

            # Each row has three probabilities in label order 0, 1, 2.
            # argmax(axis=1) picks the most likely class for each example.
            predictions = probabilities.argmax(axis=1)
            metrics = evaluate_predictions(gold, predictions)
            metrics["seconds"] = perf_counter() - start
            metrics["n_training"] = len(fold["training"])
            metrics["n_validation"] = len(gold)

            # NumPy matrices need conversion to ordinary lists before JSON saving.
            metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
            metrics["accuracy"] = float(metrics["accuracy"])
            metrics["macro_f1"] = float(metrics["macro_f1"])
            save_predictions(folder / "predictions.csv", validation_rows, predictions, probabilities)
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            if history:
                (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            # [0] selects our single CV run. The fold ID keeps scores aligned
            # across systems for the later paired Bayesian comparison.
            run = classifiers[name][0]
            run["models"][fold_id] = model_path
            run["predictions"][fold_id] = predictions.tolist()
            run["accuracies"][fold_id] = metrics["accuracy"]
            run["macro_f1"][fold_id] = metrics["macro_f1"]
            print(f"Fold {fold_id}: {name}, accuracy={metrics['accuracy']:.4f}", flush=True)

        # Save after each completed fold so earlier results survive a later failure.
        (output_dir / "results.json").write_text(json.dumps(classifiers, indent=2), encoding="utf-8")
        make_results_table(classifiers).to_csv(output_dir / "fold_scores.csv", index=False)
        summarize_results(classifiers).to_csv(output_dir / "summary.csv")

    return classifiers


# TODO: Add the third representation and architecture once their modules support them.
