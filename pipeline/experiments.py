"""Compare majority, pooling, and LSTM on the same saved MNLI folds.

The fold loop, per-fold model folder, train call, and result collection are adapted
from nlp1-student/part2/experiment.py:cross_validation. 

We keep our existing MNLI
fold loading, majority baseline, metrics, and exports around that course structure.
Reused steps are marked 'Course'; our adaptations are commented

Course license: licenses/NLP1-COURSE-LICENSE (Apache-2.0).
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import models
from .data import load_saved_folds
from .evaluation import (
    evaluate_predictions,
    make_results_table,
    save_predictions,
    summarize_results,
)


def run_experiments(
    data_dir,
    output_dir,
    tokenizer=None,
    emb_model=None,
    systems=("majority",),
    enc_kwargs=None,
    regressor_kwargs=None,
    opt_kwargs=None,
    seed=0,
    device="cpu",
    fold_ids=None,
    max_examples=None,
):
    
    #systems can contain 'majority', 'embed-and-pool', and 'lstm'. Neural systems
    systems = tuple(systems)
    

    encoder_settings = {  #keep separate constructor settings for the two encoders.
        "embed-and-pool": {"freeze_embeddings": True},
        "lstm": {"freeze_embeddings": True, "hidden_size": 50},
    }

    for name, overrides in (enc_kwargs or {}).items():
        if name not in encoder_settings:
            raise ValueError(f"Unknown encoder settings: {name}")
        
        encoder_settings[name].update(overrides)


    regressor_kwargs = dict(regressor_kwargs or {})
    opt_kwargs = dict(opt_kwargs or {})

    subset, folds = load_saved_folds(data_dir)  #premise-grouped folds.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": str(Path(data_dir).resolve()),
        "systems": list(systems),
        "enc_kwargs": {name: encoder_settings[name] for name in systems if name != "majority"},
        "regressor_kwargs": regressor_kwargs,
        "opt_kwargs": opt_kwargs,
        "seed": seed,
        "fold_seed_rule": "seed + fold_id",
        "device": str(device),
        "fold_ids": fold_ids,
        "max_examples": max_examples,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    classifiers = {  # retain format
        name: [{"models": {}, "predictions": {}, "accuracies": {}, "macro_f1": {}}]
        for name in systems
    }
    for fold_id in folds:
        if fold_ids is not None and fold_id not in fold_ids:
            continue
        training_set = folds[fold_id]["training"]
        validation_set = folds[fold_id]["validation"]
        validation_ids = folds[fold_id]["validation_ids"]
        if max_examples is not None:
            training_set = training_set[:max_examples]
            validation_set = validation_set[:max_examples]
            validation_ids = validation_ids[:max_examples]
        pairs = [pair for pair, label in validation_set]
        gold = [label for pair, label in validation_set]
        training_labels = [label for pair, label in training_set]
        validation_rows = subset.select(validation_ids)

        for name in systems:  #vergelijk systemen binnen zelfde fold
            folder = output_dir / name / f"fold-{fold_id}"
            folder.mkdir(parents=True, exist_ok=True)
            model_path = None
            model_seed = None
            saved_epoch = None
            start = perf_counter()

            if name == "majority":  #training-label majority baseline.
                prior = np.bincount(training_labels, minlength=3) / len(training_labels)
                probabilities = np.tile(prior, (len(gold), 1))
            else:
                model_path = str(folder / "model")
                model_seed = seed + int(fold_id)
                _, saved_epoch = models.train(
                    training_set,
                    validation_set,
                    tokenizer,
                    emb_model,
                    num_classes=3,  #MNLI: entailment, neutral, contradiction
                    enc_type=name,
                    enc_kwargs=encoder_settings[name],
                    regressor_kwargs=regressor_kwargs,
                    opt_kwargs=opt_kwargs,
                    device=device,
                    save_to=model_path,
                    seed=model_seed,
                )
                model = models.load(model_path, emb_model, device=device)  # score the saved final-epoch checkpoint
                probabilities = models.predict_probabilities(  # probabilities voor CSV export 
                    model, tokenizer, pairs, batch_size=opt_kwargs.get("batch_size", 64), device=device,
                )
                del model

            predictions = probabilities.argmax(axis=1)
            metrics = evaluate_predictions(gold, predictions)  #we houden accuracy, macro-F1, en confusion matrix
            metrics.update({
                "seconds": perf_counter() - start,
                "n_training": len(training_labels),
                "n_validation": len(gold),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
                "seed": model_seed,
                "saved_epoch": saved_epoch,
            })
            save_predictions(folder / "predictions.csv", validation_rows, predictions, probabilities)  #hou IDs and genres
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            run = classifiers[name][0]
            run["models"][fold_id] = model_path
            run["predictions"][fold_id] = predictions.tolist()
            run["accuracies"][fold_id] = metrics["accuracy"]
            run["macro_f1"][fold_id] = metrics["macro_f1"]
            print(f"Fold {fold_id}: {name}, accuracy={metrics['accuracy']:.4f}", flush=True)

        #sla alle info op
        (output_dir / "results.json").write_text(json.dumps(classifiers, indent=2), encoding="utf-8")
        make_results_table(classifiers).to_csv(output_dir / "fold_scores.csv", index=False)
        summarize_results(classifiers).to_csv(output_dir / "summary.csv")

    return classifiers
