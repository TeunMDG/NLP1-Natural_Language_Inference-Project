"""Basic MNLI data preparation."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold


def load_mnli(dataset_id="nyu-mll/multi_nli", revision=None):
    """Load the original MNLI splits.

    Source: teun_pipeline.ipynb.
    """
    return load_dataset(dataset_id, revision=revision)


def _is_valid_example(example):
    """Check for usable texts and a valid MNLI label."""
    return (
        isinstance(example.get("premise"), str)
        and bool(example["premise"].strip())
        and isinstance(example.get("hypothesis"), str)
        and bool(example["hypothesis"].strip())
        and example.get("label") in (0, 1, 2)
    )


def audit_data(dataset):
    """Count labels, genres, repeated pairs and whitespace-separated word lengths.

    Source: teun_data_exploration.ipynb; extra audit checks added for this project.
    """
    pair_labels = defaultdict(Counter)
    for example in dataset:
        if _is_valid_example(example):
            pair = (example["premise"], example["hypothesis"])
            pair_labels[pair][example["label"]] += 1

    lengths = {}
    for field in ("premise", "hypothesis"):
        counts = [len(text.split()) for text in dataset[field] if isinstance(text, str)]
        lengths[field] = {
            "min": min(counts) if counts else None,
            "mean": mean(counts) if counts else None,
            "median": median(counts) if counts else None,
            "max": max(counts) if counts else None,
        }

    return {
        "num_examples": len(dataset),
        "label_counts": dict(Counter(dataset["label"])),
        "genre_counts": dict(Counter(dataset["genre"])),
        "duplicate_rows": sum(
            count - 1 for labels in pair_labels.values() for count in labels.values()
        ),
        "conflicting_pairs": [
            {"premise": premise, "hypothesis": hypothesis, "label_counts": dict(labels)}
            for (premise, hypothesis), labels in pair_labels.items()
            if len(labels) > 1
        ],
        "word_lengths": lengths,
    }


def clean_data(dataset):
    """Keep valid rows and return the cleaned data and removal counts.

    Source: project cleaning rules; select() from teun_pipeline.ipynb.
    """
    kept_indices = [
        index
        for index, (premise, hypothesis, label) in enumerate(zip(
            dataset["premise"][:], dataset["hypothesis"][:], dataset["label"][:]
        ))
        if _is_valid_example({"premise": premise, "hypothesis": hypothesis, "label": label})
    ]
    cleaned_data = dataset.select(kept_indices)

    return cleaned_data, {
        "before": len(dataset),
        "after": len(cleaned_data),
        "removed": len(dataset) - len(cleaned_data),
    }


def _premise_key(premise):
    """Ignore extra whitespace when grouping premises."""
    return " ".join(premise.split())


def select_training_subset(training_data, n_examples=None, seed=0):
    """Select roughly n_examples rows, keeping whole premise groups together.

    Source: Teun's subset selection, adapted to sample groups.
    """
    if n_examples is None:
        return training_data
    if n_examples == len(training_data):
        return training_data

    premise_groups = defaultdict(list)
    for index, premise in enumerate(training_data["premise"][:]):
        premise_groups[_premise_key(premise)].append(index)

    group_keys = list(premise_groups)
    random.Random(seed).shuffle(group_keys)

    selected_indices = []
    for key in group_keys:
        selected_indices.extend(premise_groups[key])
        if len(selected_indices) >= n_examples:
            break

    return training_data.select(selected_indices)


def make_cv_splits(training_data, n_folds=5, seed=0):
    """Create folds containing sentence pairs, labels and row indices.

    Source: part1.experiment fold format; StratifiedGroupKFold for grouping.
    """
    pairs = []
    groups = []
    for example in training_data:
        pairs.append(
            ((example["premise"], example["hypothesis"]), int(example["label"]))
        )
        groups.append(_premise_key(example["premise"]))

    labels = [label for pair, label in pairs]
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    folds = {}

    for fold_id, (training_ids, validation_ids) in enumerate(
        splitter.split(pairs, labels, groups=groups)
    ):
        training_ids = training_ids.tolist()
        validation_ids = validation_ids.tolist()
        folds[fold_id] = {
            "training": [pairs[index] for index in training_ids],
            "validation": [pairs[index] for index in validation_ids],
            "training_ids": training_ids,
            "validation_ids": validation_ids,
        }

    return folds


def save_data_manifest(config, training_data, folds, output_dir):
    """Save settings, selected pair IDs and fold assignments to manifest.json.

    Source: agreed experiment-recording plan; pathlib usage from part1.experiment.
    """
    pair_ids = list(training_data["pairID"])
    manifest = {
        "config": config,
        "num_examples": len(training_data),
        "pair_ids": pair_ids,
        "grouping": "premise with normalized whitespace",
        "folds": {
            str(fold_id): {
                "training_ids": fold["training_ids"],
                "validation_ids": fold["validation_ids"],
                "training_pair_ids": [pair_ids[index] for index in fold["training_ids"]],
                "validation_pair_ids": [pair_ids[index] for index in fold["validation_ids"]],
            }
            for fold_id, fold in folds.items()
        },
    }
    output_path = Path(output_dir) / "manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_data_manifest(input_dir):
    """Read the settings and fold assignments saved by save_data_manifest."""
    path = Path(input_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_saved_folds(input_dir, dataset=None):
    """Restore the subset and folds saved in the manifest.

    Source: our manifest; part1.experiment fold format.
    """
    manifest = load_data_manifest(input_dir)
    if dataset is None:
        config = manifest["config"]
        dataset = load_mnli(config["dataset"], config.get("revision"))["train"]

    source_ids = dataset["pairID"][:]
    positions = {pair_id: index for index, pair_id in enumerate(source_ids)}

    counts = Counter(source_ids)
    if any(counts[pair_id] > 1 for pair_id in manifest["pair_ids"]):
        # MNLI can reuse pairIDs for different hypotheses. Reproduce the saved
        # selection in that case; choosing one row per ID would lose examples.
        config = manifest["config"]
        cleaned, _ = clean_data(dataset)
        subset = select_training_subset(cleaned, config["n_examples"], config["seed"])
    else:
        subset = dataset.select([positions[pair_id] for pair_id in manifest["pair_ids"]])
    pairs = []
    for premise, hypothesis, label in zip(subset["premise"][:], subset["hypothesis"][:], subset["label"][:]):
        pairs.append(((premise, hypothesis), int(label)))

    folds = {}
    for fold_id, saved in manifest["folds"].items():
        train_ids = saved["training_ids"]
        val_ids = saved["validation_ids"]
        folds[int(fold_id)] = {
            "training": [pairs[i] for i in train_ids],
            "validation": [pairs[i] for i in val_ids],
            "training_ids": train_ids,
            "validation_ids": val_ids,
        }
    return subset, folds
