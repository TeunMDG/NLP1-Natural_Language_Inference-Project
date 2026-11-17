"""Offline checks for the preparation and evaluation retained during cleanup."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import numpy as np
import pandas as pd

from pipeline.data import load_saved_folds, make_cv_splits, save_data_manifest
from pipeline.embeddings import PretrainedEmbeddings
from pipeline.experiments import run_experiments
from pipeline.tokenization import BasicTokenizer
import run_data


class RetainedPipelineTests(unittest.TestCase):
    def setUp(self):
        # Each premise occurs twice; folds must keep those rows together.
        self.dataset = Dataset.from_list([
            {
                "pairID": f"pair-{group}-{variant}",
                "premise": f"Shared premise {group}.",
                "hypothesis": f"Hypothesis {variant} for {group}.",
                "label": group % 3,
                "genre": "fiction",
            }
            for group in range(15)
            for variant in range(2)
        ])

    def test_preparation_saves_manifest_without_bpe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(run_data, "__file__", str(root / "run_data.py")),
                patch.object(run_data, "load_mnli", return_value={"train": self.dataset}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_data.main()

            data_dir = root / "results" / "data" / "mnli-n50000-seed0"
            manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_examples"], len(self.dataset))
            self.assertEqual(len(manifest["folds"]), 5)
            self.assertNotIn("vocab_size", manifest["config"])
            self.assertNotIn("min_frequency", manifest["config"])
            self.assertEqual(list(data_dir.rglob("tokenizer.json")), [])

            subset, folds = load_saved_folds(data_dir, dataset=self.dataset)
            self.assertEqual(subset["pairID"][:], manifest["pair_ids"])
            for fold in folds.values():
                train_premises = {pair[0] for pair, label in fold["training"]}
                validation_premises = {pair[0] for pair, label in fold["validation"]}
                self.assertTrue(train_premises.isdisjoint(validation_premises))

    def test_baseline_uses_training_labels_and_preserves_prediction_order(self):
        folds = make_cv_splits(self.dataset, n_folds=5, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, output_dir = root / "data", root / "output"
            save_data_manifest(
                {"dataset": "offline-fixture", "n_examples": None, "seed": 0},
                self.dataset, folds, data_dir,
            )
            with (
                patch("pipeline.data.load_mnli", return_value={"train": self.dataset}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results = run_experiments(data_dir, output_dir)

            self.assertEqual(set(results), {"majority"})
            for fold_id, fold in folds.items():
                training_labels = [label for pair, label in fold["training"]]
                prior = np.bincount(training_labels, minlength=3) / len(training_labels)
                folder = output_dir / "majority" / f"fold-{fold_id}"
                predictions = pd.read_csv(folder / "predictions.csv")
                expected_ids = self.dataset.select(fold["validation_ids"])["pairID"][:]
                self.assertEqual(predictions["pairID"].tolist(), expected_ids)
                self.assertTrue((predictions["prediction"] == prior.argmax()).all())
                probability_columns = ["p_entailment", "p_neutral", "p_contradiction"]
                np.testing.assert_allclose(
                    predictions[probability_columns].to_numpy(),
                    np.tile(prior, (len(predictions), 1)),
                )
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                expected_accuracy = (predictions["gold_label"] == prior.argmax()).mean()
                self.assertAlmostEqual(metrics["accuracy"], expected_accuracy)
                self.assertEqual(metrics["n_training"], len(training_labels))

            saved = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["majority"][0]["accuracies"]), 5)
            self.assertEqual(len(pd.read_csv(output_dir / "fold_scores.csv")), 5)
            self.assertTrue((output_dir / "summary.csv").is_file())

    def test_neural_systems_share_folds_and_export_probabilities(self):
        folds = make_cv_splits(self.dataset, n_folds=2, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, output_dir = root / "data", root / "output"
            save_data_manifest(
                {"dataset": "offline-fixture", "n_examples": None, "seed": 0},
                self.dataset, folds, data_dir,
            )
            embedding_file = root / "tiny.txt"
            embedding_file.write_text(
                "shared 1 0 0\npremise 0 1 0\nhypothesis 0 0 1\nfor 1 1 0\n", encoding="utf-8",
            )
            embeddings = PretrainedEmbeddings(embedding_file, 3)
            tokenizer = BasicTokenizer(vocab=embeddings.vocab)
            systems = ("majority", "embed-and-pool", "lstm")
            with (
                patch("pipeline.data.load_mnli", return_value={"train": self.dataset}),
                patch.object(tokenizer, "tokens", side_effect=lambda text: text.lower().split()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results = run_experiments(
                    data_dir, output_dir, tokenizer=tokenizer, emb_model=embeddings,
                    systems=systems, enc_kwargs={"lstm": {"hidden_size": 4}},
                    regressor_kwargs={"hidden_size1": 4, "hidden_size2": 4},
                    opt_kwargs={"num_epochs": 1, "batch_size": 32}, seed=7,
                )

            self.assertEqual(set(results), set(systems))
            for name in systems:
                self.assertEqual(set(results[name][0]["accuracies"]), set(folds))
                for fold_id, fold in folds.items():
                    folder = output_dir / name / f"fold-{fold_id}"
                    table = pd.read_csv(folder / "predictions.csv")
                    expected_rows = self.dataset.select(fold["validation_ids"])
                    self.assertEqual(table["pairID"].tolist(), expected_rows["pairID"][:])
                    self.assertEqual(table["gold_label"].tolist(), expected_rows["label"][:])
                    probabilities = table[["p_entailment", "p_neutral", "p_contradiction"]].to_numpy()
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    np.testing.assert_array_equal(table["prediction"].to_numpy(), probabilities.argmax(axis=1))
                    if name != "majority":
                        config = json.loads((folder / "model.json").read_text(encoding="utf-8"))
                        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                        self.assertEqual(config["seed"], 7 + fold_id)
                        self.assertEqual(metrics["seed"], config["seed"])
                        self.assertEqual(metrics["saved_epoch"], 0)
                        self.assertTrue((folder / "model.pt").is_file())
                        self.assertTrue((folder / "model.history.json").is_file())
            self.assertEqual(len(pd.read_csv(output_dir / "fold_scores.csv")), 6)


if __name__ == "__main__":
    unittest.main()
