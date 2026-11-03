"""Plan for running our models on the saved MNLI folds."""


def run_experiments(data_dir, output_dir, representations=("indicator",),
                    architectures=("logistic",), seed=0, loglin_args=None, smlp_args=None):
    """Run the selected combinations and return their results.
    Source: part1.experiment.cross_validation_experiment is the starting point.
    """
    # TODO: Load the saved subset and folds with data.load_saved_folds.
    # Use the same examples and splits for every combination.

    # TODO: Loop over the folds and load each fold's tokenizer.
    # Use tokenization.load_fold_tokenizer, then encoders.make_encoder.
    # If an encoder needs fitting, use only the training part of the fold.

    # TODO: Start with indicators + logistic, then add the other combinations.
    # Use models.fit_model and models.predict_probabilities.
    # Keep checkpoint selection away from the outer validation fold.

    # TODO: Add a majority baseline using the training labels of each fold.

    # TODO: Evaluate with the helpers in evaluation.py.
    # Save predictions with pair IDs, probabilities, scores and training time.
    # Give each combination and fold its own output folder.

    # TODO: Collect results in the format expected by part1.analysis.
    # Each system maps to a list of runs, with models, predictions and accuracies
    # keyed by fold. Include macro_f1 too for our evaluation helpers.
    # Save settings and completed folds so we don't lose progress.

    # TODO: Return the collected results for run_experiments.py to summarize.
    raise NotImplementedError("The experiment loop still needs to be implemented.")
