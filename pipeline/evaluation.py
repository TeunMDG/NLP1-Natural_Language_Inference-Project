"""Plan for evaluating and comparing our models."""


def evaluate_predictions(gold, predictions):
    """Calculate the scores for one validation fold.
    Source: tutorial evaluation; sklearn.metrics.
    """
    # TODO: Return accuracy, macro-F1 and a confusion matrix.
    # Keep the label order fixed: 0 entailment, 1 neutral, 2 contradiction.
    raise NotImplementedError("Add the fold metrics here.")


def save_predictions(path, examples, predictions, probabilities):
    """Save enough information to look at mistakes later."""
    # TODO: Make a CSV with pairID, genre, gold label and predicted label.
    # Add the three class probabilities and keep the example order aligned.
    raise NotImplementedError("Add prediction saving here.")


def make_results_table(classifiers):
    """Put the scores from all systems and folds into one table.
    Source: part1.analysis.make_data_frame.
    """
    # TODO: Start with the existing make_data_frame helper and add macro-F1.
    # Keep the system name, run and fold so we can match results correctly.
    raise NotImplementedError("Add the results table here.")


def summarize_results(classifiers):
    """Show the average scores and how much they vary across folds."""
    # TODO: Use make_results_table and group scores by system.
    # Report the mean and standard deviation for accuracy and macro-F1.
    raise NotImplementedError("Add the score summary here.")


def compare_models(classifiers, name_a, name_b, rope=0.005):
    """Compare two systems using the tutorial's Bayesian approach.
    Source: part1.analysis.BenavoliCorrelatedT.
    """
    # TODO: Collect accuracy scores in matching run/fold order for both systems.
    # Pass the number of CV repetitions as runs, not the number of folds.
    # Start with ROPE +/-0.005 (half a percentage point) and justify that choice.
    # Report the probabilities of A being better, practical equivalence and B being better.
    raise NotImplementedError("Add the paired Bayesian comparison here.")


def plot_results(classifiers):
    """Plot the individual fold scores and their spread.
    Source: part1.analysis.plot_box_and_points.
    """
    # TODO: Use the results table with the existing plotting helper.
    raise NotImplementedError("Connect the results table to the plotting helper.")


# TODO: Inspect actual errors, e.g. negation or high word overlap.
# Evaluate matched/mismatched sets after choosing models, keeping them out of tuning.
