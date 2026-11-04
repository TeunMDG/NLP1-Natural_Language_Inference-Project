"""Evaluate and compare our models using the tutorial helpers."""
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pathlib import Path
import pandas as pd
from part1.analysis import make_data_frame, BenavoliCorrelatedT, plot_box_and_points

def evaluate_predictions(gold, predictions):
    """Calculate the scores for one validation fold.
    Source: tutorial evaluation; sklearn.metrics.
    """
    acc_score = accuracy_score(gold,predictions)
    f1 = f1_score(gold, predictions, average="macro", labels=[0, 1, 2], zero_division=0)
    cf_m = confusion_matrix(gold, predictions, labels=[0, 1, 2])
    return {"accuracy":acc_score,
            "macro_f1":f1,
            "confusion_matrix":cf_m}



def save_predictions(path, examples, predictions, probabilities):
    """Save predictions and class probabilities with the matching example IDs."""
    table = pd.DataFrame({
        "pairID": list(examples["pairID"]),
        "genre": list(examples["genre"]),
        "gold_label": list(examples["label"]),
        "prediction": predictions,
    })
    labels = ["entailment", "neutral", "contradiction"]
    for index, label in enumerate(labels):
        table[f"p_{label}"] = [row[index] for row in probabilities]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def make_results_table(classifiers):
    """Put the scores from all systems and folds into one table.
    Source: part1.analysis.make_data_frame.
    """
    table = make_data_frame(classifiers)
    macro_f1_scores = []
    for row in table.itertuples(index=False):
        score = classifiers[row.model][row.run]["macro_f1"][row.fold]
        macro_f1_scores.append(score)
    table["macro_f1"] = macro_f1_scores
    return table


def summarize_results(classifiers):
    """Show the average scores and how much they vary across folds."""
    table = make_results_table(classifiers)
    return table.groupby("model")[["accuracy", "macro_f1"]].agg(["mean", "std"])


def compare_models(classifiers, name_a, name_b, rope=0.005):
    """Compare two systems using the tutorial's Bayesian approach.
    Source: part1.analysis.BenavoliCorrelatedT.
    """
    scores_a = []
    scores_b = []
    runs_a = classifiers[name_a]
    runs_b = classifiers[name_b]

    for run_index, run_a in enumerate(runs_a):
        run_b = runs_b[run_index]
        for fold in sorted(run_a["accuracies"]):
            scores_a.append(run_a["accuracies"][fold])
            scores_b.append(run_b["accuracies"][fold])

    posterior = BenavoliCorrelatedT(
        scores_a, scores_b, runs=len(runs_a), names=[name_a, name_b]
    )
    p_a, p_tie, p_b = posterior.probs(rope)
    return {
        "A": name_a,
        "B": name_b,
        "rope": rope,
        "mean_B_minus_A": float(posterior.mean_diff),
        "p_A_better": float(p_a),
        "p_equivalent": float(p_tie),
        "p_B_better": float(p_b),
    }


def plot_results(classifiers):
    """Plot the individual fold scores and their spread.
    Source: part1.analysis.plot_box_and_points.
    """
    table = make_results_table(classifiers)
    plot_box_and_points(table)


# TODO: Inspect actual errors, e.g. negation or high word overlap.
# Evaluate matched/mismatched sets after choosing models, keeping them out of tuning.
