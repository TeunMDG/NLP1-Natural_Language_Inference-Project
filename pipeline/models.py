"""Small wrappers around the two supplied training implementations."""

import numpy as np
import torch
from part1 import loglin, smlp


# TODO: Define our third model here, starting from smlp.SparseMLP.
# Add a second hidden layer and keep the same three MNLI outputs.


def fit_model(model_option, training, validation, encoder, save_to, seed=0,
              loglin_args=None, smlp_args=None):
    """Train and save a tutorial model; return the model and its training history.
    Source: part1.experiment.cross_validation_experiment.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_option == "logistic":
        model = loglin.fit(training, encoder, save_to=str(save_to), **(loglin_args or {}))
        return model, {}
    if model_option == "mlp":
        args = dict(smlp_args or {})
        args["select_best"] = False  # Score a fixed final epoch on the outer fold.
        history, final_epoch, _ = smlp.train_neural_network_model(
            training, validation, 3, encoder, save_to=str(save_to), **args
        )
        model = smlp.load(str(save_to))
        return model, {"epochs": history, "saved_epoch": final_epoch + 1}
    # TODO: Add an elif model_option == "mlp2" branch here.
    # Train and save the third model, then return (model, history).
    # Reuse smlp.train_one_epoch and smlp.evaluate where possible.


def predict_probabilities(model_option, pairs, model, encoder):
    """Predict probabilities in MNLI label order: entailment, neutral, contradiction.
    Source: part1.loglin.predict_probs and part1.smlp.predict_probs.
    """
    if model_option == "logistic":
        probabilities = np.zeros((len(pairs), 3))
        probabilities[:, model.classes_.astype(int)] = loglin.predict_probs(pairs, model, encoder)
        return probabilities
    if model_option == "mlp":
        return smlp.predict_probs(pairs, model, encoder)
    # TODO: Add an elif model_option == "mlp2" branch here.
    # Return its three class probabilities, in the same order as the other models.
