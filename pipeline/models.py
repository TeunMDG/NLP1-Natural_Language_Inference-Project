"""Small wrappers around the two supplied training implementations."""

import numpy as np
import torch
from part1 import loglin, smlp
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader


class SparseMLP2(smlp.SparseMLP):
    """Our third architecture: the tutorial SparseMLP with a second hidden layer.
    The first (sparse) layer and forward() are inherited unchanged; only the part
    after the first hidden layer gets an extra Linear + ReLU + Dropout.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1, hidden_dim2=64):
        super().__init__(input_dim, hidden_dim, num_classes, dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
            nn.LogSoftmax(dim=-1),
        )


def train_mlp2(training, validation, encoder, save_to, hidden_dim=128, hidden_dim2=64,
               dropout=0.1, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
               num_epochs=20, device="cpu"):
    """Train SparseMLP2 for a fixed number of epochs and keep the final model.
    Source: part1.smlp.train_neural_network_model (same data handling, optimiser and loss).
    The validation fold is only monitored, never used to pick an epoch.
    """
    X_train = csr_matrix([encoder(x) for x, y in training])
    y_train = [int(y) for x, y in training]
    X_val = csr_matrix([encoder(x) for x, y in validation])
    y_val = [int(y) for x, y in validation]

    train_loader = DataLoader(smlp.SparseDataset(X_train, y_train), batch_size=batch_size,
                              shuffle=True, collate_fn=smlp.sparse_collate)
    val_loader = DataLoader(smlp.SparseDataset(X_val, y_val), batch_size=batch_size,
                            shuffle=False, collate_fn=smlp.sparse_collate)

    model = SparseMLP2(len(encoder), hidden_dim, 3, dropout, hidden_dim2).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    for epoch in range(num_epochs):
        train_metrics = smlp.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = smlp.evaluate(model, val_loader, criterion, device)
        history.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
        print(f"epoch {epoch + 1}: train acc {train_metrics['accuracy']:.4f}, "
              f"val acc {val_metrics['accuracy']:.4f}")

    torch.save(model.state_dict(), f"{save_to}.pt")
    return model, history


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
    if model_option == "mlp2":
        model, history = train_mlp2(training, validation, encoder, save_to, **(smlp_args or {}))
        return model, {"epochs": history, "saved_epoch": len(history)}         
    raise ValueError(f"unknown model option: {model_option}")


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
    if model_option == "mlp2":
        return smlp.predict_probs(pairs, model, encoder)
    raise ValueError(f"unknown model option: {model_option}")