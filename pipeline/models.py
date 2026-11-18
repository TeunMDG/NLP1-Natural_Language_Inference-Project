"""Train the tutorial's classifier on MNLI premise/hypothesis pairs.

SoftmaxRegressor comes from part2/clslib.py. Training, evaluation, loading, and
prediction follow part2/experiment.py, using our paired dataset and encoders.
MNLI adaptations and changes to checkpoint handling are marked inline.
Course license: licenses/NLP1-COURSE-LICENSE (Apache-2.0).
"""

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .embeddings import PretrainedEmbeddings
from .encoders import EmbedAndPoolEncoder, LSTMEncoder, PairEncoder
from .sequences import LabelledCorpus, collate_labelled_texts
from .tokenization import Tokenizer, tokenize_pairs


class SoftmaxRegressor(nn.Module):
    #The course MLP: encoded inputs to three MNLI label log-probabilities.

    def __init__(self, hidden_size1, hidden_size2, num_classes, encoder: nn.Module, dropout=0.):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(encoder.enc_dim, hidden_size1),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size2, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        h = self.encoder(x)  # combined nu
        logprobs = self.output_layer(h)
        return logprobs


def training_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    #run de tutorial training epoch met de paired inputs
    if isinstance(device, str):
        device = torch.device(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = tuple(sentence.to(device) for sentence in X_batch)  #beide zin tensors
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logprobs = model(X_batch)
        loss = criterion(logprobs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def evaluate(model, dataloader, criterion, device="cpu"):
    # Course evaluate aangepast
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = tuple(sentence.to(device) for sentence in X_batch)  # <--- both sentence tensors.
            y_batch = y_batch.to(device)
            logprobs = model(X_batch)
            loss = criterion(logprobs, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_examples += y_batch.size(0)
            preds_list.append(predictions.tolist())
            labels_list.append(y_batch.tolist())

    return {
        "predictions": preds_list,
        "labels": labels_list,
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train(
    training_set,
    validation_set,
    tokenizer: Tokenizer,
    emb_model: PretrainedEmbeddings,
    num_classes: int,
    enc_type: str,
    save_to="temp",
    enc_kwargs=None,
    regressor_kwargs=None,
    opt_kwargs=None,
    device="cpu",
    seed=0,  # <--- seed.
):
    
    enc_kwargs = dict(enc_kwargs or {"freeze_embeddings": True}) #flip als we trained embeddings willen
    regressor_kwargs = {"hidden_size1": 50, "hidden_size2": 50, "dropout": 0., **(regressor_kwargs or {})}
    opt_kwargs = {
        "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20,
        **(opt_kwargs or {}),
    }
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    time_start = time.time()

    print("Tokenizing sentence pairs...")
    training_pairs = tokenize_pairs(training_set, tokenizer)  # tokenize both sentences with their labels
    validation_pairs = tokenize_pairs(validation_set, tokenizer)
    training_vectors = [pair for pair, label in training_pairs]
    training_labels = [label for pair, label in training_pairs]
    validation_vectors = [pair for pair, label in validation_pairs]
    validation_labels = [label for pair, label in validation_pairs]
    if not training_vectors or not validation_vectors:
        raise ValueError("Training and validation sets must both contain examples")
    time_tokenizing = time.time()
    print(f"Tokenization done! Time taken: {time_tokenizing - time_start:.3f} seconds")

    pad_id = emb_model.vocab.index(emb_model.pad_token)
    train_dataset = LabelledCorpus(training_vectors, training_labels)
    val_dataset = LabelledCorpus(validation_vectors, validation_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt_kwargs["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt_kwargs["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )
    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_tokenizing:.3f} seconds")
    if isinstance(device, str):
        device = torch.device(device)

    enc_kwargs["padding_idx"] = pad_id
    regressor_kwargs["num_classes"] = num_classes
    args = {
        "enc_kwargs": enc_kwargs,
        "regressor_kwargs": regressor_kwargs,
        "enc_type": enc_type,
        "opt_kwargs": opt_kwargs,
        "seed": seed,
        # added the next ones for analysis
        "checkpoint": "final epoch",
        "saved_epoch": opt_kwargs["num_epochs"] - 1,
        "vocabulary": emb_model.vocab.words,
        "tokenizer_kwargs": {
            "lowercase": tokenizer.lowercase,
            "remove_punct": tokenizer.remove_punct,
            "remove_stopwords": tokenizer.remove_stopwords,
            "language": tokenizer.language,
        },
    }
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)  #new experiment output folder.
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)

    encoder_args = dict(enc_kwargs)
    pair_features = encoder_args.pop("pair_features", "concat")  # P2: pair-level option, not a sentence-encoder argument

    if enc_type == "embed-and-pool":
        encoder = EmbedAndPoolEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **encoder_args,
        )
    elif enc_type == "lstm":
        encoder = LSTMEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **encoder_args,
        )
    else:
        raise ValueError(f"I do not know this encoder type: {enc_type}")

    model = SoftmaxRegressor(
        encoder=PairEncoder(encoder, features=pair_features),  # <<<< zelfde gedeelde sencoder en concatenate
        **regressor_kwargs,
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_kwargs["learning_rate"], weight_decay=opt_kwargs["weight_decay"],
    )

    metrics = []
    for epoch in tqdm(range(opt_kwargs["num_epochs"])):
        train_metrics = training_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
        )

    torch.save(model.state_dict(), f"{save_to}.pt")  #save the final epoch not  the best validation epoch.
    with open(f"{save_to}.history.json", "w", encoding="utf-8") as f:  #retain learning curves and predictions.
        json.dump(metrics, f, indent=2)
    print(f"Training done! Time taken: {time.time() - time_dataloader:.3f} seconds")
    return metrics, epoch


def load(path, emb_model: PretrainedEmbeddings, device="cpu"):
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)
    
    if isinstance(device, str):
        device = torch.device(device)

    encoder_args = dict(args["enc_kwargs"])
    pair_features = encoder_args.pop("pair_features", "concat")  # P2: same as in train()

    if args["enc_type"] == "embed-and-pool":
        encoder = EmbedAndPoolEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **encoder_args,
        )
    elif args["enc_type"] == "lstm":
        encoder = LSTMEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **encoder_args,
        )
    else:
        raise ValueError(f"I do not know this encoder type: {args['enc_type']}")
    model = SoftmaxRegressor(
        encoder=PairEncoder(encoder, features=pair_features),  #same pair wrapper used in training.
        **args["regressor_kwargs"],
    ).to(device)
    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f, map_location=device, weights_only=True)  # reload op CPU of GPU.
    model.load_state_dict(state_dict)
    return model


def predict_logprobs(model: SoftmaxRegressor, tokenizer: Tokenizer, pairs, batch_size: int, device="cpu"):
    """Return log-probabilities in input-pair order, following the course predictor."""
    vectors = [(tokenizer.ids(premise), tokenizer.ids(hypothesis)) for premise, hypothesis in pairs]  #aangpast voor pairs
    labels = [0 for _ in vectors]
    pad_id = model.encoder.pad_idx
    dl = DataLoader(
        LabelledCorpus(vectors, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)  # match models device naar input tensors
    model.eval()
    logprobs_list = []
    with torch.no_grad():
        for x_batch, y_batch in dl:
            x_batch = tuple(sentence.to(device) for sentence in x_batch)  #<- aangepasst voor pairs
            logprobs = model(x_batch)
            logprobs_list.extend(logprobs.cpu().numpy())
    return np.asarray(logprobs_list, dtype=np.float32).reshape(-1, model.num_classes)


def predict_probabilities(model, tokenizer, pairs, batch_size=64, device="cpu"):
    """Return probabilities in MNLI label order: entailment, neutral, contradiction."""
    return np.exp(predict_logprobs(model, tokenizer, pairs, batch_size, device))  #reuse existing probability exports


# TODO(Project 2): add any early stopping only with a separate development split
# inside training data; keep the outer fold available for evaluation.
