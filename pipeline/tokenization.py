"""BPE tokenization helpers for MNLI cross-validation folds."""

from pathlib import Path

from part1.experiment import train_tokenizer


def train_fold_tokenizers(folds, vocab_size=1000, min_frequency=2):
    """Train one tokenizer per fold on training-only ((premise, hypothesis), label) pairs.
    Source: part1.experiment.train_tokenizer (adapted per fold).
    """
    fold_tokenizers = {}

    for fold_id, fold in folds.items():
        training_texts = []

        for (premise, hypothesis), label in fold["training"]:
            training_texts.append(premise)
            training_texts.append(hypothesis)

        #Bad fold data
        if not training_texts:
            raise ValueError(f"Fold {fold_id} bevat geen trainingsvoorbeelden.")

        fold_tokenizers[fold_id] = train_tokenizer(
            training_texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    return fold_tokenizers


def inspect_tokenization(tokenizer, examples, max_examples=3):
    """Return the sentences, tokens and IDs for a few sentence pairs.
    Source: tutorial.ipynb, cell 121.
    """
    if max_examples < 0:
        raise ValueError("max_examples moet 0 of groter zijn.")

    inspection = []

    for (premise, hypothesis), label in examples[:max_examples]:
        premise_encoding = tokenizer.encode(premise)
        hypothesis_encoding = tokenizer.encode(hypothesis)

        inspection.append(
            {
                "label": label,
                "premise": premise,
                "premise_tokens": premise_encoding.tokens,
                "premise_ids": premise_encoding.ids,
                "hypothesis": hypothesis,
                "hypothesis_tokens": hypothesis_encoding.tokens,
                "hypothesis_ids": hypothesis_encoding.ids,
            }
        )

    return inspection


def save_fold_tokenizers(tokenizers, output_dir):
    """Save each tokenizer to its fold directory and return the file paths.
    Source: teun_pipeline.ipynb, part1.experiment.cross_validation_experiment.
    
    (we have to discuss output_dir and if we want to push to the git)
    """
    output_dir = Path(output_dir)
    saved_paths = {}

    for fold_id, tokenizer in tokenizers.items():
        tokenizer_path = output_dir / f"fold-{fold_id}" / "tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        saved_paths[fold_id] = tokenizer_path

    return saved_paths


# TODO: Add load_fold_tokenizer when tokenizer loading has been covered.


