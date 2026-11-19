# NLP1 - Project 2

MNLI sentence-pair classification using GloVe with mean pooling or an LSTM.
We compare these with a majority baseline, using code adapted from the Part 2 tutorial.

## Setup

Use Python 3.14 or newer and `uv`. Run these commands from the project root:

```powershell
uv sync
uv run python -m nltk.downloader punkt_tab
```

## Prepare the data

```powershell
uv run python run_data.py
```

This prepares roughly 50,000 MNLI examples and five premise-grouped folds.
Data settings are in `run_data.py`. Skip this step if you already have the matching
manifest in `results/data/`.

## Run the experiments

```powershell
uv run python run_experiments.py
```

This runs majority, GloVe pooling, and GloVe LSTM on the saved folds.
GloVe is downloaded on the first run and reused afterwards. The script uses a GPU
if CUDA is available, otherwise CPU.

Change models, seeds, epochs, and other training settings in `run_experiments.py`.
Keep its data path in sync with the settings in `run_data.py`.
Models, predictions, and scores are saved under `results/part2/`.

## Quick checks

```powershell
uv run python -m unittest discover -s tests -v
```

The tests use small local examples and do not download MNLI or GloVe.

See [PROJECT2_PLAN.md](PROJECT2_PLAN.md) for what we still need to do.
