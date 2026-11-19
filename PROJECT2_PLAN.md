# Project 2 plan
## The two models

GloVe + pooling + MLP: average the word vectors in each sentence, then classify the pair.

GloVe + LSTM + MLP: let an LSTM read each sentence in order, then classify the pair.

Both use the same tokenizer and GloVe vocabulary. We encode the premise and
hypothesis separately with shared weights, then put their vectors next to each other:

```text
premise    -> sentence encoder -> p
hypothesis -> same encoder     -> h

[p, h] -> MLP -> entailment / neutral / contradiction
```

We're using as much of the Part 2 tutorial code as possible. The main adjustment
is handling two sentences instead of one. Changes are commented in the code.

## What we want to find out

- Do pretrained embeddings improve on the Project 1 baseline?
- Does the LSTM improve on pooling, and on which kinds of examples?
- Does fine-tuning GloVe help? How does it compare with training random embeddings?
- What happens with less training data or different regularization settings?
- Where do the models still make mistakes?


## Done

- [x] Write the plan and clean up the old project, keeping useful data/evaluation code.
- [x] Add GloVe loading and word tokenization.
- [x] Add pooling, LSTM, and the sentence-pair wrapper.
- [x] Add paired batches and padding.
- [x] Add the classifier, training, evaluation, saving/loading, and prediction.
- [x] Connect both models to the saved folds and result exports.
- [x] Set the models, settings, seeds, and CPU/GPU in run_experiments.py.
## Still to do


- [ ] Download GloVe and record the exact file/version and checksum.
- [ ] Make NLTK's `punkt_tab` available.
- [ ] Run a small MNLI experiment and check the saved predictions.
- [ ] Run the comparisons with different seeds and selected settings.
- [ ] Add embedding coverage, word similarities, learning curves, and error analysis.
- [ ] Evaluate the chosen models on matched/mismatched MNLI.
- [ ] Write up the results and what we learned.
