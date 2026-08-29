# Data used to train the difficulty probe

The probe is trained from the `train` split of the Hugging Face
`deepmind/code_contests` dataset.

Only records satisfying both conditions below are used:

- `source == 2`, which identifies Codeforces problems;
- `cf_rating > 0`, which removes unrated problems.

For each retained problem:

- **Input text:** the problem `description`, placed in the Qwen chat template
  with the system instruction to put the final answer in `\boxed{}`;
- **Model feature:** the frozen Qwen2.5-1.5B-Instruct final-layer hidden state
  at the last input token (1,536 values);
- **Training target:** the problem's Codeforces rating, stored as
  `real_difficulty`.

The base language model is not fine-tuned. Its hidden-state embeddings are
cached in `data/statistics/codecontests_emb_<model-tag>.parquet`, and a single
linear layer is trained to regress from each embedding to the rating.

The examples are divided into approximately 64% training, 16% validation, and
20% test data using seed 42. Feature and target standardization are fit only on
the training/validation portion. After training, both standardizations are
folded into the linear layer, so the saved probe accepts raw embeddings and
outputs a Codeforces-scale difficulty score.
