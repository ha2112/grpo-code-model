# Afterburner GRPO

This route trains the Afterburner policy with verl GRPO and execution-based
rewards from Monolith.

## Prepare the data

Install the dependencies in `requirements.txt`, then run:

```bash
python3 afterburner_dataset.py
```

This downloads `Elfsong/Venus_Python` and writes `venus_train.parquet` and
`venus_test.parquet` under `~/data/venus`.

## Train

Deploy or access a Monolith endpoint, install verl, and launch from this
directory:

```bash
./afterburner_train.sh
```

The launcher defaults to `~/data/venus` for the parquet files and the
`Elfsong/Qwen2.5-Coder-3B-Instruct-Venus-Cold-Start` policy. Override them with
`AFTERBURNER_DATA_DIR` and `AFTERBURNER_MODEL_PATH`; override `HF_HOME` to
choose the model cache location.

The reward function submits generated Python solutions to
`https://monolith.cool/execute`, so training requires network access to that
service or a compatible endpoint configured in
`afterburner_reward_function.py`.
