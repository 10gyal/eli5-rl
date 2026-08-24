# ELI5-RL

## Dataset setup

The setup creates two JSON Lines files in `data/`:

- `math_train_1000.jsonl`: 1,000 seeded samples from the nlile MATH train split.
- `math_500_test.jsonl`: the fixed 500-row nlile MATH test split (MATH-500).

Change the sample sizes, seed, or output names in `config.yaml`.
The source revisions are fixed to exact commits for reproducible downloads.
Then run:

```bash
uv sync
uv run python prepare_data.py --config config.yaml
```

The command also writes `data/manifest.json`. The manifest records the row
counts, columns, dataset fingerprints, and SHA-256 checksums.

## Qwen3-0.6B-Base MATH-500 benchmark

Edit `benchmark_config.yaml` to change the model, device, token limit, or
evaluation range. The default setup follows the LM Evaluation Harness
`minerva_math500` method: four chain-of-thought examples, greedy decoding, and
symbolic answer checking. The Qwen technical report uses this four-shot method
for base-model MATH evaluation. It does not report MATH-500 for the base model.
The report gives Qwen3-0.6B-Base a score of 32.44 on the original MATH test.
Treat the local MATH-500 result as a separate benchmark result.

Run the complete 500-problem benchmark:

```bash
uv sync
uv run python benchmark_math500.py --config benchmark_config.yaml
```

On an NVIDIA GPU, `device: auto` selects CUDA. The default batch size is 16
for a 24 GB GPU. The loader uses BF16 on supported GPUs, SDPA attention, and
TF32 support. The project pins PyTorch 2.8.0 and uses the CUDA 12.8 PyTorch
package on Linux. After a new checkout, run `uv sync`, then confirm the setup:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The last value must be `True`. If CUDA reports an out-of-memory error, reduce
`evaluation.batch_size` in `benchmark_config.yaml` to 8 or 4 and run the same
command again. Completed problems are not repeated. CUDA results are written
to `results/qwen3_0.6b_base_math500_cuda/` so they do not mix with earlier MPS
results.

On a RunPod image that already has PyTorch 2.8.0 with CUDA 12.8, reuse that
package instead of running `uv sync`:

```bash
uv venv --system-site-packages --python python3 runpod-venv
uv pip install --python runpod-venv/bin/python \
  "transformers>=4.51.0,<6" \
  "math-verify[antlr4_13_2]==0.9.0" \
  "pyyaml>=6.0.2,<7"
runpod-venv/bin/python benchmark_math500.py --config benchmark_config.yaml
```

Run a short smoke test in a separate output directory:

```bash
uv run python benchmark_math500.py \
  --config benchmark_config.yaml \
  --max-samples 1 \
  --output-dir results/smoke
```

The benchmark writes `results.jsonl` after each problem. It can resume an
interrupted run. It also writes `summary.json` with total accuracy and accuracy
by subject and difficulty level. Scoring uses Math-Verify for symbolic
mathematical equivalence.

## Gemini ELI5 teacher data

Put the Gemini API key in `.env`:

```text
GOOGLE_API_KEY=your-key
```

Edit `teacher_config.yaml` to change the model, thinking level, output paths,
retry settings, or sample range. Run one paid smoke-test request first:

```bash
uv sync
uv run python generate_eli5.py \
  --config teacher_config.yaml \
  --max-samples 1 \
  --output-file data/math_eli5_smoke.jsonl
```

Submit all 1,000 rows with the asynchronous Gemini Batch API:

```bash
uv run python generate_eli5_batch.py submit --config teacher_config.yaml
```

The submit command saves the Gemini job name in
`data/gemini_eli5_batch_job.json`. Check its state and collect it after it
finishes:

```bash
uv run python generate_eli5_batch.py status --config teacher_config.yaml
uv run python generate_eli5_batch.py collect --config teacher_config.yaml
```

To keep one terminal open until the job finishes, use `wait` instead of
`status` and `collect`. The Batch API processes the requests asynchronously and
costs 50% of the standard request price. The 1,000 prompts fit in one inline
batch below the 20 MB limit.

Both teacher scripts use `load_dotenv` to read `GOOGLE_API_KEY`. They read the
gold answer directly from the nlile dataset `answer` field. Gemini returns
structured JSON. Math-Verify checks each generated answer before the row is
saved. Batch collection is resumable and writes invalid responses to the
configured failure JSONL file. The direct script remains available for small
tests and retries.

## Controlled SFT comparison

`sft_train.py` supports two full-parameter fine-tuning conditions from one
shared `sft_config.yaml`:

- `original`: the official MATH solution is the target.
- `eli5`: the accepted Gemini ELI5 solution and boxed gold answer are the
  target.

Both conditions use the same 989 accepted sample IDs, the same 940 training
rows, the same 49 validation rows, and the same optimizer settings. The student
input contains only the problem. MATH-500 is not used during training or model
selection.

Validate the comparison without loading or training the model:

```bash
runpod-venv/bin/python sft_train.py --config sft_config.yaml --validate-only
```

The RunPod environment needs Accelerate in addition to the benchmark packages:

```bash
uv pip install --python runpod-venv/bin/python "accelerate>=1.10.0,<2"
```

On the NVIDIA RunPod server, train the official-solution condition:

```bash
runpod-venv/bin/python sft_train.py --config sft_config.yaml --condition original
```

Train the ELI5 condition:

```bash
runpod-venv/bin/python sft_train.py --config sft_config.yaml --condition eli5
```

The default setup uses BF16 full-parameter training, SDPA attention, gradient
checkpointing, a per-device batch size of 2, eight accumulation steps, and an
effective batch size of 16 on one GPU. Checkpoints and final models go to
separate directories under `outputs/`. Use the same configuration for both
conditions. If memory is insufficient, set the batch size to 1 and accumulation
steps to 16 before running either condition.

After both runs finish, evaluate them with separate MATH-500 result directories:

```bash
runpod-venv/bin/python benchmark_math500.py --config benchmark_sft_original.yaml
runpod-venv/bin/python benchmark_math500.py --config benchmark_sft_eli5.yaml
```
