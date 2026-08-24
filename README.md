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
TF32 support. If CUDA reports an out-of-memory error, reduce
`evaluation.batch_size` in `benchmark_config.yaml` to 8 or 4 and run the same
command again. Completed problems are not repeated. CUDA results are written
to `results/qwen3_0.6b_base_math500_cuda/` so they do not mix with earlier MPS
results.

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

Then generate all 1,000 rows:

```bash
uv run python generate_eli5.py --config teacher_config.yaml
```

The script uses `load_dotenv` to read `GOOGLE_API_KEY`. It reads the gold answer
directly from the nlile dataset `answer` field. Gemini returns structured JSON.
Math-Verify checks the generated answer before the row is saved. The script
saves each successful row immediately, retries temporary or validation failures,
supports resume, and writes unresolved failures to a separate JSONL file.
