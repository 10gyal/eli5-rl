"""Benchmark Qwen3-0.6B-Base on MATH-500 with the Minerva four-shot method."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from math_verify import parse, verify


REQUIRED_SECTIONS = ("model", "evaluation", "prompt", "generation")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the benchmark configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer.")
    for section_name in REQUIRED_SECTIONS:
        if not isinstance(config.get(section_name), dict):
            raise ValueError(f"{section_name} must be a mapping.")

    required_values = {
        "model.name": config["model"].get("name"),
        "evaluation.dataset_path": config["evaluation"].get("dataset_path"),
        "evaluation.output_dir": config["evaluation"].get("output_dir"),
        "prompt.fewshot_path": config["prompt"].get("fewshot_path"),
    }
    for key, value in required_values.items():
        if not value:
            raise ValueError(f"{key} is required.")
    if config["prompt"].get("style") != "minerva_4shot":
        raise ValueError("prompt.style must be minerva_4shot.")

    max_samples = config["evaluation"].get("max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int) or max_samples <= 0
    ):
        raise ValueError("evaluation.max_samples must be null or positive.")
    start_index = config["evaluation"].get("start_index", 0)
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("evaluation.start_index must be zero or positive.")
    batch_size = config["evaluation"].get("batch_size", 1)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("evaluation.batch_size must be positive.")
    if config["generation"].get("max_new_tokens", 0) <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split items into ordered batches."""
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_prompt(
    problem: str, fewshot_examples: list[dict[str, Any]]
) -> str:
    """Build the Minerva four-shot completion prompt."""
    sections = [
        f"Problem:\n{example['problem']}\n\nSolution:{example['solution']}"
        for example in fewshot_examples
    ]
    sections.append(f"Problem:\n{problem}\n\nSolution:")
    return "\n\n".join(sections)


def extract_final_answer(response: str) -> str:
    """Extract the Minerva final-answer field, with full-response fallback."""
    matches = re.findall(
        r"Final Answer:\s*The final answer is\s*(.*?)(?:\s*I hope it is correct\.|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = matches[-1].strip() if matches else response
    return re.sub(r"\.\s*\$$", "$", candidate.strip())


def score_answer(
    gold_answer: str, response: str
) -> tuple[bool, list[str], list[str]]:
    """Extract and compare the reference and generated final answers."""
    gold_parsed = parse(rf"$\boxed{{{gold_answer}}}$")
    prediction_parsed = parse(extract_final_answer(response))
    correct = bool(gold_parsed and prediction_parsed and verify(gold_parsed, prediction_parsed))
    return correct, [str(item) for item in gold_parsed], [str(item) for item in prediction_parsed]


def choose_device(requested: str, torch_module: Any) -> str:
    """Select the requested or best available device."""
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(requested: str, device: str, torch_module: Any) -> Any:
    """Select a safe inference data type."""
    if requested != "auto":
        value = getattr(torch_module, requested, None)
        if value is None:
            raise ValueError(f"Unsupported dtype: {requested}")
        return value
    if device == "cuda":
        return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16
    if device == "mps":
        return torch_module.float16
    return torch_module.float32


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    """Load valid completed rows for resume support."""
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed[str(row["unique_id"])] = row
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid result at {path}:{line_number}") from error
    return completed


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Calculate accuracy for each value of one field."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(bool(row["correct"]))
    return {
        name: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for name, values in sorted(groups.items())
    }


def build_summary(
    rows: list[dict[str, Any]], config: dict[str, Any], elapsed_seconds: float
) -> dict[str, Any]:
    """Build the benchmark summary."""
    correct = sum(bool(row["correct"]) for row in rows)
    total = len(rows)
    return {
        "model": config["model"],
        "seed": config["seed"],
        "evaluation": config["evaluation"],
        "prompt": config["prompt"],
        "generation": config["generation"],
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "by_subject": grouped_accuracy(rows, "subject"),
        "by_level": grouped_accuracy(rows, "level"),
    }


def run_benchmark(
    config_path: Path,
    max_samples_override: int | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Run the configured benchmark and return its summary."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    config = load_config(config_path)
    project_dir = config_path.resolve().parent
    evaluation = config["evaluation"]
    dataset_path = Path(evaluation["dataset_path"])
    output_dir = output_dir_override or Path(evaluation["output_dir"])
    if not dataset_path.is_absolute():
        dataset_path = project_dir / dataset_path
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    rows = read_jsonl(dataset_path)
    start_index = evaluation.get("start_index", 0)
    max_samples = (
        max_samples_override
        if max_samples_override is not None
        else evaluation.get("max_samples")
    )
    rows = rows[start_index : start_index + max_samples if max_samples else None]
    if not rows:
        raise ValueError("The selected evaluation range is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    completed = load_completed(results_path) if evaluation.get("resume", True) else {}

    seed = config["seed"]
    random.seed(seed)
    set_seed(seed)
    device = choose_device(config["model"].get("device", "auto"), torch)
    dtype = choose_dtype(config["model"].get("dtype", "auto"), device, torch)
    if device == "cuda" and config["model"].get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
    model_name = config["model"]["name"]
    revision = config["model"].get("revision")
    print(f"Loading {model_name} on {device} with {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_load_args: dict[str, Any] = {}
    attention_implementation = config["model"].get("attention_implementation")
    if attention_implementation:
        model_load_args["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        **model_load_args,
    ).to(device)
    model.eval()

    generation = config["generation"]
    generation_args = {
        key: generation[key]
        for key in (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "stop_strings",
            "repetition_penalty",
        )
        if key in generation
    }
    if not generation_args.get("do_sample", False):
        for sampling_key in ("temperature", "top_p", "top_k", "min_p"):
            generation_args.pop(sampling_key, None)
    fewshot_path = Path(config["prompt"]["fewshot_path"])
    if not fewshot_path.is_absolute():
        fewshot_path = project_dir / fewshot_path
    fewshot_examples = read_jsonl(fewshot_path)
    if len(fewshot_examples) != 4:
        raise ValueError("The Minerva prompt must contain four examples.")

    started = time.perf_counter()
    file_mode = "a" if completed else "w"
    with results_path.open(file_mode, encoding="utf-8") as result_stream:
        pending: list[tuple[int, dict[str, Any]]] = []
        for position, example in enumerate(rows, start=1):
            if str(example["unique_id"]) in completed:
                print(f"[{position}/{len(rows)}] Skip {example['unique_id']}")
            else:
                pending.append((position, example))

        batch_size = evaluation.get("batch_size", 1)
        for batch in batches(pending, batch_size):
            prompts = [
                build_prompt(example["problem"], fewshot_examples)
                for _, example in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            batch_started = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        **generation_args,
                        pad_token_id=tokenizer.pad_token_id,
                        tokenizer=tokenizer,
                    )
            except torch.OutOfMemoryError as error:
                raise RuntimeError(
                    "CUDA ran out of memory. Reduce evaluation.batch_size in "
                    "benchmark_config.yaml and resume the run."
                ) from error
            batch_elapsed = time.perf_counter() - batch_started
            input_width = inputs["input_ids"].shape[1]

            for batch_index, (position, example) in enumerate(batch):
                unique_id = str(example["unique_id"])
                token_ids = generated[batch_index, input_width:].tolist()
                if tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
                response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                correct, parsed_gold, parsed_prediction = score_answer(
                    str(example["answer"]), response
                )
                result = {
                    **example,
                    "response": response,
                    "parsed_gold": parsed_gold,
                    "parsed_prediction": parsed_prediction,
                    "correct": correct,
                    "input_tokens": int(
                        inputs["attention_mask"][batch_index].sum().item()
                    ),
                    "output_tokens": len(token_ids),
                    "batch_elapsed_seconds": batch_elapsed,
                }
                result_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_stream.flush()
                completed[unique_id] = result
                status = "correct" if correct else "wrong"
                print(f"[{position}/{len(rows)}] {unique_id}: {status}")

    selected_ids = {str(row["unique_id"]) for row in rows}
    selected_results = [row for key, row in completed.items() if key in selected_ids]
    summary = build_summary(selected_results, config, time.perf_counter() - started)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(
        f"Accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.2%})"
    )
    print(f"Results: {results_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("benchmark_config.yaml"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_benchmark(args.config, args.max_samples, args.output_dir)


if __name__ == "__main__":
    main()
