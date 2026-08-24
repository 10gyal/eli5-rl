"""Run controlled SFT experiments on original or ELI5 MATH solutions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TRAINING_KEYS = (
    "num_train_epochs",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file without teacher-generation dependencies."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_id(source_index: int, problem: str) -> str:
    """Create the stable identifier used by the prepared teacher data."""
    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]
    return f"math_train_{source_index:04d}_{digest}"


def load_sft_config(path: Path) -> dict[str, Any]:
    """Load and validate the shared SFT configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The SFT configuration root must be a mapping.")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer.")
    for section in ("model", "data", "conditions", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping.")
    for key in ("name", "revision"):
        if not config["model"].get(key):
            raise ValueError(f"model.{key} is required.")
    if not isinstance(config["data"].get("max_length"), int):
        raise ValueError("data.max_length must be an integer.")
    eval_fraction = config["data"].get("eval_fraction")
    if not isinstance(eval_fraction, (int, float)) or not 0 < eval_fraction < 1:
        raise ValueError("data.eval_fraction must be between zero and one.")
    if "{problem}" not in config["data"].get("prompt_template", ""):
        raise ValueError("data.prompt_template must contain {problem}.")
    for name in ("original", "eli5"):
        condition = config["conditions"].get(name)
        if not isinstance(condition, dict):
            raise ValueError(f"conditions.{name} must be a mapping.")
        for key in ("data_file", "target_field", "output_dir"):
            if not condition.get(key):
                raise ValueError(f"conditions.{name}.{key} is required.")
    for key in REQUIRED_TRAINING_KEYS:
        if key not in config["training"]:
            raise ValueError(f"training.{key} is required.")
    return config


def resolve_path(config_path: Path, value: str | Path) -> Path:
    """Resolve a path relative to the configuration file."""
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent / path


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add stable sample IDs and source indexes when they are absent."""
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row_index, source in enumerate(rows):
        row = dict(source)
        source_index = int(row.get("source_index", row_index))
        identifier = str(
            row.get("sample_id") or sample_id(source_index, row["problem"])
        )
        if identifier in identifiers:
            raise ValueError(f"Duplicate sample ID: {identifier}")
        identifiers.add(identifier)
        row["source_index"] = source_index
        row["sample_id"] = identifier
        normalized.append(row)
    return normalized


def load_condition_rows(
    config: dict[str, Any], config_path: Path, condition_name: str
) -> list[dict[str, Any]]:
    """Load one condition and apply its optional shared-ID filter."""
    condition = config["conditions"][condition_name]
    rows = normalize_rows(
        read_jsonl(resolve_path(config_path, condition["data_file"]))
    )
    reference_file = condition.get("reference_ids_file")
    if reference_file:
        reference_rows = normalize_rows(
            read_jsonl(resolve_path(config_path, reference_file))
        )
        reference_ids = {row["sample_id"] for row in reference_rows}
        rows = [row for row in rows if row["sample_id"] in reference_ids]
        found_ids = {row["sample_id"] for row in rows}
        missing = reference_ids - found_ids
        if missing:
            raise ValueError(
                f"{condition_name} is missing {len(missing)} reference sample IDs."
            )
    target_field = condition["target_field"]
    for row in rows:
        if not str(row.get(target_field, "")).strip():
            raise ValueError(
                f"{row['sample_id']} has no value in target field {target_field}."
            )
    return rows


def split_rows(
    rows: list[dict[str, Any]], eval_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a stable split that depends only on seed and sample ID."""
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode("utf-8")
        ).hexdigest(),
    )
    eval_count = max(1, round(len(rows) * eval_fraction))
    eval_ids = {row["sample_id"] for row in ranked[:eval_count]}
    train_rows = [row for row in rows if row["sample_id"] not in eval_ids]
    eval_rows = [row for row in rows if row["sample_id"] in eval_ids]
    return train_rows, eval_rows


def format_example(
    config: dict[str, Any], condition_name: str, row: dict[str, Any]
) -> tuple[str, str]:
    """Build a problem-only prompt and its supervised target."""
    condition = config["conditions"][condition_name]
    prompt = config["data"]["prompt_template"].format(problem=row["problem"])
    if not prompt.endswith((" ", "\n", "\t")):
        prompt += "\n"
    target = str(row[condition["target_field"]]).strip()
    if condition.get("append_boxed_answer", False):
        answer_field = condition.get("answer_field", "answer")
        answer = str(row.get(answer_field, "")).strip()
        if not answer:
            raise ValueError(f"{row['sample_id']} has no answer in {answer_field}.")
        target += (
            "\n\nTherefore, the final answer is "
            + rf"$\boxed{{{answer}}}$."
        )
    return prompt, target


def validate_conditions(
    config: dict[str, Any], config_path: Path
) -> dict[str, dict[str, Any]]:
    """Confirm both conditions use exactly the same IDs and split."""
    loaded = {
        name: load_condition_rows(config, config_path, name)
        for name in ("original", "eli5")
    }
    original_ids = {row["sample_id"] for row in loaded["original"]}
    eli5_ids = {row["sample_id"] for row in loaded["eli5"]}
    if original_ids != eli5_ids:
        raise ValueError(
            "The original and ELI5 conditions do not contain the same sample IDs."
        )

    summary: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, tuple[set[str], set[str]]] = {}
    for name, rows in loaded.items():
        train_rows, eval_rows = split_rows(
            rows, config["data"]["eval_fraction"], config["seed"]
        )
        split_ids[name] = (
            {row["sample_id"] for row in train_rows},
            {row["sample_id"] for row in eval_rows},
        )
        target_characters = [len(format_example(config, name, row)[1]) for row in rows]
        summary[name] = {
            "total_rows": len(rows),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "mean_target_characters": round(
                sum(target_characters) / len(target_characters), 2
            ),
            "max_target_characters": max(target_characters),
        }
    if split_ids["original"] != split_ids["eli5"]:
        raise ValueError("The conditions do not use the same train/eval split.")
    return summary


class TokenizedSFTDataset:
    """Small in-memory completion-only causal language modeling dataset."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        config: dict[str, Any],
        condition_name: str,
        tokenizer: Any,
    ) -> None:
        self.items: list[dict[str, list[int]]] = []
        max_length = config["data"]["max_length"]
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The tokenizer has no EOS token.")
        for row in rows:
            prompt, target = format_example(config, condition_name, row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids or target_ids[-1] != eos_token_id:
                target_ids.append(eos_token_id)
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                raise ValueError(
                    f"{row['sample_id']} uses {len(input_ids)} tokens, above "
                    f"data.max_length={max_length}. Increase max_length; the "
                    "trainer will not remove part of a mathematical solution."
                )
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class CompletionCollator:
    """Dynamically pad inputs while keeping prompt labels masked."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def write_manifest(
    output_dir: Path,
    config_path: Path,
    condition_name: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    """Record the exact data IDs used by a training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "condition": condition_name,
        "config": str(config_path.resolve()),
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "eval_sample_ids": [row["sample_id"] for row in eval_rows],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def train(
    config_path: Path,
    condition_name: str,
    resume_from_checkpoint: str | bool | None = None,
) -> None:
    """Fine-tune one condition on a CUDA GPU."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = load_sft_config(config_path)
    validate_conditions(config, config_path)
    if condition_name not in config["conditions"]:
        raise ValueError(f"Unknown condition: {condition_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT training.")
    if config["training"].get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16.")

    set_seed(config["seed"])
    rows = load_condition_rows(config, config_path, condition_name)
    train_rows, eval_rows = split_rows(
        rows, config["data"]["eval_fraction"], config["seed"]
    )
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], revision=model_config["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TokenizedSFTDataset(
        train_rows, config, condition_name, tokenizer
    )
    eval_dataset = TokenizedSFTDataset(eval_rows, config, condition_name, tokenizer)
    dtype = torch.bfloat16 if model_config.get("dtype") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        dtype=dtype,
        attn_implementation=model_config.get("attention_implementation", "sdpa"),
    )
    model.config.use_cache = False

    training = config["training"]
    output_dir = resolve_path(
        config_path, config["conditions"][condition_name]["output_dir"]
    )
    write_manifest(output_dir, config_path, condition_name, train_rows, eval_rows)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"qwen3-0.6b-sft-{condition_name}",
        num_train_epochs=training["num_train_epochs"],
        learning_rate=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_steps=training.get("warmup_steps", 0),
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training.get("max_grad_norm", 1.0),
        optim=training.get("optim", "adamw_torch_fused"),
        bf16=training.get("bf16", True),
        tf32=training.get("tf32", True),
        use_cache=False,
        logging_steps=training.get("logging_steps", 5),
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=training.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=training.get("save_steps", 50),
        save_total_limit=training.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=training.get("report_to", "none"),
        dataloader_num_workers=training.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("sft_config.yaml"),
    )
    parser.add_argument("--condition", choices=("original", "eli5"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint", nargs="?", const=True, default=None
    )
    args = parser.parse_args()

    config = load_sft_config(args.config)
    if args.validate_only:
        print(json.dumps(validate_conditions(config, args.config), indent=2))
        return
    if args.condition is None:
        parser.error("--condition is required unless --validate-only is used.")
    train(args.config, args.condition, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
