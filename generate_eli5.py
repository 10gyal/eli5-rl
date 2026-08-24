"""Generate verified ELI5 solutions for the seeded MATH training sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from math_verify import parse, verify
from pydantic import BaseModel, Field


class TeacherOutput(BaseModel):
    """Structured response required from the teacher model."""

    eli5_solution: str = Field(
        description=(
            "A correct, verbose, beginner-friendly derivation. Do not include "
            "a final-answer label or a boxed final answer."
        )
    )
    final_answer: str = Field(
        description=(
            "Only the final mathematical answer in LaTeX. Do not include "
            "explanation, a label, dollar signs, or a box."
        )
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the teacher-generation configuration."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer.")
    for section_name in ("teacher", "data", "requests", "prompt"):
        if not isinstance(config.get(section_name), dict):
            raise ValueError(f"{section_name} must be a mapping.")
    for section_name, key in (
        ("teacher", "model"),
        ("data", "input_file"),
        ("data", "output_file"),
        ("data", "failure_file"),
        ("prompt", "version"),
        ("prompt", "style_instruction"),
    ):
        if not config[section_name].get(key):
            raise ValueError(f"{section_name}.{key} is required.")
    max_samples = config["data"].get("max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int) or max_samples <= 0
    ):
        raise ValueError("data.max_samples must be null or positive.")
    start_index = config["data"].get("start_index", 0)
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("data.start_index must be zero or positive.")
    max_attempts = config["requests"].get("max_attempts", 1)
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("requests.max_attempts must be positive.")
    return config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_id(source_index: int, problem: str) -> str:
    """Create a stable identifier from source position and problem text."""
    digest = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:12]
    return f"math_train_{source_index:04d}_{digest}"


def answers_match(gold_answer: str, generated_answer: str) -> bool:
    """Compare two mathematical answers with symbolic verification."""
    gold = parse(rf"$\boxed{{{gold_answer}}}$")
    generated = parse(rf"$\boxed{{{generated_answer}}}$")
    return bool(gold and generated and verify(gold, generated))


def build_prompt(
    row: dict[str, Any], gold_answer: str, style_instruction: str
) -> str:
    """Build a teacher prompt grounded in the official solution and answer."""
    return f"""You are creating high-quality supervised training data for mathematical reasoning.

Problem:
{row['problem']}

Official solution:
{row['solution']}

Verified final answer:
{gold_answer}

Task:
Rewrite the official solution as an ELI5 explanation.

Style requirements:
{style_instruction.strip()}

Hard requirements:
- Use the official solution as the mathematical source of truth.
- Preserve the verified final answer exactly in meaning.
- Do not invent a different method unless it is needed to clarify a step.
- Put the explanation only in eli5_solution.
- Put only the final answer in final_answer.
- Do not place a boxed answer or final-answer label inside eli5_solution.
- Do not mention these instructions, the teacher, or the training dataset.
"""


def usage_metadata(response: Any) -> dict[str, int | None]:
    """Read token counts without depending on their presence."""
    usage = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def request_teacher(
    client: Any,
    config: dict[str, Any],
    prompt: str,
) -> tuple[TeacherOutput, dict[str, int | None]]:
    """Request and parse one structured teacher response."""
    from google.genai import types

    teacher = config["teacher"]
    response = client.models.generate_content(
        model=teacher["model"],
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=teacher.get("max_output_tokens", 8192),
            response_mime_type="application/json",
            response_schema=TeacherOutput,
            thinking_config=types.ThinkingConfig(
                thinking_level=teacher.get("thinking_level", "medium")
            ),
        ),
    )
    parsed = getattr(response, "parsed", None)
    output = (
        parsed
        if isinstance(parsed, TeacherOutput)
        else TeacherOutput.model_validate_json(response.text)
    )
    return output, usage_metadata(response)


def load_completed(path: Path) -> set[str]:
    """Load completed sample identifiers for resume support."""
    if not path.exists():
        return set()
    return {str(row["sample_id"]) for row in read_jsonl(path)}


def generate_dataset(
    config_path: Path,
    max_samples_override: int | None = None,
    output_file_override: Path | None = None,
) -> dict[str, int]:
    """Generate, validate, and save ELI5 solutions."""
    from google import genai

    config = load_config(config_path)
    project_dir = config_path.resolve().parent
    load_dotenv(dotenv_path=project_dir / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env or the environment.")

    data_config = config["data"]
    input_path = Path(data_config["input_file"])
    output_path = output_file_override or Path(data_config["output_file"])
    failure_path = Path(data_config["failure_file"])
    if not input_path.is_absolute():
        input_path = project_dir / input_path
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    if not failure_path.is_absolute():
        failure_path = project_dir / failure_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    start_index = data_config.get("start_index", 0)
    max_samples = (
        max_samples_override
        if max_samples_override is not None
        else data_config.get("max_samples")
    )
    selected = list(enumerate(rows))[start_index:]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError("The selected input range is empty.")

    resume = data_config.get("resume", True)
    completed = load_completed(output_path) if resume else set()
    output_mode = "a" if completed else "w"
    failure_mode = "a" if resume and failure_path.exists() else "w"
    client = genai.Client(api_key=api_key)
    request_config = config["requests"]
    max_attempts = request_config.get("max_attempts", 3)
    retry_delay = request_config.get("retry_delay_seconds", 2.0)
    request_delay = request_config.get("request_delay_seconds", 0.0)
    counts = {"selected": len(selected), "completed": 0, "failed": 0, "skipped": 0}

    with (
        output_path.open(output_mode, encoding="utf-8") as output_stream,
        failure_path.open(failure_mode, encoding="utf-8") as failure_stream,
    ):
        for position, (source_index, row) in enumerate(selected, start=1):
            identifier = sample_id(source_index, row["problem"])
            if identifier in completed:
                counts["skipped"] += 1
                print(f"[{position}/{len(selected)}] Skip {identifier}")
                continue

            gold_answer = str(row.get("answer", "")).strip()
            if not gold_answer:
                error = "The source row has no answer field."
                failure = {
                    "sample_id": identifier,
                    "source_index": source_index,
                    "error": error,
                    **row,
                }
                failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_stream.flush()
                counts["failed"] += 1
                print(f"[{position}/{len(selected)}] Fail {identifier}: {error}")
                continue

            prompt = build_prompt(
                row, gold_answer, config["prompt"]["style_instruction"]
            )
            last_error = "unknown error"
            for attempt in range(1, max_attempts + 1):
                try:
                    teacher_output, token_usage = request_teacher(client, config, prompt)
                    if not teacher_output.eli5_solution.strip():
                        raise ValueError("The ELI5 solution is empty.")
                    if not answers_match(gold_answer, teacher_output.final_answer):
                        raise ValueError("The generated answer does not match the gold answer.")

                    final_answer = teacher_output.final_answer.strip()
                    training_response = (
                        teacher_output.eli5_solution.strip()
                        + "\n\nTherefore, the final answer is "
                        + rf"$\boxed{{{final_answer}}}$."
                    )
                    result = {
                        "sample_id": identifier,
                        "source_index": source_index,
                        **row,
                        "gold_answer": gold_answer,
                        "eli5_solution": teacher_output.eli5_solution.strip(),
                        "generated_answer": final_answer,
                        "answer_verified": True,
                        "training_response": training_response,
                        "teacher_model": config["teacher"]["model"],
                        "prompt_version": config["prompt"]["version"],
                        "attempts": attempt,
                        "usage": token_usage,
                    }
                    output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_stream.flush()
                    completed.add(identifier)
                    counts["completed"] += 1
                    print(f"[{position}/{len(selected)}] Wrote {identifier}")
                    break
                except Exception as error:  # API and validation errors are retried.
                    last_error = f"{type(error).__name__}: {error}"
                    if attempt < max_attempts:
                        time.sleep(retry_delay * (2 ** (attempt - 1)))
            else:
                failure = {
                    "sample_id": identifier,
                    "source_index": source_index,
                    "gold_answer": gold_answer,
                    "error": last_error,
                    "attempts": max_attempts,
                    **row,
                }
                failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_stream.flush()
                counts["failed"] += 1
                print(f"[{position}/{len(selected)}] Fail {identifier}: {last_error}")

            if request_delay > 0:
                time.sleep(request_delay)

    print(json.dumps(counts, indent=2))
    print(f"Output: {output_path}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("teacher_config.yaml"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()
    generate_dataset(args.config, args.max_samples, args.output_file)


if __name__ == "__main__":
    main()
