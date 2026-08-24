"""Submit and collect Gemini Batch API jobs for the ELI5 MATH dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from generate_eli5 import (
    TeacherOutput,
    answers_match,
    build_prompt,
    load_completed,
    load_config,
    read_jsonl,
    sample_id,
    usage_metadata,
)


TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def utc_now() -> str:
    """Return a UTC timestamp for local job records."""
    return datetime.now(UTC).isoformat()


def state_name(job: Any) -> str:
    """Return a stable text form of a batch job state."""
    state = getattr(job, "state", None)
    return str(getattr(state, "name", state or "JOB_STATE_UNSPECIFIED"))


def resolve_path(project_dir: Path, value: str | Path) -> Path:
    """Resolve a configuration path relative to the project directory."""
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON file only after its new contents are complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_runtime(config_path: Path) -> tuple[dict[str, Any], Path, Any]:
    """Load configuration, API key, and Gemini client."""
    from google import genai

    config = load_config(config_path)
    if not isinstance(config.get("batch"), dict):
        raise ValueError("batch must be a mapping in teacher_config.yaml.")
    for key in ("job_file", "display_name"):
        if not config["batch"].get(key):
            raise ValueError(f"batch.{key} is required.")

    project_dir = config_path.resolve().parent
    load_dotenv(dotenv_path=project_dir / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in .env or the environment.")
    return config, project_dir, genai.Client(api_key=api_key)


def request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the generation configuration shared by all batch requests."""
    teacher = config["teacher"]
    return {
        "max_output_tokens": teacher.get("max_output_tokens", 8192),
        "response_mime_type": "application/json",
        "response_schema": TeacherOutput.model_json_schema(),
        "thinking_config": {
            "thinking_level": teacher.get("thinking_level", "medium")
        },
        "seed": config["seed"],
    }


def build_batch_requests(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ordered inline requests and their local row metadata."""
    data_config = config["data"]
    start_index = data_config.get("start_index", 0)
    limit = max_samples if max_samples is not None else data_config.get("max_samples")
    selected = list(enumerate(rows))[start_index:]
    if limit is not None:
        selected = selected[:limit]

    requests: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    common_config = request_config(config)
    for source_index, row in selected:
        answer = str(row.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"Source row {source_index} has no answer field.")
        identifier = sample_id(source_index, row["problem"])
        prompt = build_prompt(row, answer, config["prompt"]["style_instruction"])
        requests.append(
            {
                "contents": [
                    {
                        "parts": [{"text": prompt}],
                        "role": "user",
                    }
                ],
                "config": common_config,
            }
        )
        metadata.append(
            {
                "sample_id": identifier,
                "source_index": source_index,
            }
        )
    if not requests:
        raise ValueError("The selected input range is empty.")
    return requests, metadata


def get_job_file(
    config: dict[str, Any], project_dir: Path, override: Path | None
) -> Path:
    """Resolve the local batch job record path."""
    return resolve_path(project_dir, override or config["batch"]["job_file"])


def submit(
    config_path: Path,
    max_samples: int | None = None,
    job_file_override: Path | None = None,
) -> dict[str, Any]:
    """Submit all incomplete selected rows as one inline batch job."""
    config, project_dir, client = load_runtime(config_path)
    job_file = get_job_file(config, project_dir, job_file_override)
    if job_file.exists():
        existing = json.loads(job_file.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Job record already exists for {existing.get('job_name', 'an unknown job')}: "
            f"{job_file}. Collect it or select another --job-file."
        )

    input_path = resolve_path(project_dir, config["data"]["input_file"])
    output_path = resolve_path(project_dir, config["data"]["output_file"])
    rows = read_jsonl(input_path)
    requests, metadata = build_batch_requests(config, rows, max_samples)
    completed = load_completed(output_path)
    pending = [
        (request, item)
        for request, item in zip(requests, metadata, strict=True)
        if item["sample_id"] not in completed
    ]
    if not pending:
        raise ValueError("All selected samples are already complete.")
    requests = [item[0] for item in pending]
    metadata = [item[1] for item in pending]

    encoded_size = len(json.dumps(requests, ensure_ascii=False).encode("utf-8"))
    if encoded_size >= 20_000_000:
        raise ValueError(
            f"Inline batch is {encoded_size:,} bytes. It must be below 20,000,000."
        )

    job = client.batches.create(
        model=config["teacher"]["model"],
        src=requests,
        config={"display_name": config["batch"]["display_name"]},
    )
    record = {
        "version": 1,
        "job_name": job.name,
        "state": state_name(job),
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
        "model": config["teacher"]["model"],
        "config_path": str(config_path.resolve()),
        "input_file": str(input_path.resolve()),
        "output_file": str(output_path.resolve()),
        "failure_file": str(
            resolve_path(project_dir, config["data"]["failure_file"]).resolve()
        ),
        "request_count": len(requests),
        "request_bytes": encoded_size,
        "samples": metadata,
        "collected_sample_ids": [],
    }
    atomic_write_json(job_file, record)
    print(f"Submitted {len(requests):,} requests as {job.name}")
    print(f"Request size: {encoded_size:,} bytes")
    print(f"Job record: {job_file}")
    return record


def read_record(job_file: Path) -> dict[str, Any]:
    """Read one local batch job record."""
    if not job_file.exists():
        raise FileNotFoundError(f"Batch job record does not exist: {job_file}")
    return json.loads(job_file.read_text(encoding="utf-8"))


def refresh_status(client: Any, job_file: Path) -> tuple[Any, dict[str, Any]]:
    """Fetch job state and update its local record."""
    record = read_record(job_file)
    job = client.batches.get(name=record["job_name"])
    record["state"] = state_name(job)
    record["updated_at"] = utc_now()
    if getattr(job, "error", None):
        record["job_error"] = str(job.error)
    atomic_write_json(job_file, record)
    print(f"{record['job_name']}: {record['state']}")
    return job, record


def status(config_path: Path, job_file_override: Path | None = None) -> str:
    """Print and return the current batch job state."""
    config, project_dir, client = load_runtime(config_path)
    job_file = get_job_file(config, project_dir, job_file_override)
    job, _ = refresh_status(client, job_file)
    return state_name(job)


def result_row(
    config: dict[str, Any],
    row: dict[str, Any],
    source_index: int,
    identifier: str,
    response: Any,
) -> dict[str, Any]:
    """Validate one batch response and build its saved training row."""
    gold_answer = str(row["answer"]).strip()
    teacher_output = TeacherOutput.model_validate_json(response.text)
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
    return {
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
        "attempts": 1,
        "request_mode": "batch",
        "usage": usage_metadata(response),
    }


def collect(
    config_path: Path, job_file_override: Path | None = None
) -> dict[str, int]:
    """Validate and save all responses from one completed batch job."""
    config, project_dir, client = load_runtime(config_path)
    job_file = get_job_file(config, project_dir, job_file_override)
    job, record = refresh_status(client, job_file)
    state = state_name(job)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch is not ready to collect: {state}")
    responses = getattr(getattr(job, "dest", None), "inlined_responses", None)
    if responses is None:
        raise RuntimeError("The completed batch has no inline responses.")
    if len(responses) != len(record["samples"]):
        raise RuntimeError(
            f"Expected {len(record['samples'])} responses, received {len(responses)}."
        )

    rows = read_jsonl(Path(record["input_file"]))
    output_path = Path(record["output_file"])
    failure_path = Path(record["failure_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    collected = set(record.get("collected_sample_ids", []))
    counts = {"completed": 0, "failed": 0, "skipped": 0}

    with (
        output_path.open("a", encoding="utf-8") as output_stream,
        failure_path.open("a", encoding="utf-8") as failure_stream,
    ):
        for position, (item, inline) in enumerate(
            zip(record["samples"], responses, strict=True), start=1
        ):
            identifier = item["sample_id"]
            source_index = item["source_index"]
            if identifier in collected or identifier in completed:
                counts["skipped"] += 1
                continue
            row = rows[source_index]
            if sample_id(source_index, row["problem"]) != identifier:
                raise RuntimeError(f"Input row changed for {identifier}.")
            try:
                if getattr(inline, "error", None):
                    raise RuntimeError(str(inline.error))
                response = getattr(inline, "response", None)
                if response is None:
                    raise RuntimeError("The batch item has no response.")
                saved = result_row(
                    config, row, source_index, identifier, response
                )
                output_stream.write(json.dumps(saved, ensure_ascii=False) + "\n")
                output_stream.flush()
                completed.add(identifier)
                counts["completed"] += 1
                print(f"[{position}/{len(responses)}] Wrote {identifier}")
            except Exception as error:
                failure = {
                    "sample_id": identifier,
                    "source_index": source_index,
                    "gold_answer": str(row.get("answer", "")).strip(),
                    "error": f"{type(error).__name__}: {error}",
                    "request_mode": "batch",
                    **row,
                }
                failure_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_stream.flush()
                counts["failed"] += 1
                print(f"[{position}/{len(responses)}] Fail {identifier}: {error}")

            collected.add(identifier)
            record["collected_sample_ids"] = sorted(collected)
            record["updated_at"] = utc_now()
            atomic_write_json(job_file, record)

    print(json.dumps(counts, indent=2))
    print(f"Output: {output_path}")
    return counts


def wait_and_collect(
    config_path: Path,
    job_file_override: Path | None = None,
    poll_seconds_override: int | None = None,
) -> dict[str, int]:
    """Poll a submitted job until it ends, then collect successful results."""
    config, project_dir, client = load_runtime(config_path)
    job_file = get_job_file(config, project_dir, job_file_override)
    poll_seconds = poll_seconds_override or config["batch"].get(
        "poll_interval_seconds", 30
    )
    while True:
        job, _ = refresh_status(client, job_file)
        state = state_name(job)
        if state in TERMINAL_STATES:
            break
        time.sleep(poll_seconds)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch ended without success: {state}")
    return collect(config_path, job_file_override)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "collect", "wait"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("teacher_config.yaml"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--job-file", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=None)
    args = parser.parse_args()

    if args.action == "submit":
        submit(args.config, args.max_samples, args.job_file)
    elif args.action == "status":
        status(args.config, args.job_file)
    elif args.action == "collect":
        collect(args.config, args.job_file)
    else:
        wait_and_collect(args.config, args.job_file, args.poll_seconds)


if __name__ == "__main__":
    main()
