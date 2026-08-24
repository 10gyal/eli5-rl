import json
from pathlib import Path
from types import SimpleNamespace

from generate_eli5 import load_config, read_jsonl, sample_id
from generate_eli5_batch import (
    atomic_write_json,
    build_batch_requests,
    result_row,
    state_name,
)


def test_build_batch_requests_uses_dataset_answer():
    config = load_config(Path("teacher_config.yaml"))
    rows = read_jsonl(Path("data/math_train_1000.jsonl"))[:2]

    requests, metadata = build_batch_requests(config, rows)

    assert len(requests) == 2
    assert len(metadata) == 2
    assert rows[0]["answer"] in requests[0]["contents"][0]["parts"][0]["text"]
    assert metadata[0]["sample_id"] == sample_id(0, rows[0]["problem"])
    assert requests[0]["config"]["seed"] == 42
    assert requests[0]["config"]["response_mime_type"] == "application/json"


def test_result_row_verifies_and_formats_response():
    config = load_config(Path("teacher_config.yaml"))
    row = {
        "problem": "What is one half?",
        "solution": "Divide one by two.",
        "answer": r"\frac{1}{2}",
        "subject": "Prealgebra",
        "level": 1,
        "unique_id": "test/1.json",
    }
    response = SimpleNamespace(
        text=json.dumps(
            {
                "eli5_solution": "Split one whole into two equal parts.",
                "final_answer": "0.5",
            }
        ),
        usage_metadata=None,
    )

    saved = result_row(config, row, 0, "sample", response)

    assert saved["answer_verified"] is True
    assert saved["request_mode"] == "batch"
    assert saved["generated_answer"] == "0.5"
    assert saved["training_response"].endswith(r"$\boxed{0.5}$.")


def test_state_name_supports_sdk_enum_shape():
    job = SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_RUNNING"))
    assert state_name(job) == "JOB_STATE_RUNNING"


def test_atomic_write_json(tmp_path):
    path = tmp_path / "job.json"
    atomic_write_json(path, {"job_name": "batches/1"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "job_name": "batches/1"
    }
