import json
from pathlib import Path

import pytest

from benchmark_math500 import (
    build_prompt,
    build_summary,
    extract_final_answer,
    load_completed,
    load_config,
    score_answer,
)


def test_benchmark_config_is_valid():
    config = load_config(Path("benchmark_config.yaml"))

    assert config["seed"] == 42
    assert config["model"]["name"] == "Qwen/Qwen3-0.6B-Base"
    assert config["prompt"]["style"] == "minerva_4shot"
    assert config["generation"]["do_sample"] is False


@pytest.mark.parametrize(
    ("gold", "response"),
    [
        (r"\frac{1}{2}", r"The answer is $\boxed{0.5}$"),
        ("12", r"The final answer is $\boxed{12}$"),
    ],
)
def test_score_answer_accepts_equivalent_answers(gold, response):
    correct, parsed_gold, parsed_prediction = score_answer(gold, response)

    assert correct
    assert parsed_gold
    assert parsed_prediction


def test_extract_final_answer_ignores_intermediate_boxes():
    response = (
        r"First, $\boxed{3}$. Final Answer: The final answer is "
        r"$(3,\frac{\pi}{2}).$ I hope it is correct. Problem:"
    )

    assert extract_final_answer(response) == r"$(3,\frac{\pi}{2})$"
    assert score_answer(r"(3,\frac{\pi}{2})", response)[0]


def test_build_prompt_uses_four_examples_and_completion_format():
    examples = [
        {"problem": f"example {index}", "solution": str(index)}
        for index in range(4)
    ]

    prompt = build_prompt("target", examples)

    assert prompt.count("Problem:\n") == 5
    assert prompt.endswith("Problem:\ntarget\n\nSolution:")


def test_load_completed_keeps_latest_row(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '\n'.join(
            [
                json.dumps({"unique_id": "1", "correct": False}),
                json.dumps({"unique_id": "1", "correct": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_completed(path)["1"]["correct"] is True


def test_build_summary_groups_results():
    rows = [
        {"correct": True, "subject": "Algebra", "level": 1},
        {"correct": False, "subject": "Algebra", "level": 2},
    ]
    config = load_config(Path("benchmark_config.yaml"))

    summary = build_summary(rows, config, elapsed_seconds=1.0)

    assert summary["accuracy"] == 0.5
    assert summary["by_subject"]["Algebra"]["total"] == 2
