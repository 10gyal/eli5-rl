import json
from pathlib import Path

from generate_eli5 import (
    answers_match,
    build_prompt,
    load_completed,
    load_config,
    sample_id,
)


def test_teacher_config_is_valid():
    config = load_config(Path("teacher_config.yaml"))

    assert config["seed"] == 42
    assert config["teacher"]["model"] == "gemini-3.5-flash"
    assert config["data"]["max_samples"] is None


def test_answers_match_uses_mathematical_equivalence():
    assert answers_match(r"\frac{1}{2}", "0.5")
    assert not answers_match("3", "4")


def test_build_prompt_contains_all_grounding_fields():
    row = {"problem": "What is 1+1?", "solution": "It is 2."}
    prompt = build_prompt(row, "2", "Explain each small step.")

    assert "What is 1+1?" in prompt
    assert "It is 2." in prompt
    assert "Verified final answer:\n2" in prompt
    assert "Explain each small step." in prompt


def test_sample_id_is_stable_and_position_sensitive():
    assert sample_id(4, "problem") == sample_id(4, "problem")
    assert sample_id(4, "problem") != sample_id(5, "problem")


def test_load_completed(tmp_path):
    path = tmp_path / "complete.jsonl"
    path.write_text(
        json.dumps({"sample_id": "one"}) + "\n" + json.dumps({"sample_id": "two"}) + "\n",
        encoding="utf-8",
    )

    assert load_completed(path) == {"one", "two"}
