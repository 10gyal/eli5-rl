from pathlib import Path

import pytest

from sft_train import (
    TokenizedSFTDataset,
    format_example,
    load_condition_rows,
    load_sft_config,
    split_rows,
    validate_conditions,
)


class TinyTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(1, len(text.split()) + 1))


def test_conditions_use_the_same_rows_and_split():
    path = Path("sft_config.yaml")
    config = load_sft_config(path)
    summary = validate_conditions(config, path)

    assert summary["original"]["total_rows"] == 989
    assert summary["eli5"]["total_rows"] == 989
    assert summary["original"]["train_rows"] == 940
    assert summary["original"]["eval_rows"] == 49
    assert summary["original"]["train_rows"] == summary["eli5"]["train_rows"]
    assert summary["original"]["eval_rows"] == summary["eli5"]["eval_rows"]


def test_split_is_stable_and_independent_of_row_order():
    rows = [{"sample_id": f"id-{index}"} for index in range(20)]
    train_a, eval_a = split_rows(rows, 0.2, 42)
    train_b, eval_b = split_rows(list(reversed(rows)), 0.2, 42)

    assert {row["sample_id"] for row in train_a} == {
        row["sample_id"] for row in train_b
    }
    assert {row["sample_id"] for row in eval_a} == {
        row["sample_id"] for row in eval_b
    }


def test_student_prompt_contains_problem_only():
    path = Path("sft_config.yaml")
    config = load_sft_config(path)
    row = load_condition_rows(config, path, "eli5")[0]
    prompt, target = format_example(config, "eli5", row)

    assert row["problem"] in prompt
    assert row["solution"] not in prompt
    assert row["answer"] not in prompt
    assert row["eli5_solution"] in target
    assert target.endswith(rf"$\boxed{{{row['answer']}}}$.")


def test_original_target_is_not_rewritten():
    path = Path("sft_config.yaml")
    config = load_sft_config(path)
    row = load_condition_rows(config, path, "original")[0]
    _, target = format_example(config, "original", row)
    assert target == row["solution"].strip()


def test_dataset_masks_prompt_and_supervises_target():
    config = {
        "data": {"max_length": 20, "prompt_template": "Problem: {problem}\n"},
        "conditions": {
            "eli5": {
                "target_field": "eli5_solution",
                "append_boxed_answer": False,
            }
        },
    }
    rows = [
        {
            "sample_id": "one",
            "problem": "one plus one",
            "eli5_solution": "two",
        }
    ]
    dataset = TokenizedSFTDataset(rows, config, "eli5", TinyTokenizer())
    item = dataset[0]

    assert item["labels"][:-2] == [-100] * (len(item["labels"]) - 2)
    assert item["labels"][-1] == TinyTokenizer.eos_token_id


def test_dataset_rejects_overlength_solutions():
    config = {
        "data": {"max_length": 2, "prompt_template": "Problem: {problem}\n"},
        "conditions": {
            "original": {
                "target_field": "solution",
                "append_boxed_answer": False,
            }
        },
    }
    rows = [{"sample_id": "one", "problem": "a b", "solution": "c d"}]
    with pytest.raises(ValueError, match="above data.max_length"):
        TokenizedSFTDataset(rows, config, "original", TinyTokenizer())
