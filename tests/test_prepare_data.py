import json
from pathlib import Path

import pytest
import yaml
from datasets import Dataset

from prepare_data import load_config, load_section, write_jsonl


def test_load_config_accepts_project_config():
    config = load_config(Path("config.yaml"))

    assert config["seed"] == 42
    assert config["post_training"]["sample_size"] == 1000
    assert config["final_eval"]["sample_size"] is None


def test_load_section_uses_seed_and_sample_size(monkeypatch):
    source = Dataset.from_dict({"value": list(range(20))})
    monkeypatch.setattr("prepare_data.load_dataset", lambda *args, **kwargs: source)
    section = {
        "dataset": "example/data",
        "split": "train",
        "sample_size": 5,
        "shuffle": True,
    }

    first = load_section(section, seed=42)
    second = load_section(section, seed=42)

    assert first["value"] == second["value"]
    assert len(first) == 5


def test_load_config_rejects_nonpositive_sample_size(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "post_training": {
                    "dataset": "a",
                    "split": "train",
                    "output_file": "a.jsonl",
                    "sample_size": 0,
                },
                "final_eval": {
                    "dataset": "b",
                    "split": "test",
                    "output_file": "b.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be null or positive"):
        load_config(config_path)


def test_write_jsonl_returns_checksum_and_writes_rows(tmp_path):
    dataset = Dataset.from_dict({"problem": ["1+1", "2+2"], "answer": ["2", "4"]})
    output_path = tmp_path / "data.jsonl"

    checksum = write_jsonl(dataset, output_path)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert len(checksum) == 64
    assert rows == [
        {"problem": "1+1", "answer": "2"},
        {"problem": "2+2", "answer": "4"},
    ]
