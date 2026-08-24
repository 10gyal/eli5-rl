"""Download and prepare the MATH post-training and MATH-500 evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate the YAML configuration."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer.")
    for section_name in ("post_training", "final_eval"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping.")
        for key in ("dataset", "split", "output_file"):
            if not section.get(key):
                raise ValueError(f"{section_name}.{key} is required.")
        sample_size = section.get("sample_size")
        if sample_size is not None and (
            not isinstance(sample_size, int) or sample_size <= 0
        ):
            raise ValueError(f"{section_name}.sample_size must be null or positive.")
    return config


def load_section(section: dict[str, Any], seed: int) -> Dataset:
    """Load, combine, shuffle, and select one configured dataset section."""
    dataset_name = section["dataset"]
    split = section["split"]
    revision = section.get("revision")
    names = section.get("configs") or [None]

    parts = [
        load_dataset(dataset_name, name=name, split=split, revision=revision)
        for name in names
    ]
    dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    sample_size = section.get("sample_size")
    if sample_size is not None and sample_size > len(dataset):
        raise ValueError(
            f"Requested {sample_size} rows from {dataset_name}, but only "
            f"{len(dataset)} rows are available."
        )
    if section.get("shuffle", False):
        dataset = dataset.shuffle(seed=seed)
    if sample_size is not None:
        dataset = dataset.select(range(sample_size))
    return dataset


def write_jsonl(dataset: Dataset, output_path: Path) -> str:
    """Write UTF-8 JSON Lines data and return its SHA-256 checksum."""
    digest = hashlib.sha256()
    with output_path.open("wb") as output_file:
        for row in dataset:
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            output_file.write(line)
            digest.update(line)
    return digest.hexdigest()


def prepare(config_path: Path) -> dict[str, Any]:
    """Create both output datasets and a reproducibility manifest."""
    config = load_config(config_path)
    seed = config["seed"]
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "seed": seed,
        "config": str(config_path.resolve()),
        "outputs": {},
    }
    for section_name in ("post_training", "final_eval"):
        section = config[section_name]
        dataset = load_section(section, seed)
        output_path = output_dir / section["output_file"]
        checksum = write_jsonl(dataset, output_path)
        manifest["outputs"][section_name] = {
            "path": str(output_path.resolve()),
            "dataset": section["dataset"],
            "revision": section.get("revision"),
            "split": section["split"],
            "rows": len(dataset),
            "columns": dataset.column_names,
            "fingerprint": dataset._fingerprint,
            "sha256": checksum,
        }
        print(f"Wrote {len(dataset):,} rows to {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
    print(f"Wrote manifest to {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML configuration path (default: config.yaml next to this script)",
    )
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
