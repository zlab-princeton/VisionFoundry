#!/usr/bin/env python3
"""Convert VisionFoundry SFT annotations to verl multimodal RL parquet."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


def ability_name(category: str) -> str:
    name = category.split(":", 1)[0].strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "unknown"


def remap_image(
    image: str, source_image_root: Path | None, image_root: Path | None
) -> Path:
    path = Path(image)
    if image_root is None:
        return path
    if source_image_root is not None:
        try:
            relative = path.relative_to(source_image_root)
        except ValueError as exc:
            raise ValueError(
                f"Image {path} is not below --source-image-root {source_image_root}"
            ) from exc
        return image_root / relative
    if path.is_absolute():
        raise ValueError(
            "Absolute image paths require --source-image-root when --image-root is used"
        )
    return image_root / path


def convert_item(
    item: dict[str, Any],
    split: str,
    index: int,
    data_source: str,
    source_image_root: Path | None,
    image_root: Path | None,
    check_images: bool,
) -> dict[str, Any]:
    messages = item.get("messages", [])
    if len(messages) < 2:
        raise ValueError(f"qid={item.get('qid')} has fewer than two messages")

    question = str(messages[0].get("content", "")).strip()
    if "<image>" not in question:
        question = f"<image>\n{question}"
    answer = str(messages[-1].get("content", "")).strip()
    if not answer:
        raise ValueError(f"qid={item.get('qid')} has an empty answer")

    images = item.get("images") or []
    if len(images) != 1:
        raise ValueError(f"qid={item.get('qid')} expected one image, got {len(images)}")
    image_path = remap_image(images[0], source_image_root, image_root)
    if check_images and not image_path.is_file():
        raise FileNotFoundError(image_path)

    metadata = item.get("metadata") or {}
    category = str(metadata.get("category", "unknown"))
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "images": [{"image": str(image_path)}],
        "ability": ability_name(category),
        "reward_model": {"style": "model", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "qid": item.get("qid"),
            "question": question,
            "answer": answer,
            "category": category,
            "metadata": metadata,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="SFT annotations.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", default="visionfoundry")
    parser.add_argument("--source-image-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    items = json.loads(Path(args.source).read_text(encoding="utf-8"))
    if not isinstance(items, list) or len(items) < 2:
        raise ValueError("The source must be a JSON list containing at least two items")
    if not 0 < args.val_size < len(items):
        raise ValueError("--val-size must be between 1 and dataset size - 1")

    random.Random(args.seed).shuffle(items)
    splits = {"val": items[: args.val_size], "train": items[args.val_size :]}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, split_items in splits.items():
        rows = [
            convert_item(
                item,
                split,
                index,
                args.data_source,
                args.source_image_root,
                args.image_root,
                not args.skip_image_check,
            )
            for index, item in enumerate(split_items)
        ]
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(rows)} -> {output_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()


