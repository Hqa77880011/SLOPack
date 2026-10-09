import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    config = read_json(path)
    models = config["models"]
    workers = config["workers"]
    if not models or not workers:
        raise ValueError("At least one model and worker are required")
    for kind, items in (("model", models), ("worker", workers)):
        names = [item["name"] for item in items]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate {kind} names")
    sizes = config["batch_sizes"]
    if sizes != sorted(set(sizes)) or not sizes or sizes[0] != 1:
        raise ValueError("batch_sizes must be increasing, unique, and include 1")
    if any(not isinstance(b, int) or b < 1 for b in sizes):
        raise ValueError("Batch sizes must be positive integers")
    names = {model["name"] for model in models}
    for model in models:
        if model["slo_ms"] <= 0 or model.get("weight", 1) <= 0:
            raise ValueError("SLO and traffic weight must be positive")
        if not model["shapes"] or any(s <= 0 for s in model["shapes"]):
            raise ValueError("Model shapes must be positive")
        if not 0 <= model.get("priority", 0.5) <= 1:
            raise ValueError("Priority must lie in [0, 1]")
    for worker in workers:
        if worker["memory_mb"] <= 0:
            raise ValueError("Worker memory must be positive")
        if not set(worker.get("replicas", names)) <= names:
            raise ValueError("Unknown model in worker replicas")
    return config


def prepare_data(source, output, count, seed, root):
    if count < 2:
        raise ValueError("count must be at least 2")
    if source == "cifar10":
        from torchvision.datasets import CIFAR10

        train = CIFAR10(root=root, train=True, download=True)
        val = CIFAR10(root=root, train=False, download=True)
        images = train.data[:count]
        labels = np.asarray(train.targets[:count], dtype=np.int64)
        val_images = val.data[: min(count, len(val))]
        val_labels = np.asarray(val.targets[: len(val_images)], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)

        def make(n):
            labels = rng.integers(0, 10, n, dtype=np.int64)
            images = rng.integers(0, 24, (n, 32, 32, 3), dtype=np.uint8)
            for image, label in zip(images, labels):
                channel = int(label % 3)
                start = int(label * 2)
                image[:, start : start + 8, channel] = 220
            return images, labels

        images, labels = make(count)
        val_images, val_labels = make(max(2, count // 5))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=images,
        labels=labels,
        val_images=val_images,
        val_labels=val_labels,
        source=np.asarray(source),
    )
    write_json(
        output.with_suffix(".json"),
        {
            "source": source,
            "train_samples": len(images),
            "validation_samples": len(val_images),
            "seed": seed,
            "classes": 10,
            "layout": "NHWC uint8",
        },
    )
