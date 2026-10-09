from collections import deque
from dataclasses import dataclass, field

import numpy as np

from slopack.io import read_json, write_json


@dataclass
class ProfileEntry:
    model: str
    worker: str
    shape: int
    batch_size: int
    median_ms: float
    p95_ms: float
    memory_mb: float
    resident_mb: float
    relative_error_p95: float
    residual_p95_ms: float
    samples_ms: list
    heldout_samples_ms: list = field(default_factory=list)
    estimate_ms: float = field(init=False)
    guard_ms: float = field(init=False)
    stale: bool = False
    observations: int = 0
    errors: deque = field(default_factory=deque)
    residuals: deque = field(default_factory=deque)
    bias_scale: float = field(default=1.0, init=False)

    def __post_init__(self):
        if self.median_ms <= 0 or self.memory_mb < 0 or self.resident_mb < 0:
            raise ValueError("Invalid profile latency or memory")
        if not self.samples_ms or min(self.samples_ms) <= 0:
            raise ValueError("Profiles require positive execution samples")
        self.estimate_ms = self.median_ms
        self.guard_ms = self.residual_p95_ms + 0.2

    @property
    def key(self):
        return self.model, self.worker, self.shape, self.batch_size

    @property
    def conservative_ms(self):
        return self.estimate_ms * (1 + self.relative_error_p95)

    def update(self, observed_ms, settings):
        error = abs(observed_ms - self.estimate_ms) / observed_ms
        self.errors.append(error)
        self.residuals.append(max(0.0, observed_ms - self.estimate_ms))
        window = settings.stale_window
        while len(self.errors) > window:
            self.errors.popleft()
            self.residuals.popleft()
        self.observations += 1
        if self.observations % settings.guard_update_batches == 0:
            self.guard_ms = (
                float(np.percentile(self.residuals, 95)) + settings.guard_floor_ms
            )
        if len(self.errors) == window:
            threshold = settings.stale_factor * max(self.relative_error_p95, 1e-6)
            if float(np.percentile(self.errors, 95)) > threshold:
                self.stale = True
        self.estimate_ms = (
            settings.rho * observed_ms * self.bias_scale
            + (1 - settings.rho) * self.estimate_ms
        )
        return error


class Profiles:
    def __init__(self, entries, metadata):
        self.entries = {}
        self.metadata = metadata
        for entry in entries:
            if entry.key in self.entries:
                raise ValueError(f"Duplicate profile tuple {entry.key}")
            self.entries[entry.key] = entry

    @classmethod
    def load(cls, path, bias=0.0, bias_mode="sustained"):
        if bias <= -1:
            raise ValueError("Profile bias must exceed -1")
        if bias_mode not in ("initial", "sustained"):
            raise ValueError("Unknown profile bias mode")
        document = read_json(path)
        entries = [ProfileEntry(**row) for row in document["entries"]]
        for entry in entries:
            entry.estimate_ms *= 1 + bias
            entry.bias_scale = 1 + bias if bias_mode == "sustained" else 1.0
        return cls(entries, document["metadata"])

    def get(self, model, worker, shape, size):
        return self.entries.get((model, worker, shape, size))

    def resident(self, model, worker):
        values = [
            e.resident_mb
            for e in self.entries.values()
            if e.model == model and e.worker == worker
        ]
        return max(values, default=0.0)

    def snapshot(self):
        return [
            {
                "model": e.model,
                "worker": e.worker,
                "shape": e.shape,
                "batch_size": e.batch_size,
                "estimate_ms": e.estimate_ms,
                "guard_ms": e.guard_ms,
                "stale": e.stale,
                "observations": e.observations,
            }
            for e in self.entries.values()
        ]


def summarize_samples(model, worker, shape, size, samples, heldout, memory, resident):
    median = float(np.median(samples))
    heldout = np.asarray(heldout)
    return {
        "model": model,
        "worker": worker,
        "shape": shape,
        "batch_size": size,
        "median_ms": median,
        "p95_ms": float(np.percentile(samples, 95)),
        "memory_mb": float(memory),
        "resident_mb": float(resident),
        "relative_error_p95": float(np.percentile(abs(heldout - median) / heldout, 95)),
        "residual_p95_ms": float(np.percentile(np.maximum(heldout - median, 0), 95)),
        "samples_ms": [float(value) for value in samples],
        "heldout_samples_ms": [float(value) for value in heldout],
    }


def synthetic_profile(config, output, samples, seed, noise):
    if samples < 2 or noise < 0:
        raise ValueError("samples must be at least 2 and noise nonnegative")
    rng = np.random.default_rng(seed)
    entries = []
    for worker in config["workers"]:
        for model in config["models"]:
            if model["name"] not in worker.get(
                "replicas", [m["name"] for m in config["models"]]
            ):
                continue
            for shape in model["shapes"]:
                scale = (shape / model["shapes"][0]) ** 2
                for size in config["batch_sizes"]:
                    latency = (
                        model["synthetic_base_ms"]
                        + model["synthetic_item_ms"] * size**0.85 * scale
                    ) / worker["speed"]
                    observed = latency * rng.lognormal(-(noise**2) / 2, noise, samples)
                    heldout = latency * rng.lognormal(
                        -(noise**2) / 2, noise, max(20, samples // 5)
                    )
                    entries.append(
                        summarize_samples(
                            model["name"],
                            worker["name"],
                            shape,
                            size,
                            observed,
                            heldout,
                            model["activation_mb"] * size * scale,
                            model["resident_mb"],
                        )
                    )
    write_json(
        output,
        {
            "metadata": {
                "source": "synthetic",
                "seed": seed,
                "noise": noise,
                "precision": "fp32",
                "samples_per_tuple": samples,
                "formula": "(base_ms + item_ms * batch_size**0.85 * shape_scale**2) / speed",
                "units": "milliseconds and MiB",
            },
            "entries": entries,
        },
    )
