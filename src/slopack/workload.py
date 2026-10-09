import csv
from dataclasses import asdict, dataclass

import numpy as np

from slopack.io import write_csv, write_json


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_ms: float
    deadline_ms: float
    model: str
    shape: int
    priority: float
    payload_id: int = 0
    slo_class: str = "default"

    @property
    def queue_key(self):
        return self.model, self.shape


def load_trace(path, config, duration_ms):
    shapes = {m["name"]: m["shapes"] for m in config["models"]}
    requests = []
    ids = set()
    previous = -1.0
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            request = Request(
                request_id=int(row["request_id"]),
                arrival_ms=float(row["arrival_ms"]),
                deadline_ms=float(row["deadline_ms"]),
                model=row["model"],
                shape=int(row["shape"]),
                priority=float(row.get("priority", 0.5)),
                payload_id=int(row.get("payload_id", 0)),
                slo_class=row.get("slo_class", "default"),
            )
            times = [request.arrival_ms, request.deadline_ms, request.priority]
            if not all(np.isfinite(times)):
                raise ValueError("Non-finite trace value")
            if request.arrival_ms < previous or request.arrival_ms < 0:
                raise ValueError("Trace arrivals must be nonnegative and sorted")
            previous = request.arrival_ms
            if request.request_id in ids or request.payload_id < 0:
                raise ValueError("Duplicate request_id or negative payload_id")
            ids.add(request.request_id)
            if (
                request.model not in shapes
                or request.shape not in shapes[request.model]
            ):
                raise ValueError(
                    f"Unsupported request model/shape: {request.queue_key}"
                )
            if (
                request.deadline_ms <= request.arrival_ms
                or not 0 <= request.priority <= 1
            ):
                raise ValueError(
                    "Deadline must follow arrival; priority must lie in [0, 1]"
                )
            if request.arrival_ms < duration_ms:
                requests.append(request)
    if not requests:
        raise ValueError("Trace has no arrivals in the configured duration")
    return requests


def generate_trace(
    config,
    output,
    kind,
    rate,
    duration_s,
    seed,
    mixed_slo,
    payloads,
    period_ms,
    burst_factor,
    burst_start,
    burst_end,
    warmup_ms=0.0,
):
    if rate <= 0 or duration_s <= 0 or payloads < 1 or period_ms <= 0:
        raise ValueError("Rate, duration, payload count, and period must be positive")
    if burst_factor < 1 or not 0 <= burst_start < burst_end <= 1:
        raise ValueError("Invalid burst interval or factor")
    if not 0 <= warmup_ms < duration_s * 1000:
        raise ValueError("warmup must be nonnegative and shorter than duration")
    rng = np.random.default_rng(seed)
    duration = duration_s * 1000
    arrivals = []
    if kind == "periodic":
        count = rate * period_ms / 1000
        for time in np.arange(0, duration, period_ms):
            n = int(count) + int(rng.random() < count % 1)
            arrivals.extend([float(time)] * n)
    else:
        if kind == "burst":
            bounds = [0, duration * burst_start, duration * burst_end, duration]
            average = 1 + (burst_end - burst_start) * (burst_factor - 1)
            rates = [rate / average, rate * burst_factor / average, rate / average]
        else:
            bounds, rates = [0, duration], [rate]
        for left, right, local_rate in zip(bounds[:-1], bounds[1:], rates):
            time = left
            while True:
                time += rng.exponential(1000 / local_rate)
                if time >= right:
                    break
                arrivals.append(float(time))
    if not arrivals:
        raise ValueError("No arrivals generated; increase rate or duration")
    models = config["models"]
    weights = np.asarray([m.get("weight", 1) for m in models], dtype=float)
    weights /= weights.sum()
    rows = []
    strict_labels = np.zeros(len(arrivals), dtype=bool)
    if mixed_slo:
        split = int(np.searchsorted(arrivals, warmup_ms))
        for left, right in ((0, split), (split, len(arrivals))):
            labels = strict_labels[left:right]
            labels[: len(labels) // 2] = True
            rng.shuffle(labels)
    for index, arrival in enumerate(arrivals):
        model = models[int(rng.choice(len(models), p=weights))]
        strict = bool(strict_labels[index])
        slo_class = ("strict" if strict else "relaxed") if mixed_slo else "default"
        factor = (0.6 if strict else 1.4) if mixed_slo else 1.0
        request = Request(
            index,
            arrival,
            arrival + model["slo_ms"] * factor,
            model["name"],
            int(rng.choice(model["shapes"])),
            model.get("priority", 0.5),
            int(rng.integers(payloads)),
            slo_class,
        )
        rows.append(asdict(request))
    write_csv(output, rows)
    write_json(
        str(output) + ".json",
        {
            "source": "generated",
            "kind": kind,
            "seed": seed,
            "offered_qps": rate,
            "duration_s": duration_s,
            "mixed_slo": mixed_slo,
            "warmup_ms": warmup_ms,
            "payloads": payloads,
            "period_ms": period_ms,
            "burst_factor": burst_factor,
            "burst_fraction": [burst_start, burst_end],
        },
    )
