import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from slopack.io import read_json, write_csv


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(figure, output, name):
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.svg")
    plt.close(figure)


def plot_run(path, output, bin_ms=1000):
    path, output = Path(path), Path(output)
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(path / "requests.csv")
    meta = read_json(path / "run.json")
    completed = [r for r in rows if r["completed_in_window"] == "True"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    latencies = sorted(float(r["latency_ms"]) for r in completed)
    if latencies:
        axes[0].plot(
            latencies,
            np.arange(1, len(latencies) + 1) / len(latencies),
            color="#3576a8",
        )
    axes[0].set(
        xlabel="Latency (ms)",
        ylabel="CDF",
        title=f"{meta['policy']} ({meta['backend']})",
    )
    per_model = _read_csv(path / "per_model.csv")
    axes[1].bar(
        [r["model"] for r in per_model],
        [
            float(r["deadline_success_share"])
            if r["deadline_success_share"]
            else np.nan
            for r in per_model
        ],
        color="#469d91",
    )
    axes[1].tick_params(axis="x", rotation=65)
    axes[1].set(
        ylabel="On-time completions / admissions",
        ylim=(0, 1),
        title="Cohort deadline success at window end",
    )
    _save(figure, output, "latency_fairness")
    bins = defaultdict(list)
    for row in completed:
        bins[int(float(row["completion_ms"]) // bin_ms)].append(row)
    times = sorted(bins)
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(
        [t * bin_ms / 1000 for t in times],
        [np.percentile([float(r["latency_ms"]) for r in bins[t]], 99) for t in times],
        color="#3576a8",
    )
    axes[0].set(ylabel="P99 latency (ms)")
    axes[1].plot(
        [t * bin_ms / 1000 for t in times],
        [100 * np.mean([int(r["violated"]) for r in bins[t]]) for t in times],
        color="#469d91",
    )
    axes[1].set(ylabel="SLO violations (%)")
    controller = _read_csv(path / "controller.csv")
    axes[2].plot(
        [float(r["time_ms"]) / 1000 for r in controller],
        [int(r["backlog"]) for r in controller],
        color="#6a7480",
    )
    axes[2].set(xlabel="Time (s)", ylabel="Outstanding requests")
    for axis in axes:
        for row in controller:
            if row["protection"] == "True":
                left = float(row["time_ms"]) / 1000
                width = meta["scheduler_settings"]["window_ms"] / 1000
                axis.axvspan(
                    left, left + width, color="#d3a44a", alpha=0.18, linewidth=0
                )
    _save(figure, output, "timeline")
    errors = _read_csv(path / "profile_errors.csv")
    if errors:
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(
            [r["model"] + "/" + r["worker"] for r in errors],
            [float(r["mape_pct"]) for r in errors],
            color="#3576a8",
        )
        axis.tick_params(axis="x", rotation=65)
        axis.set(
            ylabel="Runtime prediction MAPE (%)",
            title=f"Online profile error ({meta['backend']})",
        )
        _save(figure, output, "profile_error")
    batch_rows = _read_csv(path / "batches.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sizes, counts = np.unique(
        [int(r["batch_size"]) for r in batch_rows], return_counts=True
    )
    axes[0].bar(sizes.astype(str), counts, color="#3576a8")
    axes[0].set(xlabel="Batch size", ylabel="Batches", title="Executed batch sizes")
    metrics = read_json(path / "metrics.json")
    utilization = metrics["utilization"]
    axes[1].bar(list(utilization), list(utilization.values()), color="#469d91")
    axes[1].set(ylabel="Busy fraction", ylim=(0, 1), title="Worker utilization")
    _save(figure, output, "batching_dispatch")


def plot_experiment(path, output):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(path / "summary.csv")
    grouped = defaultdict(list)
    grouping = (
        "experiment",
        "trace_kind",
        "offered_qps",
        "policy",
        "ablation",
        "profile_bias",
        "wait_cap_ms",
        "guard_floor_ms",
        "timeout_ms",
    )
    for row in rows:
        grouped[tuple(row.get(k, "") for k in grouping)].append(row)
    stats = []
    metrics = (
        "slo_violation_rate",
        "p99_ms",
        "completed_qps",
        "on_time_qps",
        "jain_service",
        "jain_deadline_success",
        "profile_mape_pct",
        "profile_accuracy_pct",
        "profile_p95_error_pct",
        "scheduler_mean_us",
        "scheduler_p95_us",
        "peak_p99_ms",
        "jain_slo_class_service",
        "jain_slo_class_deadline_success",
    )
    metrics += tuple(
        key
        for key in rows[0]
        if key.startswith(("utilization_", "slo_")) and key not in metrics
    )
    for key, runs in grouped.items():
        stat = dict(zip(grouping, key))
        stat["n_runs"] = len(runs)
        for metric in metrics:
            values = [float(r[metric]) for r in runs if r.get(metric)]
            stat[metric + "_mean"] = float(np.mean(values)) if values else None
            stat[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        stats.append(stat)
    write_csv(output / "aggregate.csv", stats)
    for experiment in sorted({r.get("experiment", "compare") for r in rows}):
        selected = [r for r in stats if r.get("experiment", "compare") == experiment]
        for trace in sorted({r["trace_kind"] for r in selected}):
            subset = [r for r in selected if r["trace_kind"] == trace]
            if experiment == "load":
                figure, axes = plt.subplots(1, 3, figsize=(13, 4))
                for policy in sorted({r["policy"] for r in subset}):
                    points = sorted(
                        [r for r in subset if r["policy"] == policy],
                        key=lambda r: float(r["offered_qps"]),
                    )
                    reference = rows[0].get("reference_qps")
                    x = [
                        100 * float(r["offered_qps"]) / float(reference)
                        if reference
                        else float(r["offered_qps"])
                        for r in points
                    ]
                    for axis, metric, label, scale in zip(
                        axes,
                        ("slo_violation_rate", "p99_ms", "completed_qps"),
                        ("SLO violations (%)", "P99 latency (ms)", "Completed QPS"),
                        (100, 1, 1),
                    ):
                        axis.plot(
                            x,
                            [
                                r[metric + "_mean"] * scale
                                if r[metric + "_mean"] is not None
                                else np.nan
                                for r in points
                            ],
                            marker="o",
                            label=policy,
                        )
                        axis.set(
                            xlabel="Offered load (%)" if reference else "Offered QPS",
                            ylabel=label,
                        )
                axes[0].legend(fontsize=8)
            elif experiment == "pareto":
                figure, axis = plt.subplots(figsize=(7, 5))
                for policy in sorted({r["policy"] for r in subset}):
                    points = [
                        r
                        for r in subset
                        if r["policy"] == policy
                        and r["slo_violation_rate_mean"] is not None
                    ]
                    axis.scatter(
                        [r["completed_qps_mean"] for r in points],
                        [100 * r["slo_violation_rate_mean"] for r in points],
                        label=policy,
                    )
                axis.set(
                    xlabel="Completed QPS",
                    ylabel="SLO violations (%)",
                    title="Wait / guard operating points",
                )
                axis.legend()
            else:
                figure, axes = plt.subplots(
                    1, 3, figsize=(max(10, len(subset) * 1.2), 4)
                )
                if experiment == "bias":
                    subset.sort(key=lambda r: float(r["profile_bias"]))
                    labels = [f"{100 * float(r['profile_bias']):+g}%" for r in subset]
                elif experiment == "ablation":
                    labels = [r["ablation"] for r in subset]
                else:
                    labels = [r["policy"] for r in subset]
                for axis, metric, label, scale in zip(
                    axes,
                    ("slo_violation_rate", "p99_ms", "completed_qps"),
                    ("SLO violations (%)", "P99 latency (ms)", "Completed QPS"),
                    (100, 1, 1),
                ):
                    values = [
                        r[metric + "_mean"] * scale
                        if r[metric + "_mean"] is not None
                        else np.nan
                        for r in subset
                    ]
                    errors = [
                        r[metric + "_std"] * scale
                        if r[metric + "_std"] is not None
                        else 0
                        for r in subset
                    ]
                    axis.bar(labels, values, yerr=errors, color="#3576a8", capsize=3)
                    axis.set(ylabel=label)
                    axis.tick_params(axis="x", rotation=60)
            figure.suptitle(f"{experiment}: {trace} ({rows[0]['backend']}); seed means")
            _save(figure, output, f"{experiment}_{trace}")
            if experiment in ("compare", "ablation", "mixed", "bias"):
                figure, axes = plt.subplots(
                    1, 3, figsize=(max(10, len(subset) * 1.2), 4)
                )
                for axis, metric, label in zip(
                    axes,
                    (
                        "jain_deadline_success",
                        "profile_accuracy_pct",
                        "scheduler_mean_us",
                    ),
                    (
                        "Jain deadline success",
                        "Prediction accuracy (%)",
                        "Scheduler mean cost (us)",
                    ),
                ):
                    axis.bar(
                        labels,
                        [
                            r[metric + "_mean"]
                            if r[metric + "_mean"] is not None
                            else np.nan
                            for r in subset
                        ],
                        color="#469d91",
                    )
                    axis.set(ylabel=label)
                    axis.tick_params(axis="x", rotation=60)
                _save(figure, output, f"{experiment}_{trace}_diagnostics")
                workers = [key for key in metrics if key.startswith("utilization_")]
                figure, axis = plt.subplots(figsize=(max(8, len(subset)), 4))
                width = 0.8 / max(1, len(workers))
                x = np.arange(len(subset))
                for i, metric in enumerate(workers):
                    axis.bar(
                        x + (i - (len(workers) - 1) / 2) * width,
                        [
                            r[metric + "_mean"]
                            if r[metric + "_mean"] is not None
                            else np.nan
                            for r in subset
                        ],
                        width,
                        label=metric.removeprefix("utilization_"),
                    )
                axis.set_xticks(x, labels, rotation=60)
                axis.set(ylabel="Worker busy fraction", ylim=(0, 1))
                axis.legend()
                _save(figure, output, f"{experiment}_{trace}_utilization")
                if experiment == "mixed":
                    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                    for axis, metric, label in zip(
                        axes,
                        ("violation_rate", "p99_ms"),
                        ("SLO violations (%)", "P99 latency (ms)"),
                    ):
                        for i, slo_class in enumerate(("strict", "relaxed")):
                            field = f"slo_{slo_class}_{metric}_mean"
                            scale = 100 if metric == "violation_rate" else 1
                            axis.bar(
                                x + (i - 0.5) * 0.35,
                                [
                                    r[field] * scale
                                    if r.get(field) is not None
                                    else np.nan
                                    for r in subset
                                ],
                                0.35,
                                label=slo_class,
                            )
                        axis.set_xticks(x, labels, rotation=60)
                        axis.set(ylabel=label)
                        axis.legend()
                    _save(figure, output, "mixed_slo_classes")
