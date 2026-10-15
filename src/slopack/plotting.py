from collections import defaultdict
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from slopack.io import read_csv, read_json, write_csv
from slopack.metrics import time_series
from slopack.reporting import GROUP_FIELDS, aggregate_rows, aggregate_summary


def _save(figure, output, name):
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.svg")
    plt.close(figure)


def plot_run(path, output, bin_ms=1000):
    path, output = (Path(path), Path(output))
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(path / "requests.csv")
    meta = read_json(path / "run.json")
    completed = [r for r in rows if r["completed_in_window"] == "True"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    latencies = sorted((float(r["latency_ms"]) for r in completed))
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
    per_model = read_csv(path / "per_model.csv")
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
    controller = read_csv(path / "controller.csv")
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
    errors = read_csv(path / "profile_errors.csv")
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
    batch_rows = read_csv(path / "batches.csv")
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


def plot_experiment(path, output, violation_limit=0.04):
    path, output = (Path(path), Path(output))
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = aggregate_summary(path)
    metrics = [key.removesuffix("_mean") for key in stats[0] if key.endswith("_mean")]
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
                axes[1].axhline(
                    100, color="#6a7480", linestyle=":", label="100 ms reference"
                )
                axes[1].legend(fontsize=8)
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
                axis.axhline(100 * violation_limit, color="#6a7480", linestyle=":")
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
                    labels = [
                        r["policy"]
                        + (":" + r["ablation"] if r["ablation"] != "none" else "")
                        for r in subset
                    ]
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
            if experiment in (
                "compare",
                "ablation",
                "mixed",
                "bias",
                "dispatch",
                "burst",
            ):
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


def plot_burst_comparison(path, output, bin_ms=1000):
    path, output = (Path(path), Path(output))
    runs = [
        row for row in read_csv(path / "summary.csv") if row["experiment"] == "burst"
    ]
    if not runs:
        return
    output.mkdir(parents=True, exist_ok=True)
    observations = []
    intervals = {}
    for run in runs:
        folder = Path(run["run"])
        metadata = read_json(folder / "run.json")
        timeline = time_series(
            read_csv(folder / "requests.csv"),
            read_csv(folder / "controller.csv"),
            metadata["warmup_ms"],
            metadata["duration_ms"],
            bin_ms,
        )
        for row in timeline:
            row.update({key: run[key] for key in ("backend", "policy", "ablation")})
            row["trace_qps"] = run["offered_qps"]
            observations.append(row)
        key = (run["backend"], run["offered_qps"])
        intervals[key] = [
            value * metadata["duration_ms"] / 1000
            for value in metadata["trace_metadata"]["burst_fraction"]
        ]
    stats = aggregate_rows(
        observations,
        ("backend", "trace_qps", "policy", "ablation", "start_ms", "end_ms"),
        (
            "offered_qps",
            "completed_qps",
            "p99_ms",
            "slo_violation_rate",
            "protection_fraction",
        ),
    )
    write_csv(output / "burst_timeline.csv", stats)
    colors = ("#3576a8", "#469d91", "#6a7480")
    for (backend, offered), interval in intervals.items():
        selected = [
            row
            for row in stats
            if row["backend"] == backend and row["trace_qps"] == offered
        ]
        figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        variants = sorted({(row["policy"], row["ablation"]) for row in selected})
        for color, (policy, ablation) in zip(colors, variants):
            points = sorted(
                [
                    row
                    for row in selected
                    if row["policy"] == policy and row["ablation"] == ablation
                ],
                key=lambda row: row["start_ms"],
            )
            times = [row["start_ms"] / 1000 for row in points]
            label = policy if ablation == "none" else f"{policy}:{ablation}"
            for axis, metric, scale in zip(
                axes[:2], ("p99_ms", "slo_violation_rate"), (1, 100)
            ):
                axis.plot(
                    times,
                    [
                        row[metric + "_mean"] * scale
                        if row[metric + "_mean"] is not None
                        else np.nan
                        for row in points
                    ],
                    label=label,
                    color=color,
                )
            if policy == "slopack" and ablation == "none":
                axes[2].plot(
                    times, [row["offered_qps_mean"] for row in points], color=color
                )
                for row in points:
                    duty = row["protection_fraction_mean"]
                    if duty > 0:
                        for axis in axes:
                            axis.axvspan(
                                row["start_ms"] / 1000,
                                row["end_ms"] / 1000,
                                color="#d3a44a",
                                alpha=0.3 * duty,
                                linewidth=0,
                            )
        for axis in axes:
            for boundary in interval:
                axis.axvline(boundary, color="#6a7480", linestyle=":", linewidth=1)
        axes[0].set(
            ylabel="P99 latency (ms)",
            title=f"Burst response ({backend}, {offered} offered QPS); seed means",
        )
        axes[0].legend(fontsize=8)
        axes[1].set(ylabel="SLO violations (%)")
        axes[2].set(xlabel="Time (s)", ylabel="Offered requests/s")
        _save(figure, output, f"fig5_burst_timeline_{backend}_{float(offered):g}")


def plot_profile_accuracy(rows, output, offline=False):
    if not rows:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    if offline:
        groups["offline",] = rows
    else:
        studies = {row["experiment"] for row in rows}
        study = "compare" if "compare" in studies else sorted(studies)[0]
        for row in rows:
            if row["experiment"] == study:
                groups[tuple((row[field] for field in GROUP_FIELDS))].append(row)
    for index, (key, points) in enumerate(groups.items()):
        points = sorted(points, key=lambda row: (row["model"], row["worker"]))
        labels = [f"{row['model']}/{row['worker']}" for row in points]
        figure, axes = plt.subplots(1, 2, figsize=(max(10, len(points) * 0.8), 5))
        for axis, metric, label, color in zip(
            axes,
            ("accuracy_pct", "p95_error_pct"),
            ("100 - MAPE (%)", "P95 absolute percentage error (%)"),
            ("#3576a8", "#469d91"),
        ):
            axis.bar(
                labels,
                [row[metric + "_mean"] for row in points],
                yerr=[row.get(metric + "_std") or 0 for row in points],
                color=color,
                capsize=3,
            )
            axis.set(ylabel=label)
            axis.tick_params(axis="x", rotation=65)
        mean = np.mean([row["accuracy_pct_mean"] for row in points])
        if offline:
            title = "Held-out offline latency profile"
            name = "fig6_profile_accuracy_offline"
        else:
            context = dict(zip(GROUP_FIELDS, key))
            title = f"Online profile: {context['experiment']}, {context['trace_kind']}, {context['backend']}"
            name = f"fig6_profile_accuracy_{context['experiment']}_{context['trace_kind']}_{index}"
        figure.suptitle(f"{title}; pair mean accuracy = {mean:.2f}%")
        _save(figure, output, name)
