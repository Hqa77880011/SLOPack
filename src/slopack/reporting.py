from collections import defaultdict
from pathlib import Path
import numpy as np
from slopack.io import read_csv, read_json, write_csv, write_json

GROUP_FIELDS = (
    "backend",
    "experiment",
    "trace_kind",
    "offered_qps",
    "policy",
    "ablation",
    "profile_bias",
    "bias_mode",
    "wait_cap_ms",
    "guard_floor_ms",
    "timeout_ms",
    "static_batch_size",
    "reference_qps",
    "offered_load",
)


def aggregate_rows(rows, group_fields, metric_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple((row.get(field, "") for field in group_fields))].append(row)
    result = []
    for key, runs in groups.items():
        row = dict(zip(group_fields, key))
        row["n_runs"] = len(runs)
        for field in metric_fields:
            values = [
                float(run[field]) for run in runs if run.get(field) not in (None, "")
            ]
            row[field + "_mean"] = float(np.mean(values)) if values else None
            row[field + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        result.append(row)
    return result


def aggregate_summary(path):
    rows = read_csv(Path(path) / "summary.csv")
    if not rows:
        raise ValueError("Experiment summary has no runs")
    fields = [
        field
        for field in rows[0]
        if field not in GROUP_FIELDS and field not in ("seed", "run")
    ]
    return (rows, aggregate_rows(rows, GROUP_FIELDS, fields))


def _write_table(path, rows):
    if rows:
        fields = list(dict.fromkeys((field for row in rows for field in row)))
        write_csv(path, rows, fields)


def export_profile(profile_path, output):
    profile = read_json(profile_path)
    entries = profile["entries"]
    rows = [
        {
            key: value
            for key, value in entry.items()
            if key not in ("samples_ms", "heldout_samples_ms")
        }
        for entry in entries
    ]
    _write_table(output / "profile_tuples.csv", rows)
    wide = {}
    for entry in entries:
        if entry["batch_size"] not in (1, 16):
            continue
        key = (entry["model"], entry["shape"])
        row = wide.setdefault(key, {"model": key[0], "shape": key[1]})
        column = f"{entry['worker']}_B{entry['batch_size']}_median_ms"
        row[column] = entry["median_ms"]
    _write_table(output / "table_profile_latency.csv", list(wide.values()))
    write_json(output / "profile_metadata.json", profile["metadata"])
    errors = defaultdict(list)
    for entry in entries:
        for observed in entry.get("heldout_samples_ms", []):
            errors[entry["model"], entry["worker"]].append(
                abs(observed - entry["median_ms"]) / observed
            )
    accuracy = [
        {
            "model": model,
            "worker": worker,
            "samples": len(values),
            "mape_pct_mean": 100 * float(np.mean(values)),
            "accuracy_pct_mean": 100 * (1 - float(np.mean(values))),
            "p95_error_pct_mean": 100 * float(np.percentile(values, 95)),
        }
        for (model, worker), values in sorted(errors.items())
    ]
    _write_table(output / "table_profile_accuracy_offline.csv", accuracy)
    return accuracy


def pareto_best(stats, limit):
    groups = defaultdict(list)
    for row in stats:
        violation = row["slo_violation_rate_mean"]
        if (
            row["experiment"] == "pareto"
            and violation is not None
            and (violation <= limit)
        ):
            groups[row["backend"], row["trace_kind"], row["offered_qps"]].append(row)
    result = []
    for points in groups.values():
        winners = []
        for policy in sorted({row["policy"] for row in points}):
            selected = [row for row in points if row["policy"] == policy]
            winner = max(
                selected,
                key=lambda row: (
                    row["completed_qps_mean"],
                    -row["slo_violation_rate_mean"],
                ),
            )
            winners.append(dict(winner, selection="policy_best", violation_limit=limit))
        result.extend(winners)
        baselines = [row for row in winners if row["policy"] != "slopack"]
        method = next((row for row in winners if row["policy"] == "slopack"), None)
        if baselines:
            best = max(
                baselines,
                key=lambda row: (
                    row["completed_qps_mean"],
                    -row["slo_violation_rate_mean"],
                ),
            )
            result.append(dict(best, selection="best_baseline"))
            if method and best["completed_qps_mean"] > 0:
                method["qps_gain_over_best_baseline_pct"] = 100 * (
                    method["completed_qps_mean"] / best["completed_qps_mean"] - 1
                )
    return result


def relative_comparisons(stats):
    groups = defaultdict(list)
    for row in stats:
        if row["experiment"] != "pareto":
            groups[
                tuple(
                    (
                        row[field]
                        for field in (
                            "backend",
                            "experiment",
                            "trace_kind",
                            "offered_qps",
                            "bias_mode",
                        )
                    )
                )
            ].append(row)
    result = []
    for points in groups.values():
        method = next(
            (
                row
                for row in points
                if row["policy"] == "slopack"
                and row["ablation"] == "none"
                and (float(row["profile_bias"]) == 0)
            ),
            None,
        )
        if method is None:
            continue
        for baseline in points:
            if baseline is method:
                continue
            row = {
                field: method[field]
                for field in (
                    "backend",
                    "experiment",
                    "trace_kind",
                    "offered_qps",
                    "bias_mode",
                )
            }
            row.update(
                {
                    "baseline_policy": baseline["policy"],
                    "baseline_ablation": baseline["ablation"],
                    "baseline_profile_bias": baseline["profile_bias"],
                }
            )
            for metric in ("slo_violation_rate", "p99_ms", "burst_peak_p99_ms"):
                reference = baseline.get(metric + "_mean")
                value = method.get(metric + "_mean")
                row[metric + "_reduction_pct"] = (
                    100 * (1 - value / reference)
                    if value is not None and reference is not None and (reference > 0)
                    else None
                )
            reference = baseline["completed_qps_mean"]
            row["completed_qps_gain_pct"] = (
                100 * (method["completed_qps_mean"] / reference - 1)
                if reference > 0
                else None
            )
            result.append(row)
    return result


def profile_accuracy(rows):
    observations = []
    for run in rows:
        if (
            run["policy"] != "slopack"
            or run["ablation"] != "none"
            or float(run["profile_bias"]) != 0
        ):
            continue
        for profile in read_csv(Path(run["run"]) / "profile_errors.csv"):
            row = {field: run.get(field, "") for field in GROUP_FIELDS}
            row.update(profile)
            observations.append(row)
    return aggregate_rows(
        observations,
        GROUP_FIELDS + ("model", "worker"),
        ("mape_pct", "p95_error_pct", "accuracy_pct"),
    )


def report_experiments(path, profile, output, violation_limit=0.04, bin_ms=1000):
    if not 0 <= violation_limit <= 1 or bin_ms <= 0:
        raise ValueError(
            "violation_limit must lie in [0,1] and bin_ms must be positive"
        )
    path, output = (Path(path), Path(output))
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = aggregate_summary(path)
    offline_accuracy = export_profile(profile, output)
    _write_table(output / "aggregate.csv", stats)
    table_names = {
        "compare": "table_main",
        "load": "table_load_scaling",
        "pareto": "table_operating_points",
        "burst": "table_burst",
        "mixed": "table_mixed_slo",
        "dispatch": "table_dispatch",
        "ablation": "table_ablation",
        "bias": "table_profile_bias",
    }
    for study, name in table_names.items():
        _write_table(
            output / f"{name}.csv", [row for row in stats if row["experiment"] == study]
        )
    if any((row["experiment"] == "pareto" for row in stats)):
        fields = list(stats[0]) + [
            "selection",
            "violation_limit",
            "qps_gain_over_best_baseline_pct",
        ]
        write_csv(
            output / "pareto_best.csv", pareto_best(stats, violation_limit), fields
        )
    _write_table(output / "relative_comparisons.csv", relative_comparisons(stats))
    accuracy = profile_accuracy(rows)
    _write_table(output / "table_profile_accuracy.csv", accuracy)
    from slopack.plotting import (
        plot_burst_comparison,
        plot_experiment,
        plot_profile_accuracy,
    )

    plot_experiment(path, output / "figures", violation_limit)
    plot_burst_comparison(path, output / "figures", bin_ms)
    plot_profile_accuracy(accuracy, output / "figures")
    plot_profile_accuracy(offline_accuracy, output / "figures", offline=True)
