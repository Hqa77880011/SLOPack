from collections import defaultdict

import numpy as np

from slopack.io import read_json, write_csv, write_json


def percentile(values, quantile):
    return float(np.percentile(values, quantile)) if values else None


def jain(values):
    values = np.asarray(values, dtype=float)
    denominator = len(values) * float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator else None


def evaluate(requests, batches, start_ms, end_ms, worker_names, costs):
    if not 0 <= start_ms < end_ms:
        raise ValueError("Measurement interval must have positive duration")
    completion = {}
    for batch in batches:
        for request in batch.requests:
            completion[request.request_id] = batch.end_ms
    rows = []
    window_completions = []
    cohort = []
    for request in requests:
        end = completion.get(request.request_id)
        completed_in_window = end is not None and start_ms <= end < end_ms
        arrived_in_window = start_ms <= request.arrival_ms < end_ms
        row = {
            "request_id": request.request_id,
            "model": request.model,
            "shape": request.shape,
            "priority": request.priority,
            "slo_class": request.slo_class,
            "arrival_ms": request.arrival_ms,
            "deadline_ms": request.deadline_ms,
            "completion_ms": end,
            "latency_ms": end - request.arrival_ms if end is not None else None,
            "violated": int(end > request.deadline_ms) if end is not None else None,
            "arrived_in_window": arrived_in_window,
            "completed_in_window": completed_in_window,
        }
        rows.append(row)
        if completed_in_window:
            window_completions.append(row)
        if arrived_in_window:
            cohort.append(row)
    backlog_start = sum(
        r.arrival_ms < start_ms
        and (
            completion.get(r.request_id) is None or completion[r.request_id] >= start_ms
        )
        for r in requests
    )
    backlog_end = sum(
        completion.get(r.request_id) is None or completion[r.request_id] >= end_ms
        for r in requests
        if r.arrival_ms < end_ms
    )
    latencies = [row["latency_ms"] for row in window_completions]
    violation_rate = (
        sum(row["violated"] for row in window_completions) / len(window_completions)
        if window_completions
        else None
    )
    elapsed = end_ms - start_ms
    utilization = {}
    for worker in worker_names:
        busy = sum(
            max(0.0, min(batch.end_ms, end_ms) - max(batch.start_ms, start_ms))
            for batch in batches
            if batch.worker == worker
        )
        utilization[worker] = busy / elapsed
    per_model = []
    for model in sorted({r.model for r in requests}):
        admitted = [row for row in cohort if row["model"] == model]
        serviced = [
            row
            for row in admitted
            if row["completion_ms"] is not None and row["completion_ms"] < end_ms
        ]
        on_time = [row for row in serviced if not row["violated"]]
        per_model.append(
            {
                "model": model,
                "admitted": len(admitted),
                "completed_cohort": len(serviced),
                "on_time_cohort": len(on_time),
                "service_share": len(serviced) / len(admitted) if admitted else None,
                "deadline_success_share": len(on_time) / len(admitted)
                if admitted
                else None,
            }
        )
    error_values = [
        abs(b.observed_ms - b.prediction_ms) / b.observed_ms
        for b in batches
        if start_ms <= b.end_ms < end_ms
    ]
    latency_bins = defaultdict(list)
    for row in window_completions:
        latency_bins[int(row["completion_ms"] // 1000)].append(row["latency_ms"])
    failures = sum(
        (
            row["completion_ms"] is not None
            and row["completion_ms"] < end_ms
            and row["violated"]
        )
        or (
            row["deadline_ms"] < end_ms
            and (row["completion_ms"] is None or row["completion_ms"] >= end_ms)
        )
        for row in cohort
    )
    metrics = {
        "measurement_start_ms": start_ms,
        "measurement_end_ms": end_ms,
        "admitted": len(cohort),
        "completed": len(window_completions),
        "backlog_start": backlog_start,
        "backlog_end": backlog_end,
        "conservation_error": len(window_completions)
        - len(cohort)
        - backlog_start
        + backlog_end,
        "completed_qps": len(window_completions) / (elapsed / 1000),
        "slo_violation_rate": violation_rate,
        "cohort_deadline_failure_rate_at_end": failures / len(cohort)
        if cohort
        else None,
        "on_time_qps": sum(not row["violated"] for row in window_completions)
        / (elapsed / 1000),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "peak_p99_ms": max(
            (float(np.percentile(values, 99)) for values in latency_bins.values()),
            default=None,
        ),
        "peak_p99_bin_ms": 1000,
        "utilization": utilization,
        "jain_service": jain(
            [r["service_share"] for r in per_model if r["service_share"] is not None]
        ),
        "jain_deadline_success": jain(
            [
                r["deadline_success_share"]
                for r in per_model
                if r["deadline_success_share"] is not None
            ]
        ),
        "profile_mape_pct": 100 * float(np.mean(error_values))
        if error_values
        else None,
        "profile_accuracy_pct": 100 * (1 - float(np.mean(error_values)))
        if error_values
        else None,
        "profile_p95_error_pct": 100 * percentile(error_values, 95)
        if error_values
        else None,
        "scheduler_mean_us": float(np.mean(costs)) if costs else None,
        "scheduler_p95_us": percentile(costs, 95),
        "drained_cohort_p99_ms": percentile(
            [r["latency_ms"] for r in cohort if r["latency_ms"] is not None], 99
        ),
        "drained_cohort_violation_rate": float(
            np.mean([r["violated"] for r in cohort if r["violated"] is not None])
        )
        if cohort
        else None,
    }
    classes = []
    for label in sorted({r.slo_class for r in requests}):
        selected = [r for r in window_completions if r["slo_class"] == label]
        admitted = [r for r in cohort if r["slo_class"] == label]
        serviced = [
            r
            for r in admitted
            if r["completion_ms"] is not None and r["completion_ms"] < end_ms
        ]
        classes.append(
            {
                "slo_class": label,
                "completed": len(selected),
                "admitted": len(admitted),
                "completed_cohort": len(serviced),
                "service_share": len(serviced) / len(admitted) if admitted else None,
                "deadline_success_share": sum(not r["violated"] for r in serviced)
                / len(admitted)
                if admitted
                else None,
                "violation_rate": float(np.mean([r["violated"] for r in selected]))
                if selected
                else None,
                "p99_ms": percentile([r["latency_ms"] for r in selected], 99),
            }
        )
    metrics["jain_slo_class_service"] = jain(
        [r["service_share"] for r in classes if r["service_share"] is not None]
    )
    metrics["jain_slo_class_deadline_success"] = jain(
        [
            r["deadline_success_share"]
            for r in classes
            if r["deadline_success_share"] is not None
        ]
    )
    for row in classes:
        metrics[f"slo_{row['slo_class']}_violation_rate"] = row["violation_rate"]
        metrics[f"slo_{row['slo_class']}_p99_ms"] = row["p99_ms"]
    profile_groups = defaultdict(list)
    for batch in batches:
        if start_ms <= batch.end_ms < end_ms:
            profile_groups[(batch.model, batch.worker)].append(
                abs(batch.observed_ms - batch.prediction_ms) / batch.observed_ms
            )
    profile_rows = [
        {
            "model": model,
            "worker": worker,
            "mape_pct": 100 * float(np.mean(errors)),
            "p95_error_pct": 100 * float(np.percentile(errors, 95)),
            "accuracy_pct": 100 * (1 - float(np.mean(errors))),
        }
        for (model, worker), errors in sorted(profile_groups.items())
    ]
    return metrics, rows, per_model, classes, profile_rows


def save_run(
    output,
    requests,
    batches,
    scheduler,
    start_ms,
    end_ms,
    metadata,
    events,
    predictions=None,
):
    metrics, rows, models, classes, profile_rows = evaluate(
        requests,
        batches,
        start_ms,
        end_ms,
        scheduler.workers,
        [
            row["cost_us"]
            for row in scheduler.decision_costs
            if start_ms <= row["time_ms"] < end_ms
        ],
    )
    batch_rows = [
        {
            "batch_id": b.batch_id,
            "model": b.model,
            "shape": b.shape,
            "worker": b.worker,
            "batch_size": b.size,
            "request_ids": " ".join(str(r.request_id) for r in b.requests),
            "assigned_ms": b.assigned_ms,
            "start_ms": b.start_ms,
            "end_ms": b.end_ms,
            "prediction_ms": b.prediction_ms,
            "observed_ms": b.observed_ms,
        }
        for b in batches
    ]
    from pathlib import Path

    output = Path(output)
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", metadata)
    write_json(output / "profiles_final.json", scheduler.profiles.snapshot())
    write_csv(output / "requests.csv", rows)
    write_csv(output / "batches.csv", batch_rows)
    write_csv(output / "per_model.csv", models)
    write_csv(output / "slo_classes.csv", classes)
    write_csv(
        output / "profile_errors.csv",
        profile_rows,
        ["model", "worker", "mape_pct", "p95_error_pct", "accuracy_pct"],
    )
    write_csv(
        output / "controller.csv",
        events,
        ["time_ms", "backlog", "arrival_rate_per_ms", "protection"],
    )
    write_csv(
        output / "scheduler_costs.csv", scheduler.decision_costs, ["time_ms", "cost_us"]
    )
    if predictions is not None:
        write_csv(
            output / "predictions.csv",
            predictions,
            ["request_id", "model", "prediction"],
        )
    return metrics


def aggregate_runs(paths, output):
    rows = []
    for path in paths:
        from pathlib import Path

        path = Path(path)
        meta = read_json(path / "run.json")
        metrics = read_json(path / "metrics.json")
        row = {key: meta[key] for key in ("backend", "policy", "ablation", "seed")}
        row.update(
            {
                key: meta.get(key)
                for key in (
                    "trace_kind",
                    "offered_qps",
                    "profile_bias",
                    "wait_cap_ms",
                    "guard_floor_ms",
                    "timeout_ms",
                    "reference_qps",
                    "offered_load",
                )
            }
        )
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if not isinstance(value, dict)
            }
        )
        row.update(
            {
                f"utilization_{name}": value
                for name, value in metrics["utilization"].items()
            }
        )
        row["run"] = str(path)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output, rows, fields)
    return rows


def evaluate_saved(path):
    import csv
    from pathlib import Path

    from slopack.scheduler import Batch
    from slopack.workload import Request

    path = Path(path)
    with (path / "requests.csv").open(newline="", encoding="utf-8") as handle:
        requests = [
            Request(
                int(r["request_id"]),
                float(r["arrival_ms"]),
                float(r["deadline_ms"]),
                r["model"],
                int(r["shape"]),
                float(r["priority"]),
                slo_class=r["slo_class"],
            )
            for r in csv.DictReader(handle)
        ]
    by_id = {r.request_id: r for r in requests}
    with (path / "batches.csv").open(newline="", encoding="utf-8") as handle:
        batches = [
            Batch(
                int(r["batch_id"]),
                [by_id[int(i)] for i in r["request_ids"].split()],
                r["worker"],
                float(r["assigned_ms"]),
                float(r["prediction_ms"]),
                float(r["start_ms"]),
                float(r["end_ms"]),
                float(r["observed_ms"]),
            )
            for r in csv.DictReader(handle)
        ]
    meta = read_json(path / "run.json")
    with (path / "scheduler_costs.csv").open(newline="", encoding="utf-8") as handle:
        costs = [
            float(r["cost_us"])
            for r in csv.DictReader(handle)
            if meta["warmup_ms"] <= float(r["time_ms"]) < meta["duration_ms"]
        ]
    metrics, _, models, classes, errors = evaluate(
        requests,
        batches,
        meta["warmup_ms"],
        meta["duration_ms"],
        [w["name"] for w in meta["config"]["workers"]],
        costs,
    )
    write_json(path / "metrics.json", metrics)
    write_csv(path / "per_model.csv", models)
    write_csv(path / "slo_classes.csv", classes)
    write_csv(
        path / "profile_errors.csv",
        errors,
        ["model", "worker", "mape_pct", "p95_error_pct", "accuracy_pct"],
    )
    return metrics
