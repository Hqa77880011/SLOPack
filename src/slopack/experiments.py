from pathlib import Path

from slopack.io import read_json, write_csv, write_json
from slopack.metrics import aggregate_runs
from slopack.scheduler import ABLATIONS, POLICIES
from slopack.simulation import simulate
from slopack.workload import generate_trace, load_trace


def experiment(
    config,
    profile,
    output,
    study,
    backend,
    duration_s,
    warmup_s,
    seeds,
    rate,
    traces,
    rates,
    waits,
    guards,
    biases,
    precision,
    threads,
    payload_path,
    payloads,
    reference_qps=None,
    loads=None,
):
    if not 0 <= warmup_s < duration_s:
        raise ValueError("warmup must be shorter than duration")
    if backend == "torch" and study in ("bias", "all"):
        raise ValueError(
            "Profile-bias injection is available in simulation; select individual hardware studies"
        )
    if reference_qps is not None:
        if reference_qps <= 0 or not loads or min(loads) <= 0:
            raise ValueError(
                "Reference capacity and offered-load fractions must be positive"
            )
        rates = [reference_qps * value for value in loads]
    output = Path(output)
    studies = (
        [study]
        if study != "all"
        else ["compare", "ablation", "load", "pareto", "bias", "mixed"]
    )
    variants = []
    for current in studies:
        if current == "compare":
            variants += [
                (current, kind, rate, p, "none", 0, {}, False)
                for kind in traces
                for p in POLICIES
            ]
        elif current == "ablation":
            variants += [
                (current, kind, rate, "slopack", a, 0, {}, False)
                for kind in traces
                for a in ABLATIONS
            ]
        elif current == "load":
            variants += [
                (current, "poisson", r, p, "none", 0, {}, False)
                for r in rates
                for p in ("slopack", "dynamic_no_slack", "fixed_timeout")
            ]
        elif current == "pareto":
            variants += [
                (
                    current,
                    "poisson",
                    rate,
                    "slopack",
                    "none",
                    0,
                    {"wait_cap_ms": w, "guard_floor_ms": g},
                    False,
                )
                for w in waits
                for g in guards
            ]
            variants += [
                (
                    current,
                    "poisson",
                    rate,
                    "fixed_timeout",
                    "none",
                    0,
                    {"timeout_ms": w},
                    False,
                )
                for w in waits
            ]
        elif current == "bias":
            variants += [
                (current, "poisson", rate, "slopack", "none", b, {}, False)
                for b in biases
            ]
        elif current == "mixed":
            variants += [
                (current, "poisson", rate, p, "none", 0, {}, True)
                for p in ("slopack", "edf", "fixed_timeout", "dynamic_no_slack")
            ]
    paths = []
    trace_paths = {}
    for seed in seeds:
        for index, (
            current,
            kind,
            offered,
            policy,
            ablation,
            bias,
            overrides,
            mixed,
        ) in enumerate(variants):
            key = (seed, kind, offered, mixed)
            if key not in trace_paths:
                trace_path = (
                    output
                    / "traces"
                    / f"seed{seed}_{kind}_{offered:g}_{'mixed' if mixed else 'default'}.csv"
                )
                generate_trace(
                    config,
                    trace_path,
                    kind,
                    offered,
                    duration_s,
                    seed,
                    mixed,
                    payloads,
                    100,
                    4,
                    0.4,
                    0.6,
                )
                trace_paths[key] = trace_path
            requests = load_trace(trace_paths[key], config, duration_s * 1000)
            path = output / "runs" / f"{current}_{index:03d}_seed{seed}"
            metadata = {
                "experiment": current,
                "trace_kind": kind,
                "offered_qps": offered,
                "mixed_slo": mixed,
                "trace_path": str(trace_paths[key]),
                "trace_metadata": read_json(str(trace_paths[key]) + ".json"),
                "reference_qps": reference_qps,
                "offered_load": offered / reference_qps if reference_qps else None,
            }
            if backend == "simulation":
                metrics = simulate(
                    config,
                    profile,
                    requests,
                    path,
                    duration_s * 1000,
                    warmup_s * 1000,
                    policy,
                    ablation,
                    seed,
                    bias,
                    overrides,
                    metadata,
                )
            else:
                from slopack.runtime import execute

                metrics = execute(
                    config,
                    profile,
                    requests,
                    path,
                    duration_s * 1000,
                    warmup_s * 1000,
                    policy,
                    ablation,
                    seed,
                    precision,
                    threads,
                    payload_path,
                    overrides,
                    metadata,
                )
            paths.append(path)
            print(
                f"{path.name}: completed={metrics['completed']} violations={metrics['slo_violation_rate']} QPS={metrics['completed_qps']:.2f}",
                flush=True,
            )
    rows = aggregate_runs(paths, output / "summary.csv")
    for row, path in zip(rows, paths):
        row["experiment"] = read_json(path / "run.json")["experiment"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "summary.csv", rows, fields)
    write_json(
        output / "experiment.json",
        {
            "studies": studies,
            "backend": backend,
            "seeds": seeds,
            "duration_s": duration_s,
            "warmup_s": warmup_s,
            "runs": [str(path) for path in paths],
            "profile": str(profile),
        },
    )
    return rows
