import argparse
import json
from pathlib import Path

from slopack.io import load_config, read_json
from slopack.scheduler import ABLATIONS, POLICIES


def _config(parser):
    parser.add_argument("--config", required=True, type=Path)


def _execution(parser):
    _config(parser)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backend", choices=("simulation", "torch"), default="simulation"
    )
    parser.add_argument("--duration-s", type=float, default=20)
    parser.add_argument("--warmup-s", type=float, default=2)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data", type=Path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="SLOPack implementation: prepare, profile, schedule, evaluate, visualize"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser(
        "prepare-data", help="Download CIFAR-10 or create labeled synthetic images"
    )
    data.add_argument("--source", choices=("cifar10", "synthetic"), default="cifar10")
    data.add_argument("--output", required=True, type=Path)
    data.add_argument("--root", type=Path, default=Path("data/raw"))
    data.add_argument("--count", type=int, default=50000)
    data.add_argument("--seed", type=int, default=0)
    train = commands.add_parser(
        "train", help="Train an optional ten-class inference model"
    )
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument(
        "--architecture",
        choices=("resnet50", "efficientnet_b3", "mobilenet_v3_large", "vit_b_16"),
        default="mobilenet_v3_large",
    )
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--threads", type=int, default=1)
    profile = commands.add_parser(
        "profile", help="Build a synthetic or hardware-measured latency/memory table"
    )
    _config(profile)
    profile.add_argument("--output", required=True, type=Path)
    profile.add_argument(
        "--backend", choices=("synthetic", "torch"), default="synthetic"
    )
    profile.add_argument("--samples", type=int, default=200)
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--heldout", type=int, default=40)
    profile.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    profile.add_argument("--threads", type=int, default=1)
    profile.add_argument("--data", type=Path)
    profile.add_argument("--noise", type=float, default=0.04)
    profile.add_argument("--seed", type=int, default=0)
    trace = commands.add_parser(
        "generate-trace",
        help="Generate Poisson, periodic, or burst request descriptors",
    )
    _config(trace)
    trace.add_argument("--output", required=True, type=Path)
    trace.add_argument(
        "--kind", choices=("poisson", "periodic", "burst"), default="poisson"
    )
    trace.add_argument("--rate", type=float, default=200)
    trace.add_argument("--duration-s", type=float, default=20)
    trace.add_argument("--seed", type=int, default=0)
    trace.add_argument("--mixed-slo", action="store_true")
    trace.add_argument("--payloads", type=int, default=64)
    trace.add_argument("--period-ms", type=float, default=100)
    trace.add_argument("--burst-factor", type=float, default=4)
    trace.add_argument("--burst-start", type=float, default=0.4)
    trace.add_argument("--burst-end", type=float, default=0.6)
    run = commands.add_parser(
        "run", help="Replay descriptors through simulation or resident PyTorch workers"
    )
    _execution(run)
    run.add_argument("--trace", required=True, type=Path)
    run.add_argument("--policy", choices=POLICIES, default="slopack")
    run.add_argument("--ablation", choices=ABLATIONS, default="none")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--profile-bias", type=float, default=0)
    run.add_argument("--wait-cap-ms", type=float)
    run.add_argument("--guard-floor-ms", type=float)
    run.add_argument("--timeout-ms", type=float)
    experiment = commands.add_parser(
        "experiment",
        help="Run shared-trace comparisons, ablations, and parameter sweeps",
    )
    _execution(experiment)
    experiment.add_argument(
        "--study",
        choices=("compare", "ablation", "load", "pareto", "bias", "mixed", "all"),
        default="compare",
    )
    experiment.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    experiment.add_argument("--rate", type=float, default=200)
    experiment.add_argument(
        "--traces",
        nargs="+",
        choices=("poisson", "periodic", "burst"),
        default=["poisson", "periodic", "burst"],
    )
    experiment.add_argument(
        "--rates", nargs="+", type=float, default=[100, 150, 200, 250, 300]
    )
    experiment.add_argument("--reference-qps", type=float)
    experiment.add_argument(
        "--loads", nargs="+", type=float, default=[0.5, 0.75, 0.9, 0.95, 1.0, 1.1]
    )
    experiment.add_argument("--waits", nargs="+", type=float, default=[0.5, 2, 5, 10])
    experiment.add_argument("--guards", nargs="+", type=float, default=[0.2, 1, 2])
    experiment.add_argument(
        "--biases", nargs="+", type=float, default=[-0.1, -0.05, 0, 0.05, 0.1]
    )
    experiment.add_argument("--payloads", type=int, default=64)
    evaluate = commands.add_parser(
        "evaluate", help="Recompute metrics from a saved run's raw records"
    )
    evaluate.add_argument("--run", required=True, type=Path)
    plot = commands.add_parser(
        "plot", help="Plot run diagnostics or seed-aggregated comparisons"
    )
    plot.add_argument("--input", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    plot.add_argument("--kind", choices=("run", "experiment"), default="run")
    plot.add_argument("--bin-ms", type=float, default=1000)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        from slopack.io import prepare_data

        prepare_data(args.source, args.output, args.count, args.seed, args.root)
    elif args.command == "train":
        from slopack.training import train_classifier

        train_classifier(
            args.data,
            args.output,
            args.architecture,
            args.image_size,
            args.epochs,
            args.batch_size,
            args.lr,
            args.device,
            args.seed,
            args.max_steps,
            args.pretrained,
            args.threads,
        )
    elif args.command == "profile":
        config = load_config(args.config)
        if args.backend == "synthetic":
            from slopack.profiles import synthetic_profile

            synthetic_profile(config, args.output, args.samples, args.seed, args.noise)
        else:
            from slopack.profiling import torch_profile

            torch_profile(
                config,
                args.output,
                args.samples,
                args.warmup,
                args.heldout,
                args.precision,
                args.threads,
                args.data,
            )
    elif args.command == "generate-trace":
        from slopack.workload import generate_trace

        generate_trace(
            load_config(args.config),
            args.output,
            args.kind,
            args.rate,
            args.duration_s,
            args.seed,
            args.mixed_slo,
            args.payloads,
            args.period_ms,
            args.burst_factor,
            args.burst_start,
            args.burst_end,
        )
    elif args.command == "run":
        from slopack.workload import load_trace

        config = load_config(args.config)
        requests = load_trace(args.trace, config, args.duration_s * 1000)
        overrides = {
            key: value
            for key, value in {
                "wait_cap_ms": args.wait_cap_ms,
                "guard_floor_ms": args.guard_floor_ms,
                "timeout_ms": args.timeout_ms,
            }.items()
            if value is not None
        }
        metadata = {"trace_path": str(args.trace), "config_path": str(args.config)}
        sidecar = Path(str(args.trace) + ".json")
        if sidecar.exists():
            trace_meta = read_json(sidecar)
            metadata.update(
                {
                    "trace_kind": trace_meta.get("kind"),
                    "offered_qps": trace_meta.get("offered_qps"),
                    "trace_metadata": trace_meta,
                }
            )
        if args.backend == "simulation":
            from slopack.simulation import simulate

            result = simulate(
                config,
                args.profile,
                requests,
                args.output,
                args.duration_s * 1000,
                args.warmup_s * 1000,
                args.policy,
                args.ablation,
                args.seed,
                args.profile_bias,
                overrides,
                metadata,
            )
        else:
            if args.profile_bias:
                raise ValueError("--profile-bias is available in simulation only")
            from slopack.runtime import execute

            result = execute(
                config,
                args.profile,
                requests,
                args.output,
                args.duration_s * 1000,
                args.warmup_s * 1000,
                args.policy,
                args.ablation,
                args.seed,
                args.precision,
                args.threads,
                args.data,
                overrides,
                metadata,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "experiment":
        from slopack.experiments import experiment

        experiment(
            load_config(args.config),
            args.profile,
            args.output,
            args.study,
            args.backend,
            args.duration_s,
            args.warmup_s,
            args.seeds,
            args.rate,
            args.traces,
            args.rates,
            args.waits,
            args.guards,
            args.biases,
            args.precision,
            args.threads,
            args.data,
            args.payloads,
            args.reference_qps,
            args.loads,
        )
    elif args.command == "evaluate":
        from slopack.metrics import evaluate_saved

        print(json.dumps(evaluate_saved(args.run), indent=2, allow_nan=False))
    elif args.command == "plot":
        from slopack.plotting import plot_experiment, plot_run

        if args.kind == "run":
            plot_run(args.input, args.output, args.bin_ms)
        else:
            plot_experiment(args.input, args.output)


if __name__ == "__main__":
    main()
