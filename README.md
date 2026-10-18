# SLOPack

Implementation of **SLOPack: Slack-Aware Dynamic Batching and Heterogeneous-GPU Dispatch for Multi-Tenant Inference**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Input preparation

This workflow generates synthetic latency profiles and request arrivals for two virtual devices. These are simulator inputs, not measured GPU timings. The scheduler itself needs no model training. For the complete simulation workflow, skip the optional measured-input section.

```bash
slopack profile --config configs/reproduce.json --backend synthetic --samples 100 --seed 17 --output outputs/reproduce_profile.json
slopack generate-trace --config configs/reproduce.json --kind poisson --rate 100 --duration-s 20 --seed 17 --output data/poisson.csv
```

## Optional measured inputs and classifier training

Install `python -m pip install -e ".[torch]"` for PyTorch inference. CPU configuration uses MobileNetV3-Large; edit `configs/torch_gpu.json` for installed GPUs. Device, shapes, precision and thread count must match the measured profile. `prepare-data` downloads [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

```bash
slopack prepare-data --source cifar10 --output data/cifar10.npz
slopack profile --config configs/torch_cpu.json --backend torch --samples 200 --warmup 20 --heldout 40 --threads 1 --data data/cifar10.npz --output outputs/cpu_profile.json
slopack generate-trace --config configs/torch_cpu.json --kind poisson --rate 10 --duration-s 20 --seed 17 --output data/cpu.csv
slopack run --config configs/torch_cpu.json --profile outputs/cpu_profile.json --trace data/cpu.csv --backend torch --duration-s 20 --warmup-s 2 --threads 1 --data data/cifar10.npz --output outputs/cpu
```

Optional ten-class workload training: `slopack train --data data/cifar10.npz --architecture mobilenet_v3_large --image-size 224 --epochs 10 --batch-size 32 --seed 17 --device cpu --output checkpoints/cifar10`. Use `configs/trained_classifier.json` for that checkpoint and rebuild its profile.

## Execution and evaluation

Use the simulation command below after input preparation, or the measured backend command above after optional hardware preparation. Use the same config, profile and trace for paired policies. Duration and warm-up are in seconds; request timestamps/deadlines are in milliseconds.

```bash
slopack run --config configs/reproduce.json --profile outputs/reproduce_profile.json --trace data/poisson.csv --backend simulation --duration-s 20 --warmup-s 2 --policy slopack --output outputs/slopack
slopack evaluate --run outputs/slopack
slopack plot --kind run --input outputs/slopack --output outputs/slopack/figures
```

## Comparisons and ablations

```bash
slopack experiment --config configs/reproduce.json --profile outputs/reproduce_profile.json --backend simulation --study compare --duration-s 20 --warmup-s 2 --rate 100 --seeds 17 37 57 --output outputs/comparison
slopack report --input outputs/comparison --profile outputs/reproduce_profile.json --output outputs/comparison/report
slopack experiment --config configs/reproduce.json --profile outputs/reproduce_profile.json --backend simulation --study ablation --duration-s 20 --warmup-s 2 --rate 100 --seeds 17 37 57 --output outputs/ablation
slopack report --input outputs/ablation --profile outputs/reproduce_profile.json --output outputs/ablation/report
```

## Outputs

Runs contain `requests.csv`, `batches.csv`, `metrics.json`, `run.json`, `profiles_final.json` and controller/timeline records. Experiments write per-seed traces and run directories plus `summary.csv`/`experiment.json`. Reporting writes aggregate CSV tables and PNG/SVG figures. Optional classifier training writes `model.pt`, `history.csv` and `training.json`. Paths and seeds are explicit CLI/config inputs.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
