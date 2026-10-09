# SLOPack

SLOPack implements **Slack-Aware Dynamic Batching and Heterogeneous-GPU Dispatch for Multi-Tenant Inference**. It coordinates how long a compatible request queue waits and which resident model replica executes its batch. The scheduler uses predicted completion slack, arrival-rate estimates, worker backlog, online latency correction, queue aging, and burst protection.

The package provides an event-driven simulator and a PyTorch execution backend with one serial worker process per device. Both use the same scheduling code. The simulator runs from a latency/memory table; the PyTorch backend executes the configured models and records measured completion times.

## Install

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
slopack --help
```

The simulator needs only NumPy and Matplotlib. For profiling, training, and PyTorch execution, install a matching PyTorch/TorchVision pair for your CPU or CUDA environment using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e ".[torch]"
```

The six-model GPU configuration also needs YOLOv5 dependencies:

```bash
python -m pip install -e ".[torch,yolo]"
```

YOLOv5m is loaded from the pinned `ultralytics/yolov5` revision `f9578796b8a64889aa9e66dc06914d19f40d338e`, which supports current PyTorch checkpoint loading. Its first use downloads the source and, when requested, pretrained weights. TorchVision and BERT also download their configured weights on first use. Keep the PyTorch, TorchVision, Transformers, and model-source versions fixed when comparing runs.

## 1. Prepare inputs

The scheduler consumes request descriptors, so a simulation does not require an image dataset. The following command creates a seeded trace:

```bash
slopack generate-trace --config configs/simulation.json --kind poisson --rate 200 --duration-s 20 --seed 0 --output data/poisson.csv
```

`--rate` is aggregate offered requests/s. Model traffic shares follow each model's `weight`. Supported patterns are:

- `poisson`: independent exponential interarrival times.
- `periodic`: requests arrive together every `--period-ms` (default 100 ms); counts preserve the requested mean rate in expectation.
- `burst`: a Poisson trace with a higher rate between `--burst-start` and `--burst-end`, expressed as fractions of the duration. `--burst-factor` defaults to 4; rates outside the burst are adjusted to preserve the requested mean offered rate.

`--mixed-slo` assigns equal numbers of requests to strict and relaxed classes, multiplying the model's SLO by 0.6 and 1.4. Labels are shuffled separately within the warm-up and measurement intervals; odd counts differ by one request. Use `--warmup-s` to match the subsequent run. `--payloads` sets the range of payload IDs, defaulting to 64. A JSON sidecar records generation parameters.

Existing traffic can be supplied as a sorted CSV with this header:

```text
request_id,arrival_ms,deadline_ms,model,shape,priority,payload_id,slo_class
```

Times are milliseconds from the start of replay; deadlines are absolute. IDs must be unique, arrivals nonnegative and sorted, deadlines later than arrivals, and priorities in `[0,1]`. Model names and shape buckets must match the config. `shape` is the image side length or BERT sequence length. Requests with the same `(model, shape)` enter the same FIFO queue. `payload_id` and `slo_class` are optional, with defaults `0` and `default`.

For image execution or optional classifier training, download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):

```bash
slopack prepare-data --source cifar10 --count 50000 --output data/cifar10.npz
```

The preparation command uses TorchVision's train and test splits. It saves uint8 RGB arrays and labels as `images`, `labels`, `val_images`, and `val_labels`. Execution uses the validation images as a reusable payload bank. Set `--payloads` no higher than that bank's size when generating a hardware trace. Images are resized to the configured shape and normalized with ImageNet mean/std for classifiers; detector inputs remain in `[0,1]`.

A small generated dataset is available for trying the input and training interfaces without downloads:

```bash
slopack prepare-data --source synthetic --count 128 --seed 0 --output data/synthetic.npz
```

Its labels encode colored stripe patterns. It is illustrative input data and should not be used to interpret model accuracy on natural images. Without `--data`, the PyTorch backend uses 64 deterministic generated RGB tensors. BERT uses deterministic valid token IDs unless the NPZ additionally contains a `tokens` array of shape `[N, sequence_length]`, with IDs in `[0,30522)`. Real tokenized data should use the vocabulary of the configured BERT model; zero is treated as padding. These inputs support scheduling measurements; BERT predictions are pooled-vector norms, and detector predictions are maximum objectness scores before NMS.

## 2. Optional model training

SLOPack is a scheduling method with no trainable scheduler parameters. Pretrained resident inference models can be used directly. To train a ten-class classifier as the inference workload:

```bash
slopack train --data data/cifar10.npz --architecture mobilenet_v3_large --image-size 224 --epochs 10 --batch-size 32 --lr 0.001 --device cuda:0 --seed 0 --output checkpoints/cifar10
```

Training supports ResNet-50, EfficientNet-B3, MobileNetV3-Large, and ViT-B/16, using cross entropy and AdamW. `--pretrained` initializes the backbone with TorchVision weights. `--max-steps` can limit optimization steps; its default `0` means no step limit. The saved outputs are `model.pt`, `history.csv` (loss and validation accuracy), and `training.json`.

Use `configs/trained_classifier.json` for this checkpoint. If the architecture, image size, or output path changes, update the config accordingly. Rebuild the latency profile after changing weights, shapes, precision, devices, or CPU thread count. Classifier accuracy is a training diagnostic; it is separate from the paper's scheduling metrics.

## 3. Build the latency and memory profile

For a hardware-free workflow:

```bash
slopack profile --config configs/simulation.json --backend synthetic --samples 200 --noise 0.04 --seed 0 --output outputs/synthetic_profile.json
```

This creates a **synthetic** profile using the configured model costs and two virtual devices. The costs are illustrative, not measurements of T4/A100 and not the paper's result table. Each tuple has positive lognormal latency samples, a central estimate, held-out relative error, a guard residual, and a memory estimate. Simulation draws observed runtimes from this fixed sample bank; online EMA changes predictions without changing the underlying runtimes.

For a CPU profile:

```bash
slopack profile --config configs/torch_cpu.json --backend torch --samples 200 --warmup 20 --heldout 40 --threads 1 --output outputs/cpu_profile.json
```

`configs/torch_cpu.json` uses a randomly initialized MobileNetV3-Large at 32×32 to keep the example manageable. For the paper's model set, edit `configs/torch_gpu.json` to match the GPUs actually installed, then:

```bash
slopack profile --config configs/torch_gpu.json --backend torch --samples 200 --warmup 20 --heldout 40 --precision fp32 --threads 1 --output outputs/gpu_profile.json
```

The six adapters are [TorchVision](https://docs.pytorch.org/vision/stable/models.html) ResNet-50, EfficientNet-B3, MobileNetV3-Large, and ViT-B/16; [Transformers BERT-base](https://huggingface.co/docs/transformers/model_doc/bert); and [YOLOv5m](https://docs.ultralytics.com/yolov5/tutorials/pytorch-hub-model-loading). No request changes model or precision during scheduling. ViT replicas require one fixed image size.

`--samples` counts timed executions per `(model, worker, shape, batch_size)`; `--heldout` uses separate executions to estimate P95 relative errors and positive residuals. Both sample sets are saved, allowing model/device accuracy to be computed without resampling. CUDA events provide GPU timing summaries, while the scheduling profile uses host forward duration through CUDA synchronization. Profiles exclude preprocessing and host-to-device transfer. CUDA memory records incremental peak allocated memory plus input tensors; resident model memory is accounted for separately. Out-of-memory tuples are omitted and listed in profile metadata. CPU memory admission uses an input-size estimate rather than a measured allocator peak.

## 4. Run the method or a baseline

Replay the prepared trace:

```bash
slopack run --config configs/simulation.json --profile outputs/synthetic_profile.json --trace data/poisson.csv --backend simulation --duration-s 20 --warmup-s 2 --policy slopack --output outputs/slopack
slopack evaluate --run outputs/slopack
slopack plot --kind run --input outputs/slopack --output outputs/slopack/figures --bin-ms 1000
```

The warm-up occupies the start of the trace; metrics use `[warmup, duration)`. Arrivals after the duration are excluded. Requests admitted before the endpoint are drained afterward so the saved records also support full-cohort latency analysis. The controller does not learn that the trace has ended before that endpoint. Static batches flush their final partial prefixes at the endpoint.

For actual inference, create a matching trace and use the measured profile:

```bash
slopack generate-trace --config configs/torch_cpu.json --kind poisson --rate 10 --duration-s 20 --output data/cpu.csv
slopack run --config configs/torch_cpu.json --profile outputs/cpu_profile.json --trace data/cpu.csv --backend torch --duration-s 20 --warmup-s 2 --threads 1 --output outputs/cpu
```

For GPUs, substitute the GPU config and profile. `--data data/cifar10.npz` supplies image payloads. `--precision` and `--threads` must match the profile. Worker startup and model warm-up finish before the replay clock starts. Workers keep replicas resident and execute whole batches serially. Request tensors reside in shared CPU memory; dispatch passes descriptors and gathers/resizes inputs in the worker. Actual request latency includes controller delay, IPC, preprocessing, transfer, queueing, and model execution. Model loading and cold starts are outside the replay interval.

The `--policy` choices define concrete baselines:

| Policy | Queue selection | Batch formation | Worker selection |
| --- | --- | --- | --- |
| `slopack` | Completion-slack urgency, priority, aging | Arrival-aware guarded slack | Violation count, maximum tardiness, completion time |
| `fcfs` | Earliest FIFO head arrival | Singleton | Same completion-risk dispatcher |
| `static` | Earliest FIFO head arrival | Fixed supported size; flush at endpoint | Same completion-risk dispatcher |
| `fixed_timeout` | Earliest FIFO head arrival | Maximum feasible size or fixed timeout | Same completion-risk dispatcher |
| `edf` | Earliest FIFO head deadline | Singleton | Same completion-risk dispatcher |
| `sjf` | Shortest profiled singleton runtime | Singleton | Same completion-risk dispatcher |
| `queue_length` | Earliest FIFO head arrival | Arrival-aware growth without slack | Fewest assigned requests, then completion time |
| `fastest_gpu` | Earliest FIFO head arrival | Arrival-aware growth without slack | Lowest runtime, ignoring backlog for placement |
| `dynamic_no_slack` | Earliest FIFO head arrival | Arrival-aware growth bounded by timeout | Same completion-risk dispatcher |

Only same-model, same-shape FIFO prefixes may be batched. All policies use the same memory limits, resident replicas, precision, traces, and initial profiles. Pending requests are retained; there are no admission drops. Baselines needing runtimes receive the same profile and EMA feedback.

## 5. Comparisons, ablations, and sweeps

Each study generates a trace once for each seed/scenario and reuses it for every policy or variant. Runs start with fresh scheduler and profile state.

```bash
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study compare --duration-s 20 --warmup-s 2 --rate 200 --seeds 0 1 2 3 4 --output outputs/comparison
slopack plot --kind experiment --input outputs/comparison --output outputs/comparison/figures
```

The same command accepts `--backend torch` and a matching measured profile for hardware comparisons. Select an individual study:

| Study | Experiment |
| --- | --- |
| `compare` | Nine policies on Poisson, periodic, and burst traces |
| `ablation` | Full method and `no_arrival`, `no_hetero`, `no_queue`, `no_aging`, `no_burst` |
| `load` | Offered-rate sweep with SLOPack, dynamic batching without slack, and fixed timeout |
| `pareto` | SLOPack wait/guard sweep, timeout sweeps, static batch-size sweep, and all eight baselines |
| `bias` | Latency-profile bias at −10%, −5%, 0%, +5%, and +10%, with initial or sustained injection |
| `mixed` | Strict/relaxed SLO classes with four policies |
| `dispatch` | SLOPack, queue-length dispatch, and fastest-GPU dispatch on the selected traces |
| `burst` | Burst timeline comparison of the full method, disabled protection, and dynamic batching without slack |
| `all` | All eight studies, available on either execution backend |

For example:

```bash
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study ablation --traces burst --duration-s 20 --warmup-s 2 --rate 200 --seeds 0 1 2 3 4 --output outputs/ablation
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study load --rates 100 150 200 250 300 --output outputs/load
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study pareto --waits 0.5 2 5 10 --guards 0.2 1 2 --output outputs/pareto
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study bias --biases -0.1 -0.05 0 0.05 0.1 --output outputs/bias
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study mixed --output outputs/mixed
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study dispatch --traces poisson --output outputs/dispatch
slopack experiment --config configs/simulation.json --profile outputs/synthetic_profile.json --study burst --output outputs/burst
```

`no_arrival` waits for the bounded guarded slack without estimating fill time. `no_hetero` pins each candidate to its fastest feasible replica. `no_queue` removes backlog from predictions while workers still execute queued work. `no_aging` sets the aging coefficient to zero. `no_burst` disables protection.

Profile-bias sweeps change runtime predictions while leaving executed work and observed timings unchanged. The default `--bias-mode sustained` multiplies the initial estimate and each EMA observation by `1 + bias`, keeping the error injection active through measurement. `--bias-mode initial` biases only the initial estimate and lets EMA correct it. Both modes work with simulation and hardware execution; `run --profile-bias` applies a single bias. Trace controls `--period-ms`, `--burst-factor`, `--burst-start`, and `--burst-end` are also available on `experiment`.

The paper uses 1,200 seconds of measurement after 120 seconds of warm-up, so the corresponding CLI durations are `--duration-s 1320 --warmup-s 120`, with five seeds. Rate sweeps use requests/s; `--reference-qps` and `--loads` express percentage offered load relative to a chosen reference capacity. The following complete study uses 250 QPS as an example reference; set that value and `--rate` for the workload and hardware being compared. The paper does not define how its reference capacity was determined.

```bash
slopack experiment --config configs/torch_gpu.json --profile outputs/gpu_profile.json --backend torch --study all --duration-s 1320 --warmup-s 120 --rate 200 --reference-qps 250 --loads 0.5 0.75 0.9 0.95 1.0 1.1 --seeds 0 1 2 3 4 --data data/cifar10.npz --output outputs/paper
slopack report --input outputs/paper --profile outputs/gpu_profile.json --output outputs/paper/report --violation-limit 0.04 --bin-ms 1000
```

`report` reads saved records and the profile, exports arithmetic seed means and sample standard deviations, and creates PNG/SVG figures. `--violation-limit` is a fraction: `0.04` selects operating points with mean violation rate at most 4%. `--bin-ms` controls burst-comparison timeline resolution. Keep run directories at their recorded paths and invoke reporting from the repository root.

| Paper experiment | Study / source | Report output |
| --- | --- | --- |
| Model/device median latency, B1 and B16 | `profile` | `table_profile_latency.csv`, `profile_tuples.csv` |
| Poisson, periodic, burst policy comparison | `compare` | `table_main.csv`, `figures/compare_*.png` |
| Load scaling, including 95% offered load | `load` with `--reference-qps` | `table_load_scaling.csv`, `figures/load_poisson.png` |
| Throughput/violation trade-off | `pareto` | `table_operating_points.csv`, `pareto_best.csv`, `figures/pareto_poisson.png` |
| Burst response and protection mode | `burst` | `table_burst.csv`, `figures/burst_timeline.csv`, `figures/fig5_burst_timeline_*.png` |
| Strict/relaxed SLOs and Jain fairness | `mixed` | `table_mixed_slo.csv`, `figures/mixed_slo_classes.png` |
| Heterogeneous dispatch and per-GPU utilization | `dispatch` | `table_dispatch.csv`, `figures/dispatch_*_utilization.png` |
| Model/device prediction accuracy and P95 error | Run batch records and held-out profile samples | `table_profile_accuracy.csv`, `table_profile_accuracy_offline.csv`, `figures/fig6_profile_accuracy_*.png` |
| Five component ablations and decision cost | `ablation` | `table_ablation.csv`, `figures/ablation_*_diagnostics.png` |
| Execution-time profile bias sensitivity | `bias` | `table_profile_bias.csv`, `figures/bias_poisson.png` |

All report tables retain P50/P95/P99, completed and on-time QPS, endpoint backlogs, per-worker utilization, model/class Jain indices, profile errors, and mean/P95 scheduling cost when applicable. `relative_comparisons.csv` computes violation and P99 reductions and QPS gains against each comparison variant. `pareto_best.csv` contains each policy's highest-throughput feasible point, the best feasible baseline, and SLOPack's relative QPS gain. If no point satisfies the constraint, the table has a header and no selected rows.

## Configuration

Configs contain `models`, `workers`, `batch_sizes`, and `scheduler`. A model specifies its architecture, shape buckets, SLO, traffic weight, priority, weights, and optionally a checkpoint. Each worker specifies its name, device, memory budget in MiB, and optionally a `replicas` list. Omitted `replicas` means all configured models. Worker names index the profile, so rebuild profiles when changing them. CUDA device indices and memory budgets must match the installed hardware.

Scheduler parameters use milliseconds and requests/ms:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rho`, `eta` | 0.15, 0.2 | Execution-latency and arrival-rate EMA weights |
| `window_ms`, `burst_reference_ms` | 10, 200 | Arrival-count window and moving reference |
| `alpha`, `beta`, `gamma` | 1, 0.25, 0.02 | Slack urgency, user priority, and aging weights |
| `epsilon` | 1e-6 | Arrival-rate floor |
| `wait_cap_ms` | 5 | Maximum total controlled wait for one queue head |
| `guard_floor_ms` | 0.2 | Added to P95 positive latency residuals |
| `guard_update_batches` | 100 | Per-tuple residual-guard update interval |
| `stale_window`, `stale_factor` | 100, 2 | Disable controlled growth when recent P95 relative error exceeds twice offline P95 |
| `burst_enter_ratio`, `burst_exit_ratio` | 1.25, 1.05 | Burst entry and exit thresholds |
| `burst_enter_windows`, `burst_exit_windows` | 3, 5 | Consecutive windows for switching protection |
| `burst_wait_ms`, `burst_batch_cap` | 0.5, 8 | Protection wait limit and low-priority batch cap |
| `low_priority_threshold` | 0.5 | Priorities below this value receive the burst cap |
| `memory_reserve_mb` | 256 | Reserve beyond resident parameters and batch allocation |
| `static_batch_size`, `timeout_ms` | 8, 5 | Static and timeout baseline controls |

The GPU example reserves 512 MiB; the CPU example reserves 64 MiB. `--wait-cap-ms`, `--guard-floor-ms`, and `--timeout-ms` override those controls for a single run. Optional `energy_mode` uses configured worker `power_w` as a final tie-breaker after deadline risk and completion time; it is a power-based estimate rather than a measured energy metric.

## Outputs and metrics

Each run writes the following files under its output directory:

| File | Contents |
| --- | --- |
| `run.json` | Backend, config, effective settings, profile source, trace metadata, and timing conventions |
| `requests.csv` | Arrivals, deadlines, actual completion, latency, violation, and window membership |
| `batches.csv` | FIFO request IDs, placement, assignment/start/end times, predicted and observed runtimes |
| `metrics.json` | Window metrics, endpoint backlog, fairness, profile error, and decision cost |
| `per_model.csv`, `slo_classes.csv` | Demand-normalized service and strict/relaxed SLO breakdowns |
| `profile_errors.csv` | Model/worker MAPE, P95 relative error, and `100 − MAPE` accuracy |
| `profiles_final.json` | Online estimates, residual guards, observation counts, and stale flags |
| `controller.csv` | Window-level arrival estimates, outstanding requests, protection state |
| `timeline.csv` | Measurement-window bins: offered/completed QPS, P99, violations, and protection fraction |
| `scheduler_costs.csv` | Decision timestamps and durations in microseconds |
| `predictions.csv`, `cuda_times.csv` | PyTorch prediction summaries and CUDA-event timings |

`evaluate` recomputes metrics from the saved request/batch records. `plot` writes PNG and editable SVG figures for latency CDF, deadline success, binned P99/violations, burst-protection intervals, batch-size distribution, worker utilization, and profile error. Experiment output adds `summary.csv` with one row per run and an `aggregate.csv` from plotting with arithmetic seed means and sample standard deviations. A single seed has no standard-deviation estimate. Comparison plots include fairness, prediction accuracy, scheduling cost, device utilization, and strict/relaxed SLO breakdowns. Load and operating-point plots are computed from the same records.

Metric definitions are explicit:

- **SLO violation rate:** completed requests in the measurement window whose actual completion exceeds their individual deadline, divided by all window completions.
- **P50/P95/P99:** arrival-to-completion latency of window completions. `peak_p99_ms` is the maximum among populated 1-second completion bins across the measurement window. `burst_peak_p99_ms` applies the same calculation only inside the generated arrival-spike interval, clipped to measurement; bins begin at that interval's start.
- **Completed QPS / on-time QPS:** window completions / deadline successes divided by window duration. Warm-up arrivals may complete in this interval.
- **Backlog:** outstanding admitted requests at either boundary. `completed = admitted + backlog_start − backlog_end`; `conservation_error` reports this equality's residual.
- **Cohort deadline failure at endpoint:** late completed requests plus overdue unfinished requests from the measurement arrival cohort, divided by that cohort's admissions. Non-overdue unfinished requests remain pending.
- **Utilization:** worker forward-execution busy intervals clipped to the measurement window, divided by its duration. CPU workers report busy fraction with the same definition.
- **Jain fairness:** `(sum x)^2 / (n * sum x²)`. The primary `jain_service` uses each model's completed fraction of its measurement arrival cohort. `jain_deadline_success` uses each model's on-time fraction. SLO-class variants use the same definitions for strict/relaxed classes.
- **Profile MAPE:** mean `abs(observed − predicted) / observed` for completed batches, expressed as a percentage. Accuracy is `100 − MAPE`; P95 absolute percentage error is also reported. `profile_macro_accuracy_pct` averages accuracy equally over observed model/device pairs. Online pair tables aggregate each run's MAPE and P95 across seeds; offline pair tables pool held-out samples across supported shapes and batch sizes. Online figures use `compare` runs when available, otherwise the first available study.
- **Scheduling cost:** host `perf_counter_ns` duration of controller decision calls within the measurement interval, including scans that find no ready candidate; mean and P95 are in microseconds.

Empty completion windows have `null` latency, violation, and profile metrics. Fully drained cohort statistics are reported separately. Draining the same cohort and dividing by the same arrival duration would give identical QPS across lossless policies, so throughput comparisons use the finite measurement window and include its endpoint backlog.

Burst-comparison curves average each time bin across seeds. Orange opacity is proportional to the full method's mean protection fraction in that bin; dotted vertical lines mark the configured arrival spike. Peak burst P99 is computed per run before averaging, rather than taking the maximum of the averaged curve. The load plot's 100 ms line is a latency reference; individual request deadlines still use their model and SLO class.

## Implementation conventions

The controller implements the paper's completion estimate `t + wait + running_remaining + sum(queued_batch_runtime) + candidate_runtime`. Running work is counted once; unassigned controller requests are excluded. Queue urgency uses the minimum singleton slack over memory-feasible replicas. Placement minimizes predicted violation count, maximum tardiness, then completion time.

FIFO candidates use supported sizes. The ready candidate is the largest prefix predicted to meet all its deadlines; if none is safe, a singleton is dispatched to the least-risk worker. Waiting tests the next supported size above the currently available request count, recomputes the target runtime and memory, checks every included deadline, and requires a lower execution cost per request. Arrival, completion, 10 ms update, and wait-expiry events reevaluate the decision. Waiting does not reserve future capacity and cannot restart indefinitely for the same queue head.

The following choices fill details not numerically specified in the paper:

- Normal wait is capped at 5 ms. The default memory reserve is 256 MiB and low priority means `<0.5`.
- P95 relative-error margins inflate candidate runtime for feasibility; central estimates determine worker ranking and per-request execution cost. The residual guard is added separately. Queue backlog uses central runtime estimates.
- Burst detection uses aggregate per-model arrival EMAs against the previous 200 ms mean, starts after a full reference window, and applies the paper's consecutive entry/exit counts.
- Guard and stale checks are per profile tuple. A stale tuple still receives execution feedback, but controlled waiting for growth remains disabled until a newly built profile is loaded.
- Baseline batch rules, strict/relaxed budgets, fairness shares, finite-window accounting, shape choices, burst binning, profile-bias duration, reference capacity, and synthetic cost curves are defined above because the paper does not specify them completely.
- Shared CPU payloads are gathered and resized inside each worker instead of being maintained as permanently pinned, preprocessed buffers. Transfer and IPC costs appear in actual request latency but are not part of the execution table. CPU memory estimates are intended for the CPU example; CUDA runs use allocator measurements and runtime memory checks.
- Simulation records controller host cost separately and does not add that cost, IPC, or transfer time to its simulated request clock. Synthetic-profile plots therefore describe scheduler behavior under the configured cost model. Measured-profile simulation still draws from offline execution samples; actual hardware inference requires `--backend torch`.

## License

MIT for this repository's code. Downloaded datasets, model weights, and external model sources retain their respective licenses.
