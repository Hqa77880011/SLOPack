from time import perf_counter_ns
import numpy as np
from slopack.io import write_json
from slopack.models import forward, load_model, load_payloads, make_inputs, parameter_mb
from slopack.profiles import summarize_samples


def torch_profile(
    config,
    output,
    samples=200,
    warmup=20,
    heldout=40,
    precision="fp32",
    threads=1,
    payload_path=None,
):
    import torch

    if min(samples, heldout, threads) < 1 or warmup < 0:
        raise ValueError(
            "samples, heldout, threads must be positive and warmup nonnegative"
        )
    torch.set_num_threads(threads)
    bank = load_payloads(payload_path)
    entries = []
    skipped = []
    cuda_timings = []
    for worker in config["workers"]:
        device = torch.device(worker["device"])
        if device.type == "cuda" and (not torch.cuda.is_available()):
            raise ValueError("CUDA worker configured but CUDA is unavailable")
        for spec in config["models"]:
            if spec["name"] not in worker.get(
                "replicas", [m["name"] for m in config["models"]]
            ):
                continue
            model = load_model(spec, device, precision)
            resident = parameter_mb(model)
            for shape in spec["shapes"]:
                for size in config["batch_sizes"]:
                    inputs = None
                    result = None
                    try:
                        inputs = make_inputs(
                            spec, shape, list(range(size)), bank, device, precision
                        )
                        with torch.inference_mode():
                            for _ in range(warmup):
                                result = forward(model, inputs)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                                torch.cuda.reset_peak_memory_stats(device)
                                baseline = torch.cuda.memory_allocated(device)
                            values = []
                            gpu_values = []
                            for _ in range(samples + heldout):
                                if device.type == "cuda":
                                    begin = torch.cuda.Event(enable_timing=True)
                                    finish = torch.cuda.Event(enable_timing=True)
                                    begin.record(torch.cuda.current_stream(device))
                                start = perf_counter_ns()
                                result = forward(model, inputs)
                                if device.type == "cuda":
                                    finish.record(torch.cuda.current_stream(device))
                                    finish.synchronize()
                                    gpu_values.append(begin.elapsed_time(finish))
                                values.append((perf_counter_ns() - start) / 1000000.0)
                        if device.type == "cuda":
                            input_mb = (
                                sum(
                                    (
                                        t.numel() * t.element_size()
                                        for t in (
                                            inputs.values()
                                            if isinstance(inputs, dict)
                                            else [inputs]
                                        )
                                    )
                                )
                                / 2**20
                            )
                            memory = (
                                torch.cuda.max_memory_allocated(device) - baseline
                            ) / 2**20 + input_mb
                        else:
                            memory = size * shape * shape * 3 * 4 / 2**20
                            if spec["architecture"] == "bert_base":
                                memory = size * shape * 768 * 4 / 2**20
                        entry = summarize_samples(
                            spec["name"],
                            worker["name"],
                            shape,
                            size,
                            values[:samples],
                            values[samples:],
                            memory,
                            resident,
                        )
                        entries.append(entry)
                        if gpu_values:
                            cuda_timings.append(
                                {
                                    "model": spec["name"],
                                    "worker": worker["name"],
                                    "shape": shape,
                                    "batch_size": size,
                                    "median_ms": float(np.median(gpu_values[:samples])),
                                    "p95_ms": float(
                                        np.percentile(gpu_values[:samples], 95)
                                    ),
                                }
                            )
                        print(
                            f"{spec['name']} {worker['name']} shape={shape} B={size}: {entry['median_ms']:.3f} ms",
                            flush=True,
                        )
                    except torch.OutOfMemoryError:
                        skipped.append(
                            {
                                "model": spec["name"],
                                "worker": worker["name"],
                                "shape": shape,
                                "batch_size": size,
                                "reason": "out_of_memory",
                            }
                        )
                    finally:
                        del inputs, result
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_json(
        output,
        {
            "metadata": {
                "source": "torch_measured",
                "torch_version": torch.__version__,
                "precision": precision,
                "threads": threads,
                "samples_per_tuple": samples,
                "heldout_per_tuple": heldout,
                "warmup_iterations": warmup,
                "model_specs": config["models"],
                "worker_devices": {w["name"]: w["device"] for w in config["workers"]},
                "device_names": {
                    w["name"]: torch.cuda.get_device_name(w["device"])
                    if str(w["device"]).startswith("cuda")
                    else "CPU"
                    for w in config["workers"]
                },
                "latency_clock": "host forward plus CUDA event synchronization; excludes preprocessing and H2D",
                "memory": "CUDA peak allocated activation plus input; CPU input-size estimate",
                "units": "milliseconds and MiB",
                "skipped_tuples": skipped,
                "cuda_event_timings": cuda_timings,
            },
            "entries": entries,
        },
    )
