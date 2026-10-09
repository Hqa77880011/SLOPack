import multiprocessing as mp
from queue import Empty
from time import perf_counter_ns

from slopack.metrics import save_run
from slopack.models import (
    forward,
    load_model,
    load_payloads,
    make_inputs,
    prediction_summary,
)
from slopack.profiles import Profiles
from slopack.scheduler import Scheduler


def _worker_loop(worker, specs, precision, threads, bank, jobs, messages, reserve):
    import torch

    torch.set_num_threads(threads)
    device = torch.device(worker["device"])
    replicas = worker.get("replicas", [spec["name"] for spec in specs])
    try:
        models = {
            spec["name"]: load_model(spec, device, precision)
            for spec in specs
            if spec["name"] in replicas
        }
        spec_by_name = {spec["name"]: spec for spec in specs}
        with torch.inference_mode():
            for name, model in models.items():
                spec = spec_by_name[name]
                for shape in spec["shapes"]:
                    inputs = make_inputs(spec, shape, [0], bank, device, precision)
                    for _ in range(2):
                        forward(model, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        messages.put(("ready", worker["name"]))
        while True:
            job = jobs.get()
            if job is None:
                return
            batch, expected_memory = job
            model = models[batch.model]
            spec = spec_by_name[batch.model]
            inputs = None
            output = None
            try:
                if device.type == "cuda":
                    free, _ = torch.cuda.mem_get_info(device)
                    reclaimable = torch.cuda.memory_reserved(
                        device
                    ) - torch.cuda.memory_allocated(device)
                    available = (free + reclaimable) / 2**20 - reserve
                    if expected_memory > available:
                        messages.put(("rejected", batch.batch_id, available))
                        continue
                payloads = [r.payload_id for r in batch.requests]
                inputs = make_inputs(
                    spec, batch.shape, payloads, bank, device, precision
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter_ns()
                messages.put(("started", batch.batch_id, started))
                with torch.inference_mode():
                    if device.type == "cuda":
                        begin = torch.cuda.Event(enable_timing=True)
                        finish = torch.cuda.Event(enable_timing=True)
                        begin.record(torch.cuda.current_stream(device))
                    output = forward(model, inputs)
                    if device.type == "cuda":
                        finish.record(torch.cuda.current_stream(device))
                        finish.synchronize()
                    ended = perf_counter_ns()
                    gpu_ms = (
                        begin.elapsed_time(finish) if device.type == "cuda" else None
                    )
                    predictions = prediction_summary(spec, output)
                messages.put(
                    (
                        "completed",
                        batch.batch_id,
                        ended,
                        (ended - started) / 1e6,
                        predictions,
                        gpu_ms,
                    )
                )
            except torch.OutOfMemoryError:
                messages.put(("oom", batch.batch_id))
            finally:
                del inputs, output
    except (RuntimeError, ValueError, OSError, ImportError, KeyError) as error:
        messages.put(("error", worker["name"], f"{type(error).__name__}: {error}"))


def execute(
    config,
    profile_path,
    requests,
    output,
    duration_ms,
    warmup_ms=0,
    policy="slopack",
    ablation="none",
    seed=0,
    precision="fp32",
    threads=1,
    payload_path=None,
    overrides=None,
    metadata=None,
    startup_timeout_s=300,
    bias=0.0,
    bias_mode="sustained",
):
    import torch

    if not 0 <= warmup_ms < duration_ms:
        raise ValueError("warmup must be nonnegative and shorter than duration")
    torch.set_num_threads(threads)
    profiles = Profiles.load(profile_path, bias, bias_mode)
    if profiles.metadata["source"] != "torch_measured":
        raise ValueError("Real execution requires a measured torch profile")
    if precision != profiles.metadata["precision"]:
        raise ValueError("Profile and execution precision must match")
    if profiles.metadata["model_specs"] != config["models"]:
        raise ValueError("Model settings changed; profile them again before execution")
    devices = {w["name"]: w["device"] for w in config["workers"]}
    if profiles.metadata["worker_devices"] != devices:
        raise ValueError("Worker devices changed; profile them again before execution")
    if threads != profiles.metadata["threads"]:
        raise ValueError("CPU thread count must match the profile")
    scheduler = Scheduler(config, profiles, policy, ablation, overrides)
    bank = load_payloads(payload_path, seed=seed)
    model_specs = {spec["name"]: spec for spec in config["models"]}
    for request in requests:
        source = (
            bank["tokens"]
            if model_specs[request.model]["architecture"] == "bert_base"
            and "tokens" in bank
            else bank["images"]
        )
        if request.payload_id >= len(source):
            raise ValueError(
                "Trace payload_id exceeds payload bank; regenerate the trace with --payloads"
            )
    context = mp.get_context("spawn")
    messages = context.Queue()
    jobs = {w["name"]: context.Queue() for w in config["workers"]}
    processes = []
    try:
        for worker in config["workers"]:
            process = context.Process(
                target=_worker_loop,
                args=(
                    worker,
                    config["models"],
                    precision,
                    threads,
                    bank,
                    jobs[worker["name"]],
                    messages,
                    scheduler.settings.memory_reserve_mb,
                ),
            )
            process.start()
            processes.append(process)
        ready = set()
        startup_deadline = perf_counter_ns() + startup_timeout_s * 1e9
        while len(ready) < len(processes):
            remaining = (startup_deadline - perf_counter_ns()) / 1e9
            if remaining <= 0:
                raise RuntimeError("Worker startup timed out")
            try:
                message = messages.get(timeout=min(remaining, 1))
            except Empty:
                if any(not p.is_alive() for p in processes):
                    raise RuntimeError("Worker exited during startup")
                continue
            if message[0] == "error":
                raise RuntimeError(f"Worker {message[1]}: {message[2]}")
            ready.add(message[1])
        epoch = perf_counter_ns()
        cursor = 0
        next_tick = scheduler.settings.window_ms
        assigned = {}
        completed = []
        events = []
        predictions = []
        gpu_times = []

        def handle(message):
            kind = message[0]
            if kind == "error":
                raise RuntimeError(f"Worker {message[1]}: {message[2]}")
            batch = assigned[message[1]]
            if kind == "started":
                scheduler.started(batch, (message[2] - epoch) / 1e6)
            elif kind == "completed":
                scheduler.completed(batch, (message[2] - epoch) / 1e6, message[3])
                completed.append(batch)
                predictions.extend(
                    {
                        "request_id": r.request_id,
                        "model": batch.model,
                        "prediction": value,
                    }
                    for r, value in zip(batch.requests, message[4])
                )
                gpu_times.append(
                    {"batch_id": batch.batch_id, "cuda_event_ms": message[5]}
                )
                del assigned[batch.batch_id]
            elif kind in ("rejected", "oom"):
                scheduler.requeue(batch)
                entry = profiles.get(batch.model, batch.worker, batch.shape, batch.size)
                entry.memory_mb = float("inf")
                del assigned[batch.batch_id]
                if not scheduler.eligible(batch.model, batch.shape, 1):
                    raise RuntimeError(
                        f"No worker can execute singleton {batch.model}/{batch.shape} after memory rejection"
                    )

        while cursor < len(requests) or assigned or any(scheduler.queues.values()):
            while True:
                try:
                    handle(messages.get_nowait())
                except Empty:
                    break
            if any(not process.is_alive() for process in processes):
                raise RuntimeError("Worker exited during execution")
            now = (perf_counter_ns() - epoch) / 1e6
            while next_tick <= now:
                while (
                    cursor < len(requests) and requests[cursor].arrival_ms < next_tick
                ):
                    scheduler.add(requests[cursor])
                    cursor += 1
                scheduler.estimator.tick()
                events.append(
                    {
                        "time_ms": next_tick,
                        "backlog": sum(len(q) for q in scheduler.queues.values())
                        + sum(b.size for b in assigned.values()),
                        "arrival_rate_per_ms": sum(scheduler.estimator.rates.values()),
                        "protection": scheduler.protection,
                    }
                )
                next_tick += scheduler.settings.window_ms
            while cursor < len(requests) and requests[cursor].arrival_ms <= now:
                scheduler.add(requests[cursor])
                cursor += 1
            while True:
                decision_start = perf_counter_ns()
                batch = scheduler.decide(now, now >= duration_ms)
                scheduler.decision_costs.append(
                    {
                        "time_ms": now,
                        "cost_us": (perf_counter_ns() - decision_start) / 1000,
                    }
                )
                if batch is None:
                    break
                assigned[batch.batch_id] = batch
                entry = profiles.get(batch.model, batch.worker, batch.shape, batch.size)
                jobs[batch.worker].put((batch, entry.memory_mb))
                now = (perf_counter_ns() - epoch) / 1e6
            if (
                cursor >= len(requests)
                and not assigned
                and not any(scheduler.queues.values())
            ):
                break
            wakeup = min(
                requests[cursor].arrival_ms if cursor < len(requests) else float("inf"),
                next_tick,
                scheduler.next_wakeup(now),
            )
            timeout = max(0.0001, min(0.01, (wakeup - now) / 1000))
            try:
                handle(messages.get(timeout=timeout))
            except Empty:
                continue
        meta = {
            "backend": "torch",
            "policy": policy,
            "ablation": ablation,
            "seed": seed,
            "profile_bias": bias,
            "bias_mode": bias_mode,
            "precision": precision,
            "duration_ms": duration_ms,
            "warmup_ms": warmup_ms,
            "wait_cap_ms": scheduler.settings.wait_cap_ms,
            "guard_floor_ms": scheduler.settings.guard_floor_ms,
            "timeout_ms": scheduler.settings.timeout_ms,
            "static_batch_size": scheduler.settings.static_batch_size,
            "config": config,
            "scheduler_settings": vars(scheduler.settings),
            "profile_source": profiles.metadata,
            "drained": True,
            "latency_clock": "arrival descriptor to synchronized model forward completion; includes controller, IPC, preprocessing, H2D, queueing",
            "execution_clock": "host forward plus CUDA event synchronization; excludes preprocessing and H2D",
            "payload_source": str(payload_path)
            if payload_path
            else "generated deterministic tensors",
        }
        meta.update(metadata or {})
        result = save_run(
            output,
            requests,
            completed,
            scheduler,
            warmup_ms,
            duration_ms,
            meta,
            events,
            predictions,
        )
        from pathlib import Path
        from slopack.io import write_csv

        write_csv(
            Path(output) / "cuda_times.csv", gpu_times, ["batch_id", "cuda_event_ms"]
        )
        return result
    finally:
        for queue in jobs.values():
            queue.put(None)
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for queue in jobs.values():
            queue.close()
        messages.close()
