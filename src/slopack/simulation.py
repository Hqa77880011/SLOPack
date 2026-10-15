import hashlib
import heapq
from time import perf_counter_ns
from slopack.metrics import save_run
from slopack.profiles import Profiles
from slopack.scheduler import Scheduler


def simulate(
    config,
    profile_path,
    requests,
    output,
    duration_ms,
    warmup_ms=0,
    policy="slopack",
    ablation="none",
    seed=0,
    bias=0,
    overrides=None,
    metadata=None,
    bias_mode="sustained",
):
    if not 0 <= warmup_ms < duration_ms:
        raise ValueError("warmup must be nonnegative and shorter than duration")
    profiles = Profiles.load(profile_path, bias, bias_mode)
    scheduler = Scheduler(config, profiles, policy, ablation, overrides)
    completions = []
    batches = []
    events = []
    cursor = 0
    now = 0.0
    next_tick = scheduler.settings.window_ms

    def start_pending(worker):
        if worker.running is None and worker.pending:
            batch = worker.pending[0]
            scheduler.started(batch, now)
            entry = profiles.get(batch.model, worker.name, batch.shape, batch.size)
            identity = f"{seed}:{batch.model}:{worker.name}:{batch.shape}:{','.join((str(r.request_id) for r in batch.requests))}"
            index = int.from_bytes(
                hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little"
            ) % len(entry.samples_ms)
            observed = entry.samples_ms[index]
            heapq.heappush(
                completions, (now + observed, batch.batch_id, batch, observed)
            )

    while (
        cursor < len(requests)
        or completions
        or any(scheduler.queues.values())
        or any((w.pending for w in scheduler.workers.values()))
    ):
        next_arrival = (
            requests[cursor].arrival_ms if cursor < len(requests) else float("inf")
        )
        next_completion = completions[0][0] if completions else float("inf")
        now = min(next_arrival, next_completion, next_tick, scheduler.next_wakeup(now))
        while completions and completions[0][0] <= now + 1e-09:
            end, _, batch, observed = heapq.heappop(completions)
            scheduler.completed(batch, end, observed)
            batches.append(batch)
            start_pending(scheduler.workers[batch.worker])
        if now >= next_tick - 1e-09:
            scheduler.estimator.tick()
            events.append(
                {
                    "time_ms": now,
                    "backlog": sum((len(q) for q in scheduler.queues.values()))
                    + sum(
                        (
                            sum((b.size for b in w.pending))
                            + (w.running.size if w.running else 0)
                            for w in scheduler.workers.values()
                        )
                    ),
                    "arrival_rate_per_ms": sum(scheduler.estimator.rates.values()),
                    "protection": scheduler.protection,
                }
            )
            next_tick += scheduler.settings.window_ms
        while cursor < len(requests) and requests[cursor].arrival_ms <= now + 1e-09:
            scheduler.add(requests[cursor])
            cursor += 1
        exhausted = now >= duration_ms
        while True:
            started = perf_counter_ns()
            batch = scheduler.decide(now, exhausted)
            scheduler.decision_costs.append(
                {"time_ms": now, "cost_us": (perf_counter_ns() - started) / 1000}
            )
            if batch is None:
                break
            start_pending(scheduler.workers[batch.worker])
    meta = {
        "backend": "simulation",
        "policy": policy,
        "ablation": ablation,
        "seed": seed,
        "profile_bias": bias,
        "bias_mode": bias_mode,
        "duration_ms": duration_ms,
        "warmup_ms": warmup_ms,
        "profile_source": profiles.metadata,
        "wait_cap_ms": scheduler.settings.wait_cap_ms,
        "guard_floor_ms": scheduler.settings.guard_floor_ms,
        "timeout_ms": scheduler.settings.timeout_ms,
        "static_batch_size": scheduler.settings.static_batch_size,
        "config": config,
        "scheduler_settings": vars(scheduler.settings),
        "runtime_sampling": "fixed offline samples; deterministic by seed, tuple, and request IDs",
        "controller_cost_in_simulated_time": False,
        "drained": True,
    }
    meta.update(metadata or {})
    return save_run(
        output, requests, batches, scheduler, warmup_ms, duration_ms, meta, events
    )
