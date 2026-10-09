from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


POLICIES = (
    "slopack",
    "fcfs",
    "static",
    "fixed_timeout",
    "edf",
    "sjf",
    "queue_length",
    "fastest_gpu",
    "dynamic_no_slack",
)
ABLATIONS = (
    "none",
    "no_arrival",
    "no_hetero",
    "no_queue",
    "no_aging",
    "no_burst",
)


@dataclass
class Settings:
    rho: float = 0.15
    eta: float = 0.2
    window_ms: float = 10.0
    epsilon: float = 1e-6
    alpha: float = 1.0
    beta: float = 0.25
    gamma: float = 0.02
    wait_cap_ms: float = 5.0
    memory_reserve_mb: float = 256.0
    burst_reference_ms: float = 200.0
    burst_enter_ratio: float = 1.25
    burst_exit_ratio: float = 1.05
    burst_enter_windows: int = 3
    burst_exit_windows: int = 5
    burst_wait_ms: float = 0.5
    burst_batch_cap: int = 8
    low_priority_threshold: float = 0.5
    guard_floor_ms: float = 0.2
    guard_update_batches: int = 100
    stale_window: int = 100
    stale_factor: float = 2.0
    static_batch_size: int = 8
    timeout_ms: float = 5.0
    energy_mode: bool = False

    def __post_init__(self):
        if not 0 < self.rho <= 1 or not 0 < self.eta <= 1:
            raise ValueError("EMA weights must lie in (0, 1]")
        positive = (
            self.window_ms,
            self.epsilon,
            self.burst_reference_ms,
            self.burst_enter_windows,
            self.burst_exit_windows,
            self.guard_update_batches,
            self.stale_window,
            self.static_batch_size,
        )
        if min(positive) <= 0:
            raise ValueError("Windows, counts, and static batch size must be positive")
        if (
            min(
                self.wait_cap_ms,
                self.timeout_ms,
                self.memory_reserve_mb,
                self.guard_floor_ms,
            )
            < 0
        ):
            raise ValueError("Wait, reserve, and guard values must be nonnegative")


@dataclass
class Batch:
    batch_id: int
    requests: list
    worker: str
    assigned_ms: float
    prediction_ms: float
    start_ms: float | None = None
    end_ms: float | None = None
    observed_ms: float | None = None

    @property
    def model(self):
        return self.requests[0].model

    @property
    def shape(self):
        return self.requests[0].shape

    @property
    def size(self):
        return len(self.requests)


@dataclass
class WorkerState:
    name: str
    free_mb: float
    replicas: set
    running: Batch | None = None
    pending: deque = field(default_factory=deque)

    def backlog(self, now, profiles):
        remaining = 0.0
        if self.running is not None:
            batch = self.running
            entry = profiles.get(batch.model, self.name, batch.shape, batch.size)
            remaining = max(0.0, entry.estimate_ms - (now - batch.start_ms))
        return remaining + sum(
            profiles.get(b.model, self.name, b.shape, b.size).estimate_ms
            for b in self.pending
        )


@dataclass
class Wait:
    head_id: int
    start_ms: float
    until_ms: float


class ArrivalEstimator:
    def __init__(self, model_names, settings):
        self.settings = settings
        self.counts = dict.fromkeys(model_names, 0)
        self.rates = dict.fromkeys(model_names, 0.0)
        self.reference = deque(
            maxlen=max(1, round(settings.burst_reference_ms / settings.window_ms))
        )
        self.enter_count = 0
        self.exit_count = 0
        self.protection = False

    def observe(self, request):
        self.counts[request.model] += 1

    def tick(self):
        s = self.settings
        for model, count in self.counts.items():
            self.rates[model] = (
                s.eta * count / s.window_ms + (1 - s.eta) * self.rates[model]
            )
            self.counts[model] = 0
        rate = sum(self.rates.values())
        reference = float(np.mean(self.reference)) if self.reference else rate
        if len(self.reference) == self.reference.maxlen:
            self.enter_count = (
                self.enter_count + 1 if rate > s.burst_enter_ratio * reference else 0
            )
            self.exit_count = (
                self.exit_count + 1 if rate < s.burst_exit_ratio * reference else 0
            )
            if not self.protection and self.enter_count >= s.burst_enter_windows:
                self.protection = True
                self.exit_count = 0
            elif self.protection and self.exit_count >= s.burst_exit_windows:
                self.protection = False
                self.enter_count = 0
        self.reference.append(rate)


class Scheduler:
    def __init__(
        self, config, profiles, policy="slopack", ablation="none", overrides=None
    ):
        if policy not in POLICIES or ablation not in ABLATIONS:
            raise ValueError("Unknown policy or ablation")
        if policy != "slopack" and ablation != "none":
            raise ValueError("Ablations apply to slopack only")
        self.config = config
        self.profiles = profiles
        self.policy = policy
        self.ablation = ablation
        values = dict(config.get("scheduler", {}))
        values.update(overrides or {})
        self.settings = Settings(**values)
        for entry in profiles.entries.values():
            entry.guard_ms = entry.residual_p95_ms + self.settings.guard_floor_ms
        self.sizes = config["batch_sizes"]
        model_names = [model["name"] for model in config["models"]]
        self.workers = {}
        for worker in config["workers"]:
            replicas = set(worker.get("replicas", model_names))
            resident = sum(profiles.resident(m, worker["name"]) for m in replicas)
            free = worker["memory_mb"] - resident - self.settings.memory_reserve_mb
            self.workers[worker["name"]] = WorkerState(worker["name"], free, replicas)
        self.worker_config = {w["name"]: w for w in config["workers"]}
        self.queues = defaultdict(deque)
        self.waits = {}
        self.estimator = ArrivalEstimator(model_names, self.settings)
        self.next_batch_id = 0
        self.decision_costs = []
        for model in config["models"]:
            for shape in model["shapes"]:
                if not self.eligible(model["name"], shape, 1):
                    raise ValueError(
                        f"No memory-feasible singleton for {model['name']}/{shape}"
                    )

    @property
    def protection(self):
        return (
            self.policy == "slopack"
            and self.ablation != "no_burst"
            and self.estimator.protection
        )

    def add(self, request):
        self.queues[request.queue_key].append(request)
        self.estimator.observe(request)

    def eligible(self, model, shape, size):
        workers = []
        for worker in self.workers.values():
            entry = self.profiles.get(model, worker.name, shape, size)
            if (
                model in worker.replicas
                and entry is not None
                and entry.memory_mb <= worker.free_mb
            ):
                workers.append(worker)
        return workers

    def backlog(self, worker, now):
        return (
            0.0 if self.ablation == "no_queue" else worker.backlog(now, self.profiles)
        )

    def placement(self, requests, now, conservative=False):
        head = requests[0]
        workers = self.eligible(head.model, head.shape, len(requests))
        if self.ablation == "no_hetero" or self.policy == "fastest_gpu":
            workers = sorted(
                workers,
                key=lambda w: (
                    self.profiles.get(
                        head.model, w.name, head.shape, len(requests)
                    ).estimate_ms,
                    w.name,
                ),
            )[:1]
        choices = []
        for worker in workers:
            entry = self.profiles.get(
                head.model, worker.name, head.shape, len(requests)
            )
            runtime = entry.conservative_ms if conservative else entry.estimate_ms
            completion = now + self.backlog(worker, now) + runtime
            violations = sum(completion > r.deadline_ms for r in requests)
            tardiness = max(max(0, completion - r.deadline_ms) for r in requests)
            key = (violations, tardiness, completion)
            if self.policy == "queue_length":
                count = sum(b.size for b in worker.pending)
                count += worker.running.size if worker.running else 0
                key = (count, completion)
            if self.settings.energy_mode and not self.protection:
                power = self.worker_config[worker.name].get("power_w")
                if power is None:
                    raise ValueError("energy_mode requires power_w for every worker")
                key += (power * entry.estimate_ms / len(requests),)
            choices.append((key + (worker.name,), worker, entry, completion))
        return min(choices, key=lambda value: value[0]) if choices else None

    def queue_order(self, now):
        active = [key for key, queue in self.queues.items() if queue]

        def rank(key):
            request = self.queues[key][0]
            if self.policy in (
                "fcfs",
                "static",
                "fixed_timeout",
                "queue_length",
                "fastest_gpu",
                "dynamic_no_slack",
            ):
                return request.arrival_ms, request.request_id
            if self.policy == "edf":
                return request.deadline_ms, request.request_id
            if self.policy == "sjf":
                shortest = min(
                    self.profiles.get(
                        request.model, w.name, request.shape, 1
                    ).estimate_ms
                    for w in self.eligible(request.model, request.shape, 1)
                )
                return shortest, request.request_id
            slack = min(
                request.deadline_ms
                - now
                - self.backlog(w, now)
                - self.profiles.get(request.model, w.name, request.shape, 1).estimate_ms
                for w in self.eligible(request.model, request.shape, 1)
            )
            gamma = 0 if self.ablation == "no_aging" else self.settings.gamma
            score = -self.settings.alpha * slack + self.settings.beta * request.priority
            score += gamma * (now - request.arrival_ms)
            return -score, request.request_id

        return sorted(active, key=rank)

    def _prefix(self, queue, size):
        return [queue[i] for i in range(size)]

    def _candidate(self, key, now, exhausted):
        queue = self.queues[key]
        head = queue[0]
        cap = max(self.sizes)
        if self.protection and head.priority < self.settings.low_priority_threshold:
            cap = min(cap, self.settings.burst_batch_cap)
        supported = [
            b
            for b in self.sizes
            if b <= cap and self.eligible(head.model, head.shape, b)
        ]
        if not supported:
            return None
        available = [b for b in supported if b <= len(queue)]
        size = max(available)
        if self.policy in ("fcfs", "edf", "sjf"):
            return self._prefix(queue, 1)
        if self.policy == "static":
            target = max(b for b in supported if b <= self.settings.static_batch_size)
            if len(queue) < target and not exhausted:
                return None
            return self._prefix(queue, min(size, target))
        if self.policy != "slopack":
            target = max(supported)
            until = head.arrival_ms + self.settings.timeout_ms
            wait = self.waits.get(key)
            expired = (
                wait and wait.head_id == head.request_id and now >= wait.until_ms - 1e-9
            )
            if len(queue) < target and now < until and not exhausted and not expired:
                if self.policy == "fixed_timeout":
                    self.waits[key] = Wait(head.request_id, head.arrival_ms, until)
                    return None
                target = min(b for b in supported if b > len(queue))
                fill = (target - len(queue)) / max(
                    self.estimator.rates[head.model], self.settings.epsilon
                )
                improves = any(
                    self.profiles.get(head.model, w.name, head.shape, size) is not None
                    and self.profiles.get(
                        head.model, w.name, head.shape, target
                    ).estimate_ms
                    / target
                    < self.profiles.get(
                        head.model, w.name, head.shape, size
                    ).estimate_ms
                    / size
                    for w in self.eligible(head.model, head.shape, target)
                )
                if improves and fill <= until - now:
                    self.waits[key] = Wait(head.request_id, head.arrival_ms, now + fill)
                    return None
            return self._prefix(queue, size)
        safe_sizes = []
        for b in available:
            requests = self._prefix(queue, b)
            placement = self.placement(requests, now, conservative=True)
            if placement and placement[3] <= min(r.deadline_ms for r in requests):
                safe_sizes.append(b)
        size = max(safe_sizes, default=1)
        requests = self._prefix(queue, size)
        central = self.placement(requests, now)
        if (
            not central
            or central[3] >= min(r.deadline_ms for r in requests)
            or exhausted
        ):
            return requests
        wait = self.waits.get(key)
        if wait is not None and wait.head_id != head.request_id:
            wait = None
        start = wait.start_ms if wait else now
        cap_ms = (
            self.settings.burst_wait_ms
            if self.protection
            else self.settings.wait_cap_ms
        )
        if now >= start + cap_ms or (wait and now >= wait.until_ms - 1e-9):
            return requests
        targets = [b for b in supported if b > len(queue)]
        if not targets:
            return requests
        target = min(targets)
        included = list(queue)
        deadline = min(r.deadline_ms for r in included)
        rate = self.estimator.rates[head.model]
        fill = (target - len(queue)) / max(rate, self.settings.epsilon)
        untils = []
        allowed_worker = None
        if self.ablation == "no_hetero":
            workers = self.eligible(head.model, head.shape, target)
            allowed_worker = min(
                workers,
                key=lambda w: (
                    self.profiles.get(
                        head.model, w.name, head.shape, target
                    ).estimate_ms,
                    w.name,
                ),
            ).name
        for worker in self.eligible(head.model, head.shape, target):
            if allowed_worker and worker.name != allowed_worker:
                continue
            entry = self.profiles.get(head.model, worker.name, head.shape, target)
            current = self.profiles.get(head.model, worker.name, head.shape, size)
            if current is None or entry.stale or current.stale:
                continue
            if entry.estimate_ms / target >= current.estimate_ms / size:
                continue
            slack = deadline - now - self.backlog(worker, now) - entry.conservative_ms
            guarded = slack - entry.guard_ms
            if self.ablation == "no_arrival":
                if guarded > 0:
                    untils.append(min(now + guarded, start + cap_ms))
            elif fill <= guarded:
                untils.append(min(now + fill, now + guarded, start + cap_ms))
        if untils and max(untils) > now + 1e-9:
            self.waits[key] = Wait(head.request_id, start, max(untils))
            return None
        return requests

    def decide(self, now, exhausted=False):
        for key in self.queue_order(now):
            requests = self._candidate(key, now, exhausted)
            if not requests:
                continue
            placement = self.placement(requests, now)
            if placement is None:
                continue
            _, worker, entry, _ = placement
            queue = self.queues[key]
            for _ in requests:
                queue.popleft()
            self.waits.pop(key, None)
            batch = Batch(
                self.next_batch_id, requests, worker.name, now, entry.estimate_ms
            )
            self.next_batch_id += 1
            worker.pending.append(batch)
            return batch
        return None

    def next_wakeup(self, now):
        values = [
            wait.until_ms
            for key, wait in self.waits.items()
            if self.queues[key] and wait.until_ms > now + 1e-9
        ]
        return min(values, default=float("inf"))

    def started(self, batch, now):
        worker = self.workers[batch.worker]
        if (
            worker.running is not None
            or not worker.pending
            or worker.pending[0] is not batch
        ):
            raise RuntimeError("Worker start order does not match assigned FIFO")
        worker.pending.popleft()
        worker.running = batch
        batch.start_ms = now

    def completed(self, batch, now, observed):
        worker = self.workers[batch.worker]
        if worker.running is not batch:
            raise RuntimeError("Completion does not match running batch")
        worker.running = None
        batch.end_ms = now
        batch.observed_ms = observed
        entry = self.profiles.get(batch.model, batch.worker, batch.shape, batch.size)
        return entry.update(observed, self.settings)

    def requeue(self, batch):
        worker = self.workers[batch.worker]
        if worker.running is batch:
            worker.running = None
        else:
            worker.pending.remove(batch)
        queue = self.queues[batch.requests[0].queue_key]
        merged = list(queue) + batch.requests
        merged.sort(key=lambda request: (request.arrival_ms, request.request_id))
        self.queues[batch.requests[0].queue_key] = deque(merged)
