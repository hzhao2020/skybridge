"""
Multi-tree GP hyper-heuristic baseline for three-decision scheduling.

This module follows the core MTGP idea from Sun et al. (IEEE TSC 2024): one
individual contains three priority-rule trees for task, cloud, and resource
selection. The repository evaluates static deployment plans rather than a
multi-workflow event simulator, so endpoints are treated as the selectable
resource instances and the evolved rules are used to construct one deployment
plan that is evaluated by the shared SkyFlow evaluator.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from src.cost_latency import (
    execution_cost,
    execution_latency,
    network_latency,
    network_transfer_cost,
    storage_cost,
)
from src.data_propagation import edge_transfer_size
from src.schemas import Endpoint, Query, Scenario, SolverConfig, WorkflowDAG

Decision = Literal["task", "cloud", "instance"]
FunctionName = Literal["add", "sub", "mul", "div", "max", "min"]
QueryScenarioData = tuple[Query, Scenario, dict[str, float], dict[str, float]]

SUB_DEADLINE_EPSILON = 0.95
DEFAULT_POPULATION_SIZE = 28
DEFAULT_GENERATIONS = 16
DEFAULT_MAX_DEPTH = 5
DEFAULT_TOURNAMENT_SIZE = 3
DEFAULT_ELITE_COUNT = 2
DEFAULT_CROSSOVER_RATE = 0.80
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_FITNESS_SAMPLE_SIZE = 350
VIOLATION_PENALTY_WEIGHT = 50.0

FUNCTIONS: tuple[FunctionName, ...] = ("add", "sub", "mul", "div", "max", "min")

TASK_TERMINALS: tuple[str, ...] = (
    "SD",
    "UR",
    "ET",
    "NSK",
    "NKQ",
    "TETQ",
    "TETRK",
)
CLOUD_TERMINALS: tuple[str, ...] = ("CC", "VEC", "VST", "NKWC", "VIAT")
INSTANCE_TERMINALS: tuple[str, ...] = ("AAT", "ST", "EC", "IAT", "CC", "SD")


@dataclass(frozen=True)
class _AvgQueryScenario:
    query: Query
    input_sizes: dict[str, float]
    output_sizes: dict[str, float]


@dataclass
class _ScheduleState:
    partial: dict[str, Endpoint] = field(default_factory=dict)
    endpoint_available: dict[str, float] = field(default_factory=dict)
    node_finish: dict[str, float] = field(default_factory=dict)
    system_time: float = 0.0


@dataclass(frozen=True)
class _GPNode:
    function: FunctionName | None = None
    terminal: str | None = None
    constant: float | None = None
    left: "_GPNode | None" = None
    right: "_GPNode | None" = None

    def evaluate(self, values: dict[str, float]) -> float:
        if self.terminal is not None:
            return _clean_number(values.get(self.terminal, 0.0))
        if self.constant is not None:
            return self.constant
        if self.function is None or self.left is None or self.right is None:
            return 0.0
        a = self.left.evaluate(values)
        b = self.right.evaluate(values)
        if self.function == "add":
            return _clean_number(a + b)
        if self.function == "sub":
            return _clean_number(a - b)
        if self.function == "mul":
            return _clean_number(a * b)
        if self.function == "div":
            return 1.0 if abs(b) < 1e-12 else _clean_number(a / b)
        if self.function == "max":
            return max(a, b)
        if self.function == "min":
            return min(a, b)
        return 0.0

    def size(self) -> int:
        if self.function is None:
            return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)


@dataclass(frozen=True)
class _Individual:
    task_tree: _GPNode
    cloud_tree: _GPNode
    instance_tree: _GPNode


@dataclass(frozen=True)
class _TerminalStats:
    min_exec: dict[str, float]
    mean_exec: dict[tuple[str, str], float]
    mean_cost: dict[tuple[str, str], float]
    upward_rank: dict[str, float]
    sub_deadline: dict[str, float]
    mean_sla: float


def solve_mtgp(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
):
    """Evolve three GP priority rules and use them to build a deployment plan."""
    from src.baselines import _BaselineEvaluationCache, _node_candidates, _to_result

    t0 = time.perf_counter()
    rng = random.Random(config.random_seed)
    node_candidates = _node_candidates(workflow, endpoints, quality_level, config)
    metrics_cache = _BaselineEvaluationCache(
        workflow, endpoints, queries, scenarios, quality_level, config
    )
    avg_samples = _average_query_scenario_samples(metrics_cache.qs_data)
    terminal_stats = _precompute_terminal_stats(
        workflow,
        node_candidates,
        avg_samples,
        metrics_cache,
        config,
    )
    fitness_qs = _sample_fitness_qs(metrics_cache.qs_data, config, rng)

    best, history = _evolve_rules(
        workflow=workflow,
        node_candidates=node_candidates,
        avg_samples=avg_samples,
        terminal_stats=terminal_stats,
        metrics_cache=metrics_cache,
        fitness_qs=fitness_qs,
        config=config,
        rng=rng,
    )
    assignment = _schedule_with_individual(
        individual=best,
        workflow=workflow,
        node_candidates=node_candidates,
        avg_samples=avg_samples,
        terminal_stats=terminal_stats,
        metrics_cache=metrics_cache,
        config=config,
    )
    metrics = metrics_cache.evaluate(assignment, include_per_qs=True)
    result = _to_result(
        workflow=workflow,
        assignment=assignment,
        quality_level=quality_level,
        method="mtgp",
        metrics=metrics,
        runtime_sec=time.perf_counter() - t0,
    )
    result.num_iterations = len(history)
    result.active_scenario_count = len(fitness_qs)
    result.convergence_history = history
    return result


def solve_mtgp_3d(
    workflow: WorkflowDAG,
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
):
    """Backward-compatible alias for the MTGP baseline."""
    return solve_mtgp(workflow, endpoints, queries, scenarios, quality_level, config)


def _evolve_rules(
    *,
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    avg_samples: list[_AvgQueryScenario],
    terminal_stats: _TerminalStats,
    metrics_cache: _BaselineEvaluationCache,
    fitness_qs: list[QueryScenarioData],
    config: SolverConfig,
    rng: random.Random,
) -> tuple[_Individual, list[dict[str, float]]]:
    pop_size = _config_int(config, "mtgp_population_size", DEFAULT_POPULATION_SIZE)
    generations = _config_int(config, "mtgp_generations", DEFAULT_GENERATIONS)
    max_depth = _config_int(config, "mtgp_max_depth", DEFAULT_MAX_DEPTH)
    tournament_size = _config_int(config, "mtgp_tournament_size", DEFAULT_TOURNAMENT_SIZE)
    elite_count = min(_config_int(config, "mtgp_elite_count", DEFAULT_ELITE_COUNT), pop_size)
    crossover_rate = _config_float(config, "mtgp_crossover_rate", DEFAULT_CROSSOVER_RATE)
    mutation_rate = _config_float(config, "mtgp_mutation_rate", DEFAULT_MUTATION_RATE)

    population = [_random_individual(rng, max_depth, full=(i % 2 == 0)) for i in range(pop_size)]
    best_individual = population[0]
    best_fitness = math.inf
    history: list[dict[str, float]] = []

    for generation in range(generations):
        scored = [
            (
                _fitness(
                    individual,
                    workflow,
                    node_candidates,
                    avg_samples,
                    terminal_stats,
                    metrics_cache,
                    fitness_qs,
                    config,
                ),
                individual,
            )
            for individual in population
        ]
        scored.sort(key=lambda row: row[0])
        if scored[0][0] < best_fitness:
            best_fitness = scored[0][0]
            best_individual = scored[0][1]
        history.append(
            {
                "iteration": generation + 1,
                "active_scenario_count": len(fitness_qs),
                "objective_value": float(scored[0][0]),
                "max_violation": float(max(0.0, scored[0][0] - best_fitness)),
                "num_violated_scenarios": 0,
                "runtime_sec": 0.0,
            }
        )

        next_population = [ind for _, ind in scored[:elite_count]]
        while len(next_population) < pop_size:
            if rng.random() < crossover_rate and len(next_population) + 1 < pop_size:
                parent_a = _tournament(scored, tournament_size, rng)
                parent_b = _tournament(scored, tournament_size, rng)
                child_a, child_b = _crossover(parent_a, parent_b, rng)
                next_population.append(_maybe_mutate(child_a, rng, mutation_rate, max_depth))
                next_population.append(_maybe_mutate(child_b, rng, mutation_rate, max_depth))
            else:
                parent = _tournament(scored, tournament_size, rng)
                next_population.append(_maybe_mutate(parent, rng, 1.0, max_depth))
        population = next_population[:pop_size]

    return best_individual, history


def _fitness(
    individual: _Individual,
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    avg_samples: list[_AvgQueryScenario],
    terminal_stats: _TerminalStats,
    metrics_cache: _BaselineEvaluationCache,
    fitness_qs: list[QueryScenarioData],
    config: SolverConfig,
) -> float:
    try:
        assignment = _schedule_with_individual(
            individual=individual,
            workflow=workflow,
            node_candidates=node_candidates,
            avg_samples=avg_samples,
            terminal_stats=terminal_stats,
            metrics_cache=metrics_cache,
            config=config,
        )
    except Exception:
        return math.inf
    metrics = _evaluate_assignment_on_qs(metrics_cache, assignment, fitness_qs)
    excess_violation = max(0.0, float(metrics["violation_rate"]) - config.eta)
    return float(metrics["expected_cost"]) * (1.0 + VIOLATION_PENALTY_WEIGHT * excess_violation)


def _schedule_with_individual(
    *,
    individual: _Individual,
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    avg_samples: list[_AvgQueryScenario],
    terminal_stats: _TerminalStats,
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> dict[str, Endpoint]:
    state = _ScheduleState()
    compute_nodes = set(workflow.compute_nodes())
    remaining = set(compute_nodes)

    while remaining:
        ready = [
            node
            for node in remaining
            if all(pred in state.partial or pred not in compute_nodes for pred in workflow.predecessors(node))
        ]
        if not ready:
            raise RuntimeError("MTGP: no ready task while nodes remain unassigned")

        task_values = {
            node: _task_terminal_values(workflow, node, ready, terminal_stats)
            for node in ready
        }
        selected_node = min(
            ready,
            key=lambda n: (
                individual.task_tree.evaluate(task_values[n]),
                n,
            ),
        )

        providers = sorted({ep.provider for ep in node_candidates[selected_node]})
        selected_provider = min(
            providers,
            key=lambda p: (
                individual.cloud_tree.evaluate(
                    _cloud_terminal_values(
                        provider=p,
                        node=selected_node,
                        candidates=node_candidates[selected_node],
                        state=state,
                        avg_samples=avg_samples,
                        terminal_stats=terminal_stats,
                        metrics_cache=metrics_cache,
                        config=config,
                    )
                ),
                p,
            ),
        )
        provider_candidates = [
            ep for ep in node_candidates[selected_node] if ep.provider == selected_provider
        ]

        feasible: list[tuple[float, str, Endpoint]] = []
        fallback: list[tuple[float, str, Endpoint]] = []
        for ep in provider_candidates:
            start, finish = _estimate_times(
                workflow,
                selected_node,
                ep,
                state,
                avg_samples,
                metrics_cache,
                config,
            )
            values = _instance_terminal_values(
                workflow=workflow,
                node=selected_node,
                endpoint=ep,
                state=state,
                start=start,
                avg_samples=avg_samples,
                terminal_stats=terminal_stats,
                metrics_cache=metrics_cache,
                config=config,
            )
            priority = individual.instance_tree.evaluate(values)
            row = (priority, ep.endpoint_id, ep)
            if finish <= terminal_stats.sub_deadline[selected_node] + 1e-9:
                feasible.append(row)
            fallback.append((finish, ep.endpoint_id, ep))

        chosen = min(feasible, key=lambda row: (row[0], row[1]))[2] if feasible else min(fallback)[2]
        _commit_assignment(
            workflow=workflow,
            node=selected_node,
            endpoint=chosen,
            state=state,
            avg_samples=avg_samples,
            metrics_cache=metrics_cache,
            config=config,
        )
        remaining.remove(selected_node)

    return state.partial


def _task_terminal_values(
    workflow: WorkflowDAG,
    node: str,
    ready: list[str],
    terminal_stats: _TerminalStats,
) -> dict[str, float]:
    compute = set(workflow.compute_nodes())
    return {
        "SD": terminal_stats.sub_deadline[node],
        "UR": terminal_stats.upward_rank[node],
        "ET": terminal_stats.min_exec[node],
        "NSK": float(len([s for s in workflow.successors(node) if s in compute])),
        "NKQ": float(len(ready)),
        "TETQ": float(sum(terminal_stats.min_exec[n] for n in ready)),
        "TETRK": terminal_stats.mean_sla,
    }


def _cloud_terminal_values(
    *,
    provider: str,
    node: str,
    candidates: list[Endpoint],
    state: _ScheduleState,
    avg_samples: list[_AvgQueryScenario],
    terminal_stats: _TerminalStats,
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> dict[str, float]:
    provider_eps = [ep for ep in candidates if ep.provider == provider]
    costs = [terminal_stats.mean_cost[(node, ep.endpoint_id)] for ep in provider_eps]
    finishes = [
        _estimate_times(metrics_cache.workflow, node, ep, state, avg_samples, metrics_cache, config)[1]
        for ep in provider_eps
    ]
    viat = float(np.mean([state.endpoint_available.get(ep.endpoint_id, state.system_time) for ep in provider_eps]))
    return {
        "CC": float(
            np.mean(
                [
                    _communication_cost_to_node(node, ep, state.partial, avg_samples, metrics_cache, config)
                    for ep in provider_eps
                ]
            )
        )
        if provider_eps
        else 0.0,
        "VEC": float(np.mean(costs)) if costs else 0.0,
        "VST": min(finishes) if finishes else state.system_time,
        "NKWC": float(len(provider_eps)),
        "VIAT": viat,
    }


def _instance_terminal_values(
    *,
    workflow: WorkflowDAG,
    node: str,
    endpoint: Endpoint,
    state: _ScheduleState,
    start: float,
    avg_samples: list[_AvgQueryScenario],
    terminal_stats: _TerminalStats,
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> dict[str, float]:
    return {
        "AAT": start,
        "ST": state.system_time,
        "EC": terminal_stats.mean_cost[(node, endpoint.endpoint_id)],
        "IAT": state.endpoint_available.get(endpoint.endpoint_id, state.system_time),
        "CC": _communication_cost_to_node(node, endpoint, state.partial, avg_samples, metrics_cache, config),
        "SD": terminal_stats.sub_deadline[node],
    }


def _average_query_scenario_samples(
    qs_data: list[QueryScenarioData],
) -> list[_AvgQueryScenario]:
    grouped: dict[str, list[QueryScenarioData]] = defaultdict(list)
    for item in qs_data:
        grouped[item[0].query_id].append(item)

    samples: list[_AvgQueryScenario] = []
    for items in grouped.values():
        query = items[0][0]
        input_sizes: dict[str, float] = {}
        output_sizes: dict[str, float] = {}
        nodes = set(items[0][2]) | set(items[0][3])
        for node in nodes:
            input_sizes[node] = float(np.mean([row[2].get(node, 0.0) for row in items]))
            output_sizes[node] = float(np.mean([row[3].get(node, 0.0) for row in items]))
        samples.append(
            _AvgQueryScenario(
                query=query,
                input_sizes=input_sizes,
                output_sizes=output_sizes,
            )
        )
    return samples


def _precompute_terminal_stats(
    workflow: WorkflowDAG,
    node_candidates: dict[str, list[Endpoint]],
    avg_samples: list[_AvgQueryScenario],
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> _TerminalStats:
    mean_exec: dict[tuple[str, str], float] = {}
    mean_cost: dict[tuple[str, str], float] = {}
    min_exec: dict[str, float] = {}
    for node, cands in node_candidates.items():
        exec_values = []
        for ep in cands:
            mean_exec[(node, ep.endpoint_id)] = _mean_execution_latency(ep, node, avg_samples, config)
            mean_cost[(node, ep.endpoint_id)] = _avg_execution_cost(ep, node, avg_samples, config)
            exec_values.append(mean_exec[(node, ep.endpoint_id)])
        min_exec[node] = min(exec_values) if exec_values else 0.0

    min_bw = _min_positive_bandwidth(metrics_cache.network_index.values())
    latest_finish: dict[str, float] = {}
    upward_rank: dict[str, float] = {}
    sub_deadline: dict[str, float] = {}
    mean_sla = float(np.mean([sample.query.sla_sec for sample in avg_samples])) if avg_samples else 0.0
    compute_nodes = set(workflow.compute_nodes())

    def avg_transfer(src: str, dst: str) -> float:
        return _avg_transfer_time(src, dst, avg_samples, min_bw)

    def latest_finish_time(node: str) -> float:
        if node in latest_finish:
            return latest_finish[node]
        succs = [s for s in workflow.successors(node) if s in compute_nodes]
        if not succs:
            value = mean_sla
        else:
            value = max(
                latest_finish_time(succ) - min_exec.get(succ, 0.0) - avg_transfer(node, succ)
                for succ in succs
            )
        latest_finish[node] = value
        return value

    def upward_rank_time(node: str) -> float:
        if node in upward_rank:
            return upward_rank[node]
        succs = [s for s in workflow.successors(node) if s in compute_nodes]
        if not succs:
            value = min_exec.get(node, 0.0)
        else:
            value = min_exec.get(node, 0.0) + max(
                upward_rank_time(succ) + avg_transfer(node, succ) for succ in succs
            )
        upward_rank[node] = value
        return value

    for node in workflow.compute_nodes():
        lf = latest_finish_time(node)
        has_compute_succ = any(s in compute_nodes for s in workflow.successors(node))
        sub_deadline[node] = lf if not has_compute_succ else SUB_DEADLINE_EPSILON * lf
        upward_rank_time(node)

    return _TerminalStats(
        min_exec=min_exec,
        mean_exec=mean_exec,
        mean_cost=mean_cost,
        upward_rank=upward_rank,
        sub_deadline=sub_deadline,
        mean_sla=mean_sla,
    )


def _estimate_times(
    workflow: WorkflowDAG,
    node: str,
    endpoint: Endpoint,
    state: _ScheduleState,
    avg_samples: list[_AvgQueryScenario],
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> tuple[float, float]:
    data_ready = state.system_time
    for pred in workflow.predecessors(node):
        if pred not in state.node_finish:
            continue
        pred_ep = state.partial.get(pred)
        if pred_ep is None:
            data_ready = max(data_ready, state.node_finish[pred])
            continue
        transfer_finish = state.node_finish[pred]
        link = metrics_cache.network_index.get((pred_ep.endpoint_id, endpoint.endpoint_id))
        if link is not None:
            transfer_finish += float(
                np.mean(
                    [
                        network_latency(
                            link,
                            edge_transfer_size(pred, node, sample.output_sizes, sample.query),
                            1.0,
                            1.0,
                            config.ablation.enable_network_latency,
                        )
                        for sample in avg_samples
                    ]
                )
            )
        data_ready = max(data_ready, transfer_finish)

    iat = state.endpoint_available.get(endpoint.endpoint_id, state.system_time)
    aat = max(state.system_time, iat, data_ready)
    exec_time = _mean_execution_latency(endpoint, node, avg_samples, config)
    return aat, aat + exec_time


def _commit_assignment(
    *,
    workflow: WorkflowDAG,
    node: str,
    endpoint: Endpoint,
    state: _ScheduleState,
    avg_samples: list[_AvgQueryScenario],
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> None:
    _, finish = _estimate_times(
        workflow,
        node,
        endpoint,
        state,
        avg_samples,
        metrics_cache,
        config,
    )
    state.partial[node] = endpoint
    state.node_finish[node] = finish
    state.endpoint_available[endpoint.endpoint_id] = finish


def _mean_execution_latency(
    endpoint: Endpoint,
    node: str,
    avg_samples: list[_AvgQueryScenario],
    config: SolverConfig,
) -> float:
    return float(
        np.mean(
            [
                execution_latency(
                    endpoint,
                    sample.input_sizes.get(node, 0.0),
                    sample.output_sizes.get(node, 0.0),
                    1.0,
                )
                for sample in avg_samples
            ]
        )
    )


def _avg_execution_cost(
    endpoint: Endpoint,
    node: str,
    avg_samples: list[_AvgQueryScenario],
    config: SolverConfig,
) -> float:
    return float(
        np.mean(
            [
                execution_cost(
                    endpoint,
                    sample.input_sizes.get(node, 0.0),
                    sample.output_sizes.get(node, 0.0),
                    sample.query,
                )
                + storage_cost(
                    endpoint,
                    sample.input_sizes.get(node, 0.0),
                    sample.output_sizes.get(node, 0.0),
                    config.ablation.enable_storage_cost,
                )
                for sample in avg_samples
            ]
        )
    )


def _communication_cost_to_node(
    node: str,
    endpoint: Endpoint,
    partial: dict[str, Endpoint],
    avg_samples: list[_AvgQueryScenario],
    metrics_cache: _BaselineEvaluationCache,
    config: SolverConfig,
) -> float:
    total = 0.0
    count = 0
    for pred in metrics_cache.workflow.predecessors(node):
        pred_ep = partial.get(pred)
        if pred_ep is None:
            continue
        link = metrics_cache.network_index.get((pred_ep.endpoint_id, endpoint.endpoint_id))
        if link is None:
            continue
        for sample in avg_samples:
            total += network_transfer_cost(
                link,
                edge_transfer_size(pred, node, sample.output_sizes, sample.query),
                config.ablation.enable_network_cost,
            )
            count += 1
    return total / max(count, 1)


def _avg_transfer_time(
    src: str,
    dst: str,
    avg_samples: list[_AvgQueryScenario],
    min_bw: float,
) -> float:
    sizes = [
        edge_transfer_size(src, dst, sample.output_sizes, sample.query)
        for sample in avg_samples
    ]
    avg_size = float(np.mean(sizes)) if sizes else 0.0
    return avg_size / max(min_bw, 1e-9)


def _min_positive_bandwidth(links) -> float:
    values = [float(link.bandwidth_mb_per_sec) for link in links if link.bandwidth_mb_per_sec > 0]
    return min(values) if values else 1.0


def _evaluate_assignment_on_qs(
    metrics_cache: _BaselineEvaluationCache,
    assignment: dict[str, Endpoint],
    qs_data: list[QueryScenarioData],
) -> dict[str, float]:
    assign_full = dict(assignment)
    assign_full.setdefault("ClientSource", metrics_cache.virtual_map["virtual_ClientSource"])
    assign_full.setdefault("ClientSink", metrics_cache.virtual_map["virtual_ClientSink"])
    compute_assignment = {
        node: ep for node, ep in assign_full.items() if not metrics_cache.workflow.is_virtual(node)
    }
    costs: list[float] = []
    violations = 0
    total = 0
    for query, scenario, input_sizes, output_sizes in qs_data:
        cost = metrics_cache._total_cost(compute_assignment, query, scenario, input_sizes, output_sizes)
        latency = metrics_cache._critical_path_latency(
            assign_full,
            query,
            scenario,
            input_sizes,
            output_sizes,
        )
        costs.append(cost)
        violations += int(latency > query.sla_sec)
        total += 1
    return {
        "expected_cost": float(np.mean(costs)) if costs else 0.0,
        "violation_rate": violations / max(total, 1),
    }


def _sample_fitness_qs(
    qs_data: list[QueryScenarioData],
    config: SolverConfig,
    rng: random.Random,
) -> list[QueryScenarioData]:
    sample_size = _config_int(config, "mtgp_fitness_sample_size", DEFAULT_FITNESS_SAMPLE_SIZE)
    if len(qs_data) <= sample_size:
        return list(qs_data)
    indexes = sorted(rng.sample(range(len(qs_data)), sample_size))
    return [qs_data[i] for i in indexes]


def _random_individual(rng: random.Random, max_depth: int, full: bool) -> _Individual:
    return _Individual(
        task_tree=_random_tree(rng, TASK_TERMINALS, max_depth, full),
        cloud_tree=_random_tree(rng, CLOUD_TERMINALS, max_depth, full),
        instance_tree=_random_tree(rng, INSTANCE_TERMINALS, max_depth, full),
    )


def _random_tree(
    rng: random.Random,
    terminals: tuple[str, ...],
    max_depth: int,
    full: bool,
) -> _GPNode:
    if max_depth <= 1 or (not full and rng.random() < 0.35):
        if rng.random() < 0.12:
            return _GPNode(constant=rng.uniform(-1.0, 1.0))
        return _GPNode(terminal=rng.choice(terminals))
    return _GPNode(
        function=rng.choice(FUNCTIONS),
        left=_random_tree(rng, terminals, max_depth - 1, full),
        right=_random_tree(rng, terminals, max_depth - 1, full),
    )


def _tournament(
    scored: list[tuple[float, _Individual]],
    tournament_size: int,
    rng: random.Random,
) -> _Individual:
    contestants = rng.sample(scored, min(tournament_size, len(scored)))
    return min(contestants, key=lambda row: row[0])[1]


def _crossover(a: _Individual, b: _Individual, rng: random.Random) -> tuple[_Individual, _Individual]:
    decision: Decision = rng.choice(("task", "cloud", "instance"))
    if decision == "task":
        child_a_tree, child_b_tree = _swap_subtrees(a.task_tree, b.task_tree, rng)
        return (
            _Individual(child_a_tree, a.cloud_tree, a.instance_tree),
            _Individual(child_b_tree, b.cloud_tree, b.instance_tree),
        )
    if decision == "cloud":
        child_a_tree, child_b_tree = _swap_subtrees(a.cloud_tree, b.cloud_tree, rng)
        return (
            _Individual(a.task_tree, child_a_tree, a.instance_tree),
            _Individual(b.task_tree, child_b_tree, b.instance_tree),
        )
    child_a_tree, child_b_tree = _swap_subtrees(a.instance_tree, b.instance_tree, rng)
    return (
        _Individual(a.task_tree, a.cloud_tree, child_a_tree),
        _Individual(b.task_tree, b.cloud_tree, child_b_tree),
    )


def _maybe_mutate(
    individual: _Individual,
    rng: random.Random,
    mutation_rate: float,
    max_depth: int,
) -> _Individual:
    if rng.random() >= mutation_rate:
        return individual
    decision: Decision = rng.choice(("task", "cloud", "instance"))
    if decision == "task":
        return _Individual(
            _replace_random_subtree(
                individual.task_tree,
                lambda: _random_tree(rng, TASK_TERMINALS, max_depth // 2 + 1, full=False),
                rng,
            ),
            individual.cloud_tree,
            individual.instance_tree,
        )
    if decision == "cloud":
        return _Individual(
            individual.task_tree,
            _replace_random_subtree(
                individual.cloud_tree,
                lambda: _random_tree(rng, CLOUD_TERMINALS, max_depth // 2 + 1, full=False),
                rng,
            ),
            individual.instance_tree,
        )
    return _Individual(
        individual.task_tree,
        individual.cloud_tree,
        _replace_random_subtree(
            individual.instance_tree,
            lambda: _random_tree(rng, INSTANCE_TERMINALS, max_depth // 2 + 1, full=False),
            rng,
        ),
    )


def _swap_subtrees(a: _GPNode, b: _GPNode, rng: random.Random) -> tuple[_GPNode, _GPNode]:
    path_a = _random_path(a, rng)
    path_b = _random_path(b, rng)
    subtree_a = _subtree_at(a, path_a)
    subtree_b = _subtree_at(b, path_b)
    return _replace_at(a, path_a, subtree_b), _replace_at(b, path_b, subtree_a)


def _replace_random_subtree(
    tree: _GPNode,
    replacement_factory: Callable[[], _GPNode],
    rng: random.Random,
) -> _GPNode:
    return _replace_at(tree, _random_path(tree, rng), replacement_factory())


def _random_path(tree: _GPNode, rng: random.Random) -> tuple[int, ...]:
    paths = _all_paths(tree)
    return rng.choice(paths)


def _all_paths(tree: _GPNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if tree.left is not None:
        paths.extend(_all_paths(tree.left, prefix + (0,)))
    if tree.right is not None:
        paths.extend(_all_paths(tree.right, prefix + (1,)))
    return paths


def _subtree_at(tree: _GPNode, path: tuple[int, ...]) -> _GPNode:
    node = tree
    for step in path:
        node = node.left if step == 0 else node.right  # type: ignore[assignment]
        if node is None:
            raise ValueError("Invalid GP tree path")
    return node


def _replace_at(tree: _GPNode, path: tuple[int, ...], replacement: _GPNode) -> _GPNode:
    if not path:
        return replacement
    if tree.function is None:
        return tree
    if path[0] == 0:
        return _GPNode(
            function=tree.function,
            left=_replace_at(tree.left, path[1:], replacement) if tree.left else replacement,
            right=tree.right,
        )
    return _GPNode(
        function=tree.function,
        left=tree.left,
        right=_replace_at(tree.right, path[1:], replacement) if tree.right else replacement,
    )


def _clean_number(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(min(value, 1e12), -1e12)


def _config_int(config: SolverConfig, name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _config_float(config: SolverConfig, name: str, default: float) -> float:
    return float(getattr(config, name, default))
