"""Mixed integer/real genetic algorithm for regional constellation design."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import cmp_to_key
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


@dataclass
class Individual:
    """Six-gene Walker chromosome and its evaluated coverage attributes."""

    M: int
    N: int
    inclination_deg: float
    phase_factor_F: int
    Omega0_deg: float
    u0_deg: float
    metrics: dict[str, Any] = field(default_factory=dict)
    constraint_violation: float = math.inf
    feasible: bool = False

    @property
    def total_satellites(self) -> int:
        return int(self.M * self.N)

    def chromosome_dict(self) -> dict[str, Any]:
        return {
            "M": int(self.M),
            "N": int(self.N),
            "inclination_deg": float(self.inclination_deg),
            "phase_factor_F": int(self.phase_factor_F),
            "Omega0_deg": float(self.Omega0_deg),
            "u0_deg": float(self.u0_deg),
        }

    def record(self) -> dict[str, Any]:
        return {
            **self.chromosome_dict(),
            "total_satellites": self.total_satellites,
            "feasible": bool(self.feasible),
            "constraint_violation": float(self.constraint_violation),
            **self.metrics,
        }


@dataclass
class GARunResult:
    best: Individual
    history: pd.DataFrame
    all_candidates: pd.DataFrame
    completed_generations: int
    seed: int


def load_problem2_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def repair_individual(
    individual: Individual | dict[str, Any] | Iterable[float],
    bounds: dict[str, Any],
    fixed_MN: tuple[int, int] | None = None,
) -> Individual:
    """Project every gene to its legal mixed-variable and symmetry range."""

    if isinstance(individual, Individual):
        values = individual.chromosome_dict()
    elif isinstance(individual, dict):
        values = dict(individual)
    else:
        values = dict(
            zip(
                ["M", "N", "inclination_deg", "phase_factor_F", "Omega0_deg", "u0_deg"],
                individual,
            )
        )

    m_min, m_max = map(int, bounds["M"])
    n_min, n_max = map(int, bounds["N"])
    i_min, i_max = map(float, bounds["inclination_deg"])
    if fixed_MN is None:
        M = int(np.clip(round(float(values["M"])), m_min, m_max))
        N = int(np.clip(round(float(values["N"])), n_min, n_max))
    else:
        M, N = map(int, fixed_MN)
        if not (m_min <= M <= m_max and n_min <= N <= n_max):
            raise ValueError("fixed_MN lies outside configured search bounds.")

    inclination_deg = float(np.clip(float(values["inclination_deg"]), i_min, i_max))
    phase_factor_F = int(round(float(values["phase_factor_F"]))) % M
    raan_period_deg = 360.0 / M
    phase_period_deg = 360.0 / N
    Omega0_deg = float(values["Omega0_deg"]) % raan_period_deg
    u0_deg = float(values["u0_deg"]) % phase_period_deg
    return Individual(
        M=M,
        N=N,
        inclination_deg=inclination_deg,
        phase_factor_F=phase_factor_F,
        Omega0_deg=Omega0_deg,
        u0_deg=u0_deg,
    )


def chromosome_cache_key(individual: Individual, decimals: int = 5) -> tuple[Any, ...]:
    return (
        individual.M,
        individual.N,
        round(individual.inclination_deg, decimals),
        individual.phase_factor_F,
        round(individual.Omega0_deg, decimals),
        round(individual.u0_deg, decimals),
    )


def constraint_violation(
    metrics: dict[str, Any], scenario: str, simulation_duration_seconds: float
) -> float:
    """Compute the transparent multi-term violation specified by the model."""

    q1 = float(metrics.get("Q1_full_region", 0.0))
    q2 = float(metrics.get("Q2_full_region", 0.0))
    p1 = float(metrics.get("P1_space_time", 0.0))
    p2 = float(metrics.get("P2_space_time", 0.0))
    worst_a1 = float(metrics.get("worst_A1", 0.0))
    max_gap_seconds = 60.0 * float(metrics.get("max_gap_time_min", 0.0))
    normalized_gap = min(1.0, max_gap_seconds / max(simulation_duration_seconds, 1.0))
    single = 100.0 * (1.0 - q1) + 10.0 * (1.0 - worst_a1) + (1.0 - p1) + normalized_gap
    if scenario == "single":
        return max(0.0, single)
    if scenario == "double":
        return max(
            0.0,
            100.0 * (1.0 - q1)
            + 100.0 * max(0.0, 0.95 - q2)
            + 10.0 * (1.0 - worst_a1)
            + (1.0 - p2)
            + normalized_gap,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def apply_evaluation(
    individual: Individual,
    metrics: dict[str, Any],
    scenario: str,
    simulation_duration_seconds: float,
    tolerance: float = 1e-12,
) -> Individual:
    evaluated = repair_individual(
        individual,
        {
            "M": [individual.M, individual.M],
            "N": [individual.N, individual.N],
            "inclination_deg": [individual.inclination_deg, individual.inclination_deg],
        },
        fixed_MN=(individual.M, individual.N),
    )
    evaluated.metrics = dict(metrics)
    q1 = float(metrics.get("Q1_full_region", 0.0))
    q2 = float(metrics.get("Q2_full_region", 0.0))
    gap = float(metrics.get("max_gap_time_min", math.inf))
    if scenario == "single":
        evaluated.feasible = bool(q1 >= 1.0 - tolerance and gap <= tolerance)
    elif scenario == "double":
        evaluated.feasible = bool(
            q1 >= 1.0 - tolerance and q2 >= 0.95 - tolerance
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    evaluated.constraint_violation = constraint_violation(
        metrics, scenario, simulation_duration_seconds
    )
    return evaluated


def _feasible_rank(individual: Individual, scenario: str) -> tuple[float, ...]:
    if scenario == "single":
        return (
            individual.total_satellites,
            -float(individual.metrics.get("P2_space_time", 0.0)),
            -float(individual.metrics.get("mean_multiplicity", 0.0)),
            float(individual.metrics.get("std_availability1", math.inf)),
        )
    return (
        individual.total_satellites,
        -float(individual.metrics.get("Q2_full_region", 0.0)),
        -float(individual.metrics.get("P2_space_time", 0.0)),
        float(individual.metrics.get("std_availability1", math.inf)),
    )


def is_better(a: Individual, b: Individual | None, scenario: str) -> bool:
    """Apply Deb's constraint-domination rule and scenario tie breakers."""

    if b is None:
        return True
    if a.feasible != b.feasible:
        return a.feasible
    if a.feasible:
        return _feasible_rank(a, scenario) < _feasible_rank(b, scenario)
    return (a.constraint_violation, a.total_satellites) < (
        b.constraint_violation,
        b.total_satellites,
    )


def sort_population(population: list[Individual], scenario: str) -> list[Individual]:
    def compare(a: Individual, b: Individual) -> int:
        if is_better(a, b, scenario):
            return -1
        if is_better(b, a, scenario):
            return 1
        return 0

    return sorted(population, key=cmp_to_key(compare))


class GeneticOptimizer:
    """Custom GA supporting mixed genes, caching, checkpoints and threads."""

    def __init__(
        self,
        config: dict[str, Any],
        scenario: str,
        evaluator: Callable[[Individual], Individual],
        seed: int,
        *,
        workers: int = 1,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        fixed_MN: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self.ga_config = config["ga"]
        self.bounds = config["search_bounds"]
        self.scenario = scenario
        self.evaluator = evaluator
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.workers = max(1, int(workers))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.resume = resume
        self.fixed_MN = fixed_MN
        self.cache: dict[tuple[Any, ...], Individual] = {}
        self.candidate_records: list[dict[str, Any]] = []

    def _random_individual(self, heuristic: bool) -> Individual:
        bounds = self.bounds
        m_min, m_max = map(int, bounds["M"])
        n_min, n_max = map(int, bounds["N"])
        i_min, i_max = map(float, bounds["inclination_deg"])
        if self.fixed_MN is not None:
            M, N = self.fixed_MN
        elif heuristic:
            h = self.config.get("heuristic_initialization", {})
            hm = h.get("M", [max(m_min, 4), min(m_max, 20)])
            hn = h.get("N", [max(n_min, 8), min(n_max, 35)])
            M = int(self.rng.integers(max(m_min, hm[0]), min(m_max, hm[1]) + 1))
            N = int(self.rng.integers(max(n_min, hn[0]), min(n_max, hn[1]) + 1))
        else:
            M = int(self.rng.integers(m_min, m_max + 1))
            N = int(self.rng.integers(n_min, n_max + 1))

        if heuristic:
            h_i = self.config.get("heuristic_initialization", {}).get(
                "inclination_deg", [max(i_min, 49.0), min(i_max, 57.0)]
            )
            inclination = float(self.rng.uniform(max(i_min, h_i[0]), min(i_max, h_i[1])))
        else:
            inclination = float(self.rng.uniform(i_min, i_max))
        values = {
            "M": M,
            "N": N,
            "inclination_deg": inclination,
            "phase_factor_F": int(self.rng.integers(0, M)),
            "Omega0_deg": float(self.rng.uniform(0.0, 360.0 / M)),
            "u0_deg": float(self.rng.uniform(0.0, 360.0 / N)),
        }
        return repair_individual(values, bounds, self.fixed_MN)

    def _initial_population(self) -> list[Individual]:
        size = int(self.ga_config["population_size"])
        random_ratio = float(
            self.config.get("heuristic_initialization", {}).get("fully_random_ratio", 0.25)
        )
        population: list[Individual] = []

        # Structured anchors expose both the single- and double-cover feasible
        # neighborhoods to a small population while random samples retain scope.
        if self.fixed_MN is None:
            m_max = int(self.bounds["M"][1])
            n_max = int(self.bounds["N"][1])
            if self.scenario == "single":
                anchor_pairs = [(38, 44), (40, 42), (42, 40), (45, 40)]
            else:
                anchor_pairs = [(48, 46), (50, 45), (55, 45), (60, 45)]
            anchor_pairs.append((m_max, n_max))
            for anchor_index, (anchor_M, anchor_N) in enumerate(anchor_pairs):
                anchor_M = int(np.clip(anchor_M, self.bounds["M"][0], m_max))
                anchor_N = int(np.clip(anchor_N, self.bounds["N"][0], n_max))
                population.append(
                    repair_individual(
                        {
                            "M": anchor_M,
                            "N": anchor_N,
                            "inclination_deg": 53.5 + 0.25 * (anchor_index % 3),
                            "phase_factor_F": 1,
                            "Omega0_deg": 0.0,
                            "u0_deg": 0.0,
                        },
                        self.bounds,
                    )
                )
        while len(population) < size:
            fully_random = self.rng.random() < random_ratio
            population.append(self._random_individual(heuristic=not fully_random))
        return population[:size]

    def _evaluate_population(
        self, population: list[Individual], generation: int
    ) -> list[Individual]:
        decimals = int(self.ga_config.get("cache_decimals", 5))
        missing: dict[tuple[Any, ...], Individual] = {}
        for individual in population:
            key = chromosome_cache_key(individual, decimals)
            if key not in self.cache:
                missing.setdefault(key, individual)

        if missing:
            items = list(missing.items())
            if self.workers > 1 and len(items) > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    evaluated_values = list(executor.map(self.evaluator, [item[1] for item in items]))
            else:
                evaluated_values = [self.evaluator(item[1]) for item in items]
            for (key, _), evaluated in zip(items, evaluated_values):
                self.cache[key] = evaluated
                record = evaluated.record()
                record.update({"generation_evaluated": generation, "seed": self.seed})
                self.candidate_records.append(record)

        evaluated_population: list[Individual] = []
        for individual in population:
            cached = self.cache[chromosome_cache_key(individual, decimals)]
            clone = Individual(**cached.chromosome_dict())
            clone.metrics = dict(cached.metrics)
            clone.constraint_violation = cached.constraint_violation
            clone.feasible = cached.feasible
            evaluated_population.append(clone)
        return evaluated_population

    def _tournament(self, population: list[Individual]) -> Individual:
        size = min(int(self.ga_config.get("tournament_size", 3)), len(population))
        indices = self.rng.choice(len(population), size=size, replace=False)
        winner = population[int(indices[0])]
        for index in indices[1:]:
            challenger = population[int(index)]
            if is_better(challenger, winner, self.scenario):
                winner = challenger
        return winner

    def _crossover(self, a: Individual, b: Individual) -> Individual:
        if self.rng.random() >= float(self.ga_config["crossover_probability"]):
            return repair_individual(a, self.bounds, self.fixed_MN)
        blend = float(self.rng.uniform(0.0, 1.0))
        values = {
            "M": a.M if self.rng.random() < 0.5 else b.M,
            "N": a.N if self.rng.random() < 0.5 else b.N,
            "inclination_deg": blend * a.inclination_deg + (1.0 - blend) * b.inclination_deg,
            "phase_factor_F": a.phase_factor_F if self.rng.random() < 0.5 else b.phase_factor_F,
            "Omega0_deg": blend * a.Omega0_deg + (1.0 - blend) * b.Omega0_deg,
            "u0_deg": blend * a.u0_deg + (1.0 - blend) * b.u0_deg,
        }
        return repair_individual(values, self.bounds, self.fixed_MN)

    def _mutate(
        self, individual: Individual, progress: float, probability_multiplier: float
    ) -> Individual:
        probability = min(
            1.0,
            float(self.ga_config["mutation_probability"]) * probability_multiplier,
        )
        values = individual.chromosome_dict()
        scale = max(0.15, 1.0 - progress)
        if self.fixed_MN is None:
            for gene in ("M", "N"):
                if self.rng.random() < probability:
                    if self.rng.random() < 0.2:
                        low, high = map(int, self.bounds[gene])
                        values[gene] = int(self.rng.integers(low, high + 1))
                    else:
                        values[gene] += int(self.rng.choice([-2, -1, 1, 2]))
        if self.rng.random() < probability:
            values["phase_factor_F"] += int(self.rng.choice([-2, -1, 1, 2]))
        if self.rng.random() < probability:
            sigma = float(self.ga_config.get("inclination_mutation_sigma_deg", 1.5)) * scale
            values["inclination_deg"] += float(self.rng.normal(0.0, sigma))
        if self.rng.random() < probability:
            values["Omega0_deg"] += float(self.rng.normal(0.0, 2.0 * scale))
        if self.rng.random() < probability:
            values["u0_deg"] += float(self.rng.normal(0.0, 2.0 * scale))
        return repair_individual(values, self.bounds, self.fixed_MN)

    def _save_checkpoint(
        self,
        generation: int,
        population: list[Individual],
        global_best: Individual,
        history: list[dict[str, Any]],
        stagnation: int,
    ) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generation": generation,
            "population": [individual.record() for individual in population],
            "global_best": global_best.record(),
            "history": history,
            "stagnation": stagnation,
            "seed": self.seed,
            "scenario": self.scenario,
            "fixed_MN": list(self.fixed_MN) if self.fixed_MN else None,
            "rng_state": self.rng.bit_generator.state,
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.checkpoint_path)

    @staticmethod
    def _individual_from_record(record: dict[str, Any]) -> Individual:
        individual = Individual(
            M=int(record["M"]),
            N=int(record["N"]),
            inclination_deg=float(record["inclination_deg"]),
            phase_factor_F=int(record["phase_factor_F"]),
            Omega0_deg=float(record["Omega0_deg"]),
            u0_deg=float(record["u0_deg"]),
        )
        standard_keys = set(individual.record()) - set(individual.metrics)
        individual.metrics = {
            key: value
            for key, value in record.items()
            if key not in standard_keys and key not in {"feasible", "constraint_violation"}
        }
        individual.feasible = bool(record.get("feasible", False))
        individual.constraint_violation = float(record.get("constraint_violation", math.inf))
        return individual

    def _load_checkpoint(self) -> tuple[int, list[Individual], Individual, list[dict[str, Any]], int] | None:
        if not self.resume or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return None
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("seed") != self.seed or payload.get("scenario") != self.scenario:
            raise ValueError("Checkpoint seed/scenario does not match the requested run.")
        population = [self._individual_from_record(row) for row in payload["population"]]
        global_best = self._individual_from_record(payload["global_best"])
        self.rng.bit_generator.state = payload["rng_state"]
        return (
            int(payload["generation"]) + 1,
            population,
            global_best,
            list(payload.get("history", [])),
            int(payload.get("stagnation", 0)),
        )

    def run(self) -> GARunResult:
        generations = int(self.ga_config["generations"])
        population_size = int(self.ga_config["population_size"])
        elite_count = max(1, int(round(population_size * float(self.ga_config["elite_ratio"]))))
        checkpoint = self._load_checkpoint()
        if checkpoint is None:
            start_generation = 0
            population = self._evaluate_population(self._initial_population(), generation=0)
            global_best = sort_population(population, self.scenario)[0]
            history: list[dict[str, Any]] = []
            stagnation = 0
        else:
            start_generation, population, global_best, history, stagnation = checkpoint
            for cached in population:
                self.cache[chromosome_cache_key(cached)] = cached

        completed_generation = max(0, start_generation - 1)
        try:
            for generation in range(start_generation, generations):
                ranked = sort_population(population, self.scenario)
                current_best = ranked[0]
                if is_better(current_best, global_best, self.scenario):
                    global_best = current_best
                    stagnation = 0
                else:
                    stagnation += 1

                feasible_count = sum(individual.feasible for individual in population)
                history.append(
                    {
                        "generation": generation,
                        "seed": self.seed,
                        "best_total_satellites": current_best.total_satellites,
                        "global_best_total_satellites": global_best.total_satellites,
                        "best_constraint_violation": current_best.constraint_violation,
                        "global_best_constraint_violation": global_best.constraint_violation,
                        "feasible_count": feasible_count,
                        "mean_constraint_violation": float(
                            np.mean([item.constraint_violation for item in population])
                        ),
                        "mean_total_satellites": float(
                            np.mean([item.total_satellites for item in population])
                        ),
                        "min_total_satellites": min(item.total_satellites for item in population),
                        "max_total_satellites": max(item.total_satellites for item in population),
                    }
                )

                mutation_multiplier = (
                    float(self.ga_config.get("stagnation_mutation_multiplier", 2.0))
                    if stagnation >= int(self.ga_config.get("stagnation_generations", 20))
                    else 1.0
                )
                next_population = [
                    repair_individual(item, self.bounds, self.fixed_MN)
                    for item in ranked[:elite_count]
                ]
                while len(next_population) < population_size:
                    parent_a = self._tournament(population)
                    parent_b = self._tournament(population)
                    child = self._crossover(parent_a, parent_b)
                    child = self._mutate(
                        child,
                        progress=(generation + 1) / max(generations, 1),
                        probability_multiplier=mutation_multiplier,
                    )
                    next_population.append(child)
                population = self._evaluate_population(next_population, generation + 1)
                completed_generation = generation
                self._save_checkpoint(generation, population, global_best, history, stagnation)

                early_stop = int(self.ga_config.get("early_stop_generations", 0))
                if early_stop > 0 and stagnation >= early_stop:
                    break
        except KeyboardInterrupt:
            self._save_checkpoint(
                completed_generation, population, global_best, history, stagnation
            )
            raise

        final_best = sort_population(population + [global_best], self.scenario)[0]
        if is_better(final_best, global_best, self.scenario):
            global_best = final_best
        return GARunResult(
            best=global_best,
            history=pd.DataFrame(history),
            all_candidates=pd.DataFrame(self.candidate_records),
            completed_generations=completed_generation + 1,
            seed=self.seed,
        )


def run_multiseed_ga(
    config: dict[str, Any],
    scenario: str,
    evaluator_factory: Callable[[int], Callable[[Individual], Individual]],
    base_seed: int,
    output_dir: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
    fixed_MN: tuple[int, int] | None = None,
    seed_count: int | None = None,
) -> GARunResult:
    """Run independent seeded searches and return the best combined result."""

    count = int(seed_count or config["ga"].get("random_seed_count", 1))
    output_dir = Path(output_dir)
    results: list[GARunResult] = []
    for offset in range(count):
        seed = int(base_seed + offset)
        suffix = f"_{fixed_MN[0]}x{fixed_MN[1]}" if fixed_MN else ""
        checkpoint = output_dir / "checkpoints" / f"{scenario}{suffix}_seed_{seed}.json"
        optimizer = GeneticOptimizer(
            config,
            scenario,
            evaluator_factory(seed),
            seed,
            workers=workers,
            checkpoint_path=checkpoint,
            resume=resume,
            fixed_MN=fixed_MN,
        )
        results.append(optimizer.run())

    best_result = results[0]
    for result in results[1:]:
        if is_better(result.best, best_result.best, scenario):
            best_result = result
    return GARunResult(
        best=best_result.best,
        history=pd.concat([result.history for result in results], ignore_index=True),
        all_candidates=pd.concat(
            [result.all_candidates for result in results], ignore_index=True
        ),
        completed_generations=sum(result.completed_generations for result in results),
        seed=best_result.seed,
    )


def enumerate_certification_pairs(
    bounds: dict[str, Any],
    best_total_satellites: int,
    margin: int,
    max_pairs: int | None,
) -> pd.DataFrame:
    """List fixed (M,N) pairs for reverse lower-satellite certification."""

    rows = []
    for M in range(int(bounds["M"][0]), int(bounds["M"][1]) + 1):
        for N in range(int(bounds["N"][0]), int(bounds["N"][1]) + 1):
            total = M * N
            if total <= best_total_satellites + margin:
                rows.append({"M": M, "N": N, "total_satellites": total})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["distance_from_best"] = abs(table["total_satellites"] - best_total_satellites)
    table["certification_priority"] = np.where(
        table["total_satellites"] < best_total_satellites, 0, 1
    )
    table = table.sort_values(
        ["certification_priority", "distance_from_best", "total_satellites", "M"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    table["selected_for_search"] = True
    if max_pairs is not None and len(table) > max_pairs:
        table.loc[max_pairs:, "selected_for_search"] = False
    return table


def local_coordinate_refinement(
    candidate: Individual,
    evaluator: Callable[[Individual], Individual],
    bounds: dict[str, Any],
    scenario: str,
    config: dict[str, Any],
) -> tuple[Individual, pd.DataFrame]:
    """Refine i, F, Omega0 and u0 near a GA candidate."""

    best = candidate
    records = [dict(best.record(), refinement_iteration=0)]
    max_iterations = int(config.get("max_iterations", 2))
    i_step = float(config.get("inclination_step_deg", 0.25))
    angle_step = float(config.get("angle_step_deg", 0.5))
    for iteration in range(1, max_iterations + 1):
        neighbors: list[Individual] = []
        for delta in (-i_step, i_step):
            values = best.chromosome_dict()
            values["inclination_deg"] += delta
            neighbors.append(repair_individual(values, bounds, (best.M, best.N)))
        for delta in (-1, 1):
            values = best.chromosome_dict()
            values["phase_factor_F"] += delta
            neighbors.append(repair_individual(values, bounds, (best.M, best.N)))
        for gene in ("Omega0_deg", "u0_deg"):
            for delta in (-angle_step, angle_step):
                values = best.chromosome_dict()
                values[gene] += delta
                neighbors.append(repair_individual(values, bounds, (best.M, best.N)))
        improved = False
        for neighbor in neighbors:
            evaluated = evaluator(neighbor)
            records.append(dict(evaluated.record(), refinement_iteration=iteration))
            if is_better(evaluated, best, scenario):
                best = evaluated
                improved = True
        if not improved:
            break
    return best, pd.DataFrame(records)
