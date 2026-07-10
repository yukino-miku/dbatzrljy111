import math

from starlink_modeling.problem2_ga import (
    GeneticOptimizer,
    Individual,
    apply_evaluation,
    enumerate_certification_pairs,
    is_better,
    repair_individual,
)


BOUNDS = {"M": [2, 8], "N": [4, 10], "inclination_deg": [40.0, 60.0]}


def test_repair_projects_all_genes_to_legal_ranges():
    repaired = repair_individual([99, -3, 80.0, 99, -721.0, 999.0], BOUNDS)
    assert 2 <= repaired.M <= 8
    assert 4 <= repaired.N <= 10
    assert 40.0 <= repaired.inclination_deg <= 60.0
    assert 0 <= repaired.phase_factor_F < repaired.M
    assert 0.0 <= repaired.Omega0_deg < 360.0 / repaired.M
    assert 0.0 <= repaired.u0_deg < 360.0 / repaired.N


def test_deb_rule_prefers_feasible_then_smaller_feasible_objective():
    metrics = {
        "Q1_full_region": 1.0,
        "Q2_full_region": 0.97,
        "P1_space_time": 1.0,
        "P2_space_time": 0.98,
        "worst_A1": 1.0,
        "max_gap_time_min": 0.0,
        "mean_multiplicity": 2.0,
        "std_availability1": 0.0,
    }
    feasible_large = apply_evaluation(Individual(8, 10, 53.0, 1, 0.0, 0.0), metrics, "single", 86400.0)
    feasible_small = apply_evaluation(Individual(6, 8, 53.0, 1, 0.0, 0.0), metrics, "single", 86400.0)
    infeasible = apply_evaluation(
        Individual(2, 4, 53.0, 1, 0.0, 0.0),
        dict(metrics, Q1_full_region=0.99, max_gap_time_min=5.0),
        "single",
        86400.0,
    )
    assert is_better(feasible_large, infeasible, "single")
    assert is_better(feasible_small, feasible_large, "single")


def test_fixed_seed_ga_is_reproducible(tmp_path):
    config = {
        "search_bounds": BOUNDS,
        "heuristic_initialization": {
            "M": [2, 8],
            "N": [4, 10],
            "inclination_deg": [49.0, 57.0],
            "fully_random_ratio": 0.3,
        },
        "ga": {
            "population_size": 10,
            "generations": 5,
            "elite_ratio": 0.1,
            "tournament_size": 3,
            "crossover_probability": 0.85,
            "mutation_probability": 0.2,
            "inclination_mutation_sigma_deg": 1.0,
            "stagnation_generations": 3,
            "stagnation_mutation_multiplier": 2.0,
            "early_stop_generations": 0,
            "cache_decimals": 5,
        },
    }

    def evaluator(individual):
        score = min(1.0, individual.total_satellites / 48.0)
        metrics = {
            "Q1_full_region": score,
            "Q2_full_region": score,
            "P1_space_time": score,
            "P2_space_time": score,
            "worst_A1": score,
            "max_gap_time_min": 0.0 if math.isclose(score, 1.0) else 10.0,
            "mean_multiplicity": score,
            "std_availability1": 1.0 - score,
        }
        return apply_evaluation(individual, metrics, "single", 86400.0)

    first = GeneticOptimizer(config, "single", evaluator, 123, checkpoint_path=tmp_path / "a.json").run()
    second = GeneticOptimizer(config, "single", evaluator, 123, checkpoint_path=tmp_path / "b.json").run()
    assert first.best.chromosome_dict() == second.best.chromosome_dict()
    assert first.best.constraint_violation == second.best.constraint_violation


def test_certification_pair_limit_prioritizes_lower_satellite_counts():
    pairs = enumerate_certification_pairs(BOUNDS, 60, margin=2, max_pairs=4)
    selected = pairs[pairs["selected_for_search"]]
    assert len(selected) == 4
    assert (selected["total_satellites"] < 60).all()
    assert (pairs["total_satellites"] <= 62).all()
