import math
import os
from multiprocessing import Pool

import ConfigSpace as CS
import numpy as np
import pandas as pd
from cmaes import CMA


from sampler.search_space import suggest_search_space
from sampler.utils import (
    Learning_results_logger,
    generate_execute_command,
    submit_commands,
    PF_DFS,
)
from sampler.dehb import run_lcbench_one_step


class CMAESSampler(object):
    def __init__(self, cfg):

        self.cfg = cfg
        self.cs, self.search_space = suggest_search_space(cfg)

        self.n_dim = len(self.search_space)

        # The search space of the CMA-ES is normalized to hypercube
        self.bounds = np.array([[0, 1] for _ in range(self.n_dim)])
        self.mean = np.array([0.5 for _ in range(self.n_dim)])
        if self.cfg.sampler.init_sigma == "small":
            self.sigma = 1 / 6
        elif self.cfg.sampler.init_sigma == "large":
            self.sigma = 0.5

        if cfg.sampler.redefine_popsize:
            self.sampler = CMA(
                mean=self.mean,
                sigma=self.sigma,
                bounds=self.bounds,
                seed=self.cfg.default.r_seed,
                population_size=self.cfg.sampler.pop_size,
            )
        else:
            self.sampler = CMA(
                mean=self.mean,
                sigma=self.sigma,
                bounds=self.bounds,
                seed=self.cfg.default.r_seed,
            )
        self.generation = 1

        self.generate_solution_number = 0
        self.previous_best_candidate_list = []

        self.results_logger = Learning_results_logger(cfg, self.cs)

        self.pf_dfs = PF_DFS(cfg)

        if self.cfg.execute_config_name in ["lcbench"]:
            from mf_gym.lcbench import MF_bench_gym

            self.mf_gym = MF_bench_gym(cfg)

    def sampling_solutions(self) -> tuple:
        org_candidates = []
        mapping_candidates = []

        for _ in range(self.sampler.population_size):
            X = self.sampler.ask()
            mapping_x = {}
            for x, hp_name in zip(X.copy(), self.cs.get_hyperparameter_names()):
                lower = self.search_space[hp_name]["Range"][0]
                upper = self.search_space[hp_name]["Range"][1]
                _x = self.search_space[hp_name]["Wrapper"](x * (upper - lower) + lower)
                if isinstance(
                    self.cs.get_hyperparameter(hp_name), CS.UniformFloatHyperparameter
                ):
                    _x = max(
                        min(_x, self.cs.get_hyperparameter(hp_name).upper),
                        self.cs.get_hyperparameter(hp_name).lower,
                    )
                    mapping_x.update({hp_name: float(_x)})
                elif isinstance(
                    self.cs.get_hyperparameter(hp_name), CS.UniformIntegerHyperparameter
                ):
                    if hp_name != "num_layers":
                        _x = max(
                            min(_x, self.cs.get_hyperparameter(hp_name).upper),
                            self.cs.get_hyperparameter(hp_name).lower,
                        )
                        mapping_x.update({hp_name: int(_x)})
                    else:
                        if 0 <= _x < 0.2:
                            mapping_x.update({hp_name: 1})
                        elif 0.2 <= _x < 0.4:
                            mapping_x.update({hp_name: 2})
                        elif 0.4 <= _x < 0.6:
                            mapping_x.update({hp_name: 3})
                        elif 0.6 <= _x < 0.8:
                            mapping_x.update({hp_name: 4})
                        elif 0.8 <= _x <= 1.0:
                            mapping_x.update({hp_name: 5})
            mapping_x.update({"solution_number": self.generate_solution_number})
            self.generate_solution_number += 1
            org_candidates.append(X)
            mapping_candidates.append(mapping_x)
        return org_candidates, mapping_candidates

    def sampling_and_evaluate(self):
        self.objective_value_list_for_pruning = []

        org_candidates, mapping_candidates = self.sampling_solutions()

        for current_fidelity in range(1, self.cfg.default.epochs + 1):
            # evalute population of current generation on objectve function
            execute_command_list = []
            obtained_objective_value_info_list = []
            # Current generation solutions
            for pop_id, candidate in enumerate(mapping_candidates):
                execute_command, obtained_objective_value_info = (
                    generate_execute_command(
                        self.cfg, candidate, current_fidelity, self.generation, pop_id
                    )
                )
                execute_command_list.append(execute_command)
                obtained_objective_value_info_list.append(obtained_objective_value_info)

            # The previous best solution unlearned to maximum fidelity.
            if len(self.previous_best_candidate_list) > 0:
                _previous_best_candidate_list_updated = []
                # Add previous best solutions that have not achieved the maximum fidelity level.
                for previous_best_candidate in self.previous_best_candidate_list:
                    # Add one level to the current fidelity level.
                    previous_best_candidate["current_fidelity"] += 1

                    execute_command, obtained_objective_value_info = (
                        generate_execute_command(
                            self.cfg,
                            previous_best_candidate["candidate"],
                            current_fidelity,
                            self.generation,
                            previous_best_candidate["pop_id"],
                            previous_best_candidate["generation"],
                            previous_best_candidate["current_fidelity"],
                        )
                    )
                    execute_command_list.append(execute_command)
                    obtained_objective_value_info_list.append(
                        obtained_objective_value_info
                    )

                    # Add update previous best solutions info.
                    # If not achieving the maximum fidelity level.
                    if (
                        previous_best_candidate["current_fidelity"]
                        != self.cfg.default.epochs
                    ):
                        _previous_best_candidate_list_updated.append(
                            previous_best_candidate
                        )
                self.previous_best_candidate_list = (
                    _previous_best_candidate_list_updated
                )

            # Evaluate
            objective_value_list = self.run_one_step_training(
                execute_command_list, obtained_objective_value_info_list
            )
            # Add learning results of the current generation
            self.objective_value_list_for_pruning.append(
                objective_value_list[: self.sampler.population_size]
            )

            # Check pruning state
            if (
                len(self.objective_value_list_for_pruning) >= 2
                and self.cfg.sampler.pruning
            ):
                # Get pruning state and regression curve.
                PRUNING, regression_curve_list = self.pf_dfs.select_fidelity_level(
                    self.objective_value_list_for_pruning,
                    current_fidelity,
                )
                # Save regression curve.
                self.results_logger.save_regression_curve(
                    regression_curve_list, self.generation, current_fidelity
                )
            else:
                PRUNING = False

            # Whether to pruning.
            if PRUNING:
                break

        solutions = []
        maxium_fidelity_objective_list = []
        current_fidelity_objective_list = []
        BEST = 1e8
        for pop_index, (org_x, objective_value) in enumerate(
            zip(org_candidates, objective_value_list[: self.sampler.population_size])
        ):
            solutions.append((org_x, objective_value))
            if BEST > objective_value:
                BEST = objective_value
                current_best_solution = {
                    "candidate": mapping_candidates[pop_index],
                    "current_fidelity": current_fidelity,
                    "pop_id": pop_index,
                    "generation": self.generation,
                }
            if self.cfg.execute_config_name in ["lcbench"]:
                current_fidelity_objective_list.append(objective_value)
                maxium_fidelity_objective_list.append(
                    self.mf_gym(mapping_candidates[pop_index], self.cfg.default.epochs)[
                        "objective"
                    ]
                )
        # Add the best solution not achieving the maximum fidelity level.
        if current_fidelity != self.cfg.default.epochs:
            self.previous_best_candidate_list.append(current_best_solution)

        self.results_logger.save_solutions(org_candidates)
        self.results_logger.save_objectives(
            objective_value_list[: self.sampler.population_size]
        )

        if self.cfg.execute_config_name in ["lcbench"]:
            self.results_logger.save_current_maximum_objective(
                current_fidelity_objective_list, maxium_fidelity_objective_list
            )

        print(
            f"Generation: {self.generation} \t"
            + f"Consumed epochs: [{len(self.results_logger.learning_result)}/{self.max_consumed_epochs}] \t"
            + f"Best objective: {self.results_logger.get_all_learning_results()['objective'].min()}"
        )
        self.results_logger.dump_all_optimization_result()
        return solutions

    def run_one_step_training(
        self, execute_command_list, obtained_objective_value_info_list
    ):

        if self.cfg.execute_config_name in ["classification", "segmentation"]:
            with open(os.environ["SGE_JOB_HOSTLIST"], "r") as f:
                available_node_list = [str(n[:-1]) for n in f.readlines()]

            para_list = []
            for execute_command_index, (
                execute_command,
                obtained_objective_value_info,
            ) in enumerate(
                zip(execute_command_list, obtained_objective_value_info_list)
            ):
                node_id = min(execute_command_index // 4, len(available_node_list) - 1)
                device_id = f"cuda:{execute_command_index % 4}"
                execute_command.append(f"default.device_id={device_id}")
                execute_command.append("default.n_gpus=1")
                para_list.append(
                    [
                        available_node_list[node_id],
                        execute_command,
                        obtained_objective_value_info,
                    ]
                )
            P = Pool(len(available_node_list) * 4)
            P.map(submit_commands, para_list)
            P.close()
        elif self.cfg.execute_config_name in ["lcbench"]:
            for obtained_objective_value_info in obtained_objective_value_info_list:
                run_lcbench_one_step(
                    self.cfg, self.mf_gym, obtained_objective_value_info
                )

        objective_value_list = []
        for obtained_objective_value_info in obtained_objective_value_info_list:
            objective_value = float(
                pd.read_csv(obtained_objective_value_info["path"])["objective"]
                .values[: int(obtained_objective_value_info["current_fidelity"])]
                .min()
            )

            self.results_logger.save_learning_results(
                1,
                float(objective_value),
                obtained_objective_value_info["generation"],
                obtained_objective_value_info["pop_id"],
                obtained_objective_value_info["sampling_id"],
            )
            objective_value_list.append(objective_value)
        return objective_value_list

    def run(self):
        self.max_consumed_epochs = self.cfg.default.epochs * self.cfg.sampler.n_trials

        if self.cfg.sampler.pruning:
            while True:
                solutions = self.sampling_and_evaluate()
                self.sampler.tell(solutions)
                self.generation += 1
                if self.max_consumed_epochs < len(self.results_logger.learning_result):
                    break
        else:
            for _ in range(
                math.ceil(self.cfg.sampler.n_trials / self.sampler.population_size)
            ):
                solutions = self.sampling_and_evaluate()
                self.sampler.tell(solutions)
                self.generation += 1

        self.results_logger.show_results()
