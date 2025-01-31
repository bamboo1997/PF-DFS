import random
import numpy as np
import math
from sampler.search_space import suggest_search_space
from sampler.utils import (
    Learning_results_logger,
    generate_execute_command,
    PF_DFS,
)
from sampler.dehb import run_lcbench_one_step
import ConfigSpace as CS
import pandas as pd


class PBILSampler(object):
    def __init__(self, cfg):
        random.seed(cfg.default.r_seed)
        np.random.seed(cfg.default.r_seed)

        self.cfg = cfg
        self.cs, self.search_space = suggest_search_space(cfg)

        self.p_vector = self.init_p_vector()

        self.generation = 1

        self.generate_solution_number = 0
        self.previous_best_candidate_list = []

        self.results_logger = Learning_results_logger(cfg, self.cs)

        self.pf_dfs = PF_DFS(cfg)

        if self.cfg.execute_config_name in ["nb301"]:
            from mf_gym.lcbench import MF_bench_gym

            self.mf_gym = MF_bench_gym(cfg)
        if self.cfg.execute_config_name in ["nb201"]:
            from mf_gym.nb201 import NB201

            self.mf_gym = NB201(cfg)

    def init_p_vector(self):
        p_vector = {}
        self.max_chrome_length = 0
        # add unconstrain hyperparameters
        for i, hp_name in enumerate(self.cs.get_all_unconditional_hyperparameters()):
            hp = self.cs.get_hyperparameter(hp_name)
            hp_name = hp.name
            hp_choices = hp.choices
            chrome_length = math.ceil(math.log2(len(hp_choices)))
            p_vector.update({hp.name: [np.array(0.5) for _ in range(chrome_length)]})
            self.max_chrome_length += chrome_length

        # add constrain hyperparameters
        for i, hp_name in enumerate(self.cs.get_all_conditional_hyperparameters()):
            hp = self.cs.get_hyperparameter(hp_name)
            hp_name = hp.name
            hp_choices = hp.choices
            chrome_length = math.ceil(math.log2(len(hp_choices)))
            p_vector.update({hp.name: [np.array(0.5) for _ in range(chrome_length)]})
            self.max_chrome_length += chrome_length
        return p_vector

    def suggest_parameter(self, hp_name):
        hp = self.cs.get_hyperparameter(hp_name)
        hp_choices = hp.choices
        while True:
            x_chrome = ""
            for i in range(len(self.p_vector[hp_name])):
                if np.random.random() > self.p_vector[hp_name][i]:
                    x_chrome += "0"
                else:
                    x_chrome += "1"
            x_chrome_decode = int(x_chrome, 2)
            if x_chrome_decode < len(hp_choices):
                break
        return x_chrome, x_chrome_decode

    def sampling(self):
        x_chrome = {}
        x_config = {}
        if self.cfg.objective.name == "nb201":
            for hp_name in self.cs.get_hyperparameter_names():
                hp = self.cs.get_hyperparameter(hp_name)
                x_value_chrome, x_value_decode = self.suggest_parameter(hp_name)
                x_chrome.update({hp_name: x_value_chrome})
                x_config.update({hp_name: hp.choices[x_value_decode]})

        elif self.cfg.objective.name == "nb301":
            # Generate unconstrain parameters
            for hp_name in self.cs.get_all_unconditional_hyperparameters():
                hp = self.cs.get_hyperparameter(hp_name)
                x_value_chrome, x_value_decode = self.suggest_parameter(hp_name)
                x_chrome.update({hp_name: x_value_chrome})
                x_config.update({hp_name: hp.choices[x_value_decode]})

            # Generate constrain parameters
            conditions = self.cs.get_conditions()
            conditional_hps = sorted(
                list(self.cs.get_all_conditional_hyperparameters())
            )

            n_conditions = dict(
                zip(
                    conditional_hps,
                    [
                        len(self.cs.get_parent_conditions_of(hp))
                        for hp in conditional_hps
                    ],
                )
            )
            conditional_hps_sorted = sorted(n_conditions, key=n_conditions.get)
            for hp_name in conditional_hps_sorted:
                conditions_to_check = np.where(
                    [
                        (
                            hp_name
                            in [child.name for child in condition.get_children()]
                            if (
                                isinstance(condition, CS.conditions.AndConjunction)
                                | isinstance(condition, CS.conditions.OrConjunction)
                            )
                            else hp_name == condition.child.name
                        )
                        for condition in conditions
                    ]
                )[0]
                checks = [
                    conditions[to_check].evaluate(
                        dict(
                            zip(
                                [
                                    parent.name
                                    for parent in conditions[to_check].get_parents()
                                ],
                                [
                                    x_config.get(parent.name)
                                    for parent in conditions[to_check].get_parents()
                                ],
                            )
                        )
                        if (
                            isinstance(
                                conditions[to_check], CS.conditions.AndConjunction
                            )
                            | isinstance(
                                conditions[to_check], CS.conditions.OrConjunction
                            )
                        )
                        else {
                            conditions[to_check].parent.name: x_config.get(
                                conditions[to_check].parent.name
                            )
                        }
                    )
                    for to_check in conditions_to_check
                ]
                if sum(checks) == len(checks):
                    hp = self.cs.get_hyperparameter(hp_name)
                    x_value_chrome, x_value_decode = self.suggest_parameter(hp_name)
                    x_chrome.update({hp_name: x_value_chrome})
                    x_config.update({hp_name: hp.choices[x_value_decode]})
        return x_chrome, x_config

    def evaluate(self):
        self.objective_value_list_for_pruning = []
        self.x_chrome_list = []
        self.candidate_list = []
        if self.cfg.sampler.resampling:
            for _ in range(self.cfg.sampler.pop_size):
                if len(self.x_chrome_list) == 0:
                    x_chrome, x_config = self.sampling()
                    x_config.update({"solution_number": self.generate_solution_number})
                    self.x_chrome_list.append(x_chrome)
                    self.candidate_list.append(x_config)
                    self.generate_solution_number += 1
                else:
                    # Check if the same individual does not exist within the generation
                    while True:
                        x_chrome, x_config = self.sampling()
                        JUDGE = True
                        for _x_chrome in self.x_chrome_list:
                            if str(x_chrome) == str(_x_chrome):
                                JUDGE = False
                        if JUDGE:
                            x_config.update(
                                {"solution_number": self.generate_solution_number}
                            )
                            self.x_chrome_list.append(x_chrome)
                            self.candidate_list.append(x_config)
                            self.generate_solution_number += 1
                            break
        else:
            for _ in range(self.cfg.sampler.pop_size):
                x_chrome, x_config = self.sampling()
                x_config.update({"solution_number": self.generate_solution_number})
                self.x_chrome_list.append(x_chrome)
                self.candidate_list.append(x_config)
                self.generate_solution_number += 1

        for current_fidelity in range(1, self.cfg.default.epochs + 1):
            # evalute population of current generation on objectve function
            execute_command_list = []
            obtained_objective_value_info_list = []
            # Current generation solutions
            for pop_id, candidate in enumerate(self.candidate_list):
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
            for obtained_objective_value_info in obtained_objective_value_info_list:
                run_lcbench_one_step(
                    self.cfg, self.mf_gym, obtained_objective_value_info
                )

            objective_value_list = []
            for obtained_objective_value_info in obtained_objective_value_info_list:
                objective_value = float(
                    pd.read_csv(obtained_objective_value_info["path"])[
                        "objective"
                    ].values[
                        int(obtained_objective_value_info["current_fidelity"] - 1),
                    ]
                )
                self.results_logger.save_learning_results(
                    1,
                    float(objective_value),
                    obtained_objective_value_info["generation"],
                    obtained_objective_value_info["pop_id"],
                    obtained_objective_value_info["sampling_id"],
                )
                objective_value_list.append(objective_value)

            # Add learning results of the current generation
            self.objective_value_list_for_pruning.append(
                objective_value_list[: self.cfg.sampler.pop_size]
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

        BEST = 1e8
        self.y_list = []
        maxium_fidelity_objective_list = []
        current_fidelity_objective_list = []
        for pop_index, (org_x, objective_value) in enumerate(
            zip(
                self.candidate_list,
                objective_value_list[: self.cfg.sampler.pop_size],
            )
        ):
            self.y_list.append(objective_value)
            if BEST > objective_value:
                BEST = objective_value
                current_best_solution = {
                    "candidate": self.candidate_list[pop_index].copy(),
                    "current_fidelity": current_fidelity,
                    "pop_id": pop_index,
                    "generation": self.generation,
                }
            current_fidelity_objective_list.append(objective_value)
            maxium_fidelity_objective_list.append(
                self.mf_gym(self.candidate_list[pop_index], self.cfg.default.epochs)[
                    "objective"
                ]
            )
        # Add the best solution not achieving the maximum fidelity level.
        if current_fidelity != self.cfg.default.epochs:
            self.previous_best_candidate_list.append(current_best_solution)

        self.results_logger.save_objectives(
            objective_value_list[: self.cfg.sampler.pop_size]
        )

        self.results_logger.save_current_maximum_objective(
            current_fidelity_objective_list, maxium_fidelity_objective_list
        )
        print(
            f"Generation: {self.generation} \t"
            + f"Consumed epochs: [{len(self.results_logger.learning_result)}/{self.max_consumed_epochs}] \t"
            + f"Best objective: {self.results_logger.get_all_learning_results()['objective'].values.astype(float).min()}"
        )
        self.results_logger.dump_all_optimization_result()

    def update(self):
        best_pop_index = self.y_list.index(min(self.y_list))
        worst_pop_index = self.y_list.index(max(self.y_list))

        BEST_POP_CHROME = self.x_chrome_list[best_pop_index]
        WORST_POP_CHROME = self.x_chrome_list[worst_pop_index]

        for hp_name in BEST_POP_CHROME.keys():
            for chrome_index, max_i in enumerate(BEST_POP_CHROME[hp_name]):
                # Positive update
                self.p_vector[hp_name][chrome_index] = (
                    self.p_vector[hp_name][chrome_index] * (1.0 - self.cfg.sampler.lr)
                    + int(max_i) * self.cfg.sampler.lr
                )
                # Negative update
                if (hp_name in BEST_POP_CHROME.keys()) and (
                    hp_name in WORST_POP_CHROME.keys()
                ):
                    if (
                        BEST_POP_CHROME[hp_name][chrome_index]
                        != WORST_POP_CHROME[hp_name][chrome_index]
                    ):
                        self.p_vector[hp_name][chrome_index] = (
                            self.p_vector[hp_name][chrome_index]
                            * (1.0 - self.cfg.sampler.negative_lr)
                            + int(max_i) * self.cfg.sampler.negative_lr
                        )
                # Mutation
                if np.random.random() < self.cfg.sampler.mutation_p:
                    self.p_vector[hp_name][chrome_index] = (
                        self.p_vector[hp_name][chrome_index]
                        * (1.0 - self.cfg.sampler.mutation_shift)
                        + np.random.randint(0, 2) * self.cfg.sampler.mutation_shift
                    )
                    if self.p_vector[hp_name][chrome_index] >= 1:
                        self.p_vector[hp_name][chrome_index] = 1
                    elif self.p_vector[hp_name][chrome_index] <= 0:
                        self.p_vector[hp_name][chrome_index] = 0
                if self.cfg.sampler.margin:
                    self.p_vector[hp_name][chrome_index] = max(
                        1 / self.max_chrome_length,
                        min(
                            self.p_vector[hp_name][chrome_index],
                            1 - (1 / self.max_chrome_length),
                        ),
                    )
        self.generation += 1

    def run(self):
        self.max_consumed_epochs = self.cfg.default.epochs * self.cfg.sampler.n_trials

        if self.cfg.sampler.pruning:
            while True:
                self.evaluate()
                self.update()
                if self.max_consumed_epochs < len(self.results_logger.learning_result):
                    break
        else:
            for _ in range(
                math.ceil(self.cfg.sampler.n_trials / self.cfg.sampler.pop_size)
            ):
                self.evaluate()
                self.update()

        self.results_logger.show_results()
