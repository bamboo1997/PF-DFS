import math
import os
from multiprocessing import Pool

import pandas as pd

from sampler.search_space import suggest_search_space
from sampler.utils import (
    Learning_results_logger,
    generate_execute_command,
    submit_commands,
)
from sampler.dehb import run_lcbench_one_step


class RandomSearchSampler(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cs, self.search_space = suggest_search_space(cfg)

        self.generation = 1

        self.generate_solution_number = 0
        self.previous_best_candidate_list = []

        self.results_logger = Learning_results_logger(cfg, self.cs)
        self.objective_value_list_for_pruning = []

        if self.cfg.execute_config_name in ["lcbench"]:
            from mf_gym.lcbench import MF_bench_gym

            self.mf_gym = MF_bench_gym(cfg)

    def sampling_and_evaluate(self):

        candidate_list = []
        for _ in range(self.cfg.sampler.n_parallel):
            param = self.cs.sample_configuration(1).get_dictionary()
            param.update({"solution_number": self.generate_solution_number})
            self.generate_solution_number += 1
            candidate_list.append(param)

        for current_fidelity in range(1, self.cfg.default.epochs + 1):
            execute_command_list = []
            obtained_objective_value_info_list = []

            for pop_id, candidate in enumerate(candidate_list):
                execute_command, obtained_objective_value_info = (
                    generate_execute_command(
                        self.cfg, candidate, current_fidelity, self.generation, pop_id
                    )
                )
                execute_command_list.append(execute_command)
                obtained_objective_value_info_list.append(obtained_objective_value_info)

            # Evaluate
            objective_value_list = self.run_one_step_training(
                execute_command_list, obtained_objective_value_info_list
            )

        self.results_logger.save_objectives(objective_value_list)

        print(
            f"Generation: {self.generation} \t"
            + f"Consumed epochs: [{len(self.results_logger.learning_result)}/{self.max_consumed_epochs}] \t"
            + f"Best objective: {self.results_logger.get_all_learning_results()['objective'].min()}"
        )

    def run_one_step_training(
        self, execute_command_list, obtained_objective_value_info_list
    ):

        if self.cfg.execute_config_name in ["classification", "segmentation"]:
            with open(os.environ["SGE_JOB_HOSTLIST"], "r") as f:
                available_node_list = [str(n[:-1]) for n in f.readlines()]

            para_list = []
            for execute_command_index, execute_command in enumerate(
                execute_command_list
            ):
                para_list.append(
                    [
                        available_node_list[
                            execute_command_index % len(available_node_list)
                        ],
                        execute_command,
                    ]
                )

            P = Pool(len(available_node_list))
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
                pd.read_csv(obtained_objective_value_info["path"])["objective"].min()
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
        for _ in range(
            math.ceil(self.cfg.sampler.n_trials / self.cfg.sampler.n_parallel)
        ):
            self.sampling_and_evaluate()
            self.generation += 1

        self.results_logger.show_results()
