from hpbandster.core.worker import Worker
from sampler.utils import (
    Learning_results_logger,
    suggest_out_dir_path,
)
import math
import os
import pandas as pd


class BaseWorker(Worker):
    def __init__(
        self,
        cfg,
        run_id,
        nameserver=None,
        nameserver_port=None,
        logger=None,
        host=None,
        id=None,
        timeout=None,
    ):
        super().__init__(run_id, nameserver, nameserver_port, logger, host, id, timeout)
        self.cfg = cfg

        self.sleep_interval = 1
        self.solution_id_list = []
        self.budget_thld = []
        self.step_log = {}

    def check_budget(self, budget):
        budget = int(math.ceil(budget))
        if budget not in self.budget_thld:
            self.budget_thld.append(budget)
        return budget

    def get_solution_number(self, config_id):
        solution_unique_id = "_".join(map(str, config_id))

        # Add generated solution unique id
        if solution_unique_id not in self.solution_id_list:
            self.solution_id_list.append(solution_unique_id)

        # Get generated solution id
        return self.solution_id_list.index(solution_unique_id)

    def get_generation_and_pop_id(self, config_id):
        # Generation is equal to the bracket number
        generation = config_id[0] + 1
        # pop_id is equal to the solution number of the bracket number.
        pop_id = config_id[2]

        if f"{generation}_{pop_id}" not in self.step_log:
            self.step_log[f"{generation}_{pop_id}"] = 0
        else:
            self.step_log[f"{generation}_{pop_id}"] += 1

        return generation, pop_id

    def get_trained_epochs(self, generation, pop_id):
        OUT_DIR = suggest_out_dir_path(self.cfg, generation, pop_id)
        if not os.path.exists(OUT_DIR + "output.csv"):
            trained_epochs = 0
        else:
            if generation % 3 == 1:
                if self.step_log[f"{generation}_{pop_id}"] == 0:
                    trained_epochs = 0
                elif self.step_log[f"{generation}_{pop_id}"] == 1:
                    trained_epochs = self.budget_thld[0]
                elif self.step_log[f"{generation}_{pop_id}"] == 2:
                    trained_epochs = self.budget_thld[1]
            elif generation % 3 == 2:
                if self.step_log[f"{generation}_{pop_id}"] == 0:
                    trained_epochs = 0
                if self.step_log[f"{generation}_{pop_id}"] == 1:
                    trained_epochs = self.budget_thld[1]
            elif generation % 3 == 0:
                trained_epochs = 0
            else:
                raise

        return trained_epochs

    def get_objective(self, obtained_objective_value_info):
        # The index is subtracted 1 from current_fidelity as it starts from 0.
        # Because current_fidelity starts from 1.
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
        return objective_value
