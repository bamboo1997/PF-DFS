import math
import os
import time

import pandas as pd
from hpbandster.core.worker import Worker

from sampler.search_space import suggest_search_space

from sampler.utils import (
    suggest_out_dir_path,
    generate_execute_command,
    Learning_results_logger,
)
from sampler.dehb import run_dnns_one_step
from worker.base_worker import BaseWorker


class DNNsWorker(BaseWorker):
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
        super().__init__(
            cfg, run_id, nameserver, nameserver_port, logger, host, id, timeout
        )
        self.cfg = cfg
        self.cs, self.search_space = suggest_search_space(cfg)

        self.results_logger = Learning_results_logger(cfg, self.cs)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        candidate = config
        budget = self.check_budget(budget)
        config_id = kwargs["config_id"]

        solution_number = self.get_solution_number(config_id)
        candidate.update({"solution_number": solution_number})

        s = time.time()

        generation, pop_id = self.get_generation_and_pop_id(config_id)

        trained_epochs = self.get_trained_epochs(generation, pop_id)

        for current_fidelity in range(trained_epochs + 1, budget + 1):
            # Generated execute command
            execute_command, obtained_objective_value_info = generate_execute_command(
                self.cfg,
                candidate,
                current_fidelity,
                generation,
                pop_id,
            )
            # Run one step of objective function
            run_dnns_one_step(self.cfg, execute_command, obtained_objective_value_info)

            # Get objective value
            objective_value = self.get_objective(obtained_objective_value_info)

        result = {
            "loss": objective_value,
            "info": {
                "budget": budget,
                "time": int(time.time() - s),
            },
        }
        self.results_logger.dump_all_optimization_result()
        return result

    @staticmethod
    def get_configspace(self):
        return self.cs
