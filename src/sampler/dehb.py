import os
import pickle
import random


import numpy as np
from dehb import DEHB
import math
import pandas as pd
import time

from sampler.search_space import suggest_search_space
from sampler.utils import (
    Learning_results_logger,
    generate_execute_command,
    submit_commands,
)


def run_dnns_one_step(cfg, execute_command, obtained_objective_value_info):
    # Get node id
    with open(os.environ["SGE_JOB_HOSTLIST"], "r") as f:
        available_node_list = [str(n[:-1]) for n in f.readlines()]
    node_id = available_node_list[0]
    execute_command.append(f"default.device_id={cfg.default.device_id}")
    execute_command.append("default.n_gpus=1")
    # Submit job
    submit_commands([node_id, execute_command, obtained_objective_value_info])


def run_lcbench_one_step(cfg, mf_gym, obtained_objective_value_info):
    # The lowest fidelity level (i.e., epochs 1)
    if not os.path.exists(obtained_objective_value_info["path"]):
        fidelity_level = 1
        objective_value = mf_gym(
            obtained_objective_value_info["candidate"], fidelity_level
        )["objective"]
        pd.DataFrame(np.array([objective_value]), columns=["objective"]).to_csv(
            obtained_objective_value_info["path"], index=False
        )
    # If there is no minimum fidelity level
    else:
        results_list = list(
            pd.read_csv(obtained_objective_value_info["path"])["objective"].values
        )
        if len(results_list) < obtained_objective_value_info["current_fidelity"]:
            fidelity_level = len(results_list) + 1
            objective_value = mf_gym(
                obtained_objective_value_info["candidate"], fidelity_level
            )["objective"]

            results_df_update = pd.DataFrame()
            results_df_update["objective"] = np.array(results_list + [objective_value])
            results_df_update.to_csv(obtained_objective_value_info["path"], index=False)
        else:
            pass


class DEHBSampler(object):
    def __init__(
        self,
        cfg,
    ):
        random.seed(cfg.default.r_seed)
        np.random.seed(cfg.default.r_seed)
        time.sleep(np.random.randint(1, 10))

        self.cfg = cfg
        self.cs, self.search_space = suggest_search_space(cfg)

        dimensions = len(self.cs.get_hyperparameters())
        min_budget = self.cfg.sampler.min_budget
        max_budget = self.cfg.sampler.max_budget

        self.dehb = DEHB(
            f=self.clac_objective,
            cs=self.cs,
            dimensions=dimensions,
            min_fidelity=min_budget,
            max_fidelity=max_budget,
            n_workers=1,
            seed=self.cfg.default.r_seed,
            output_path=cfg.out_dir,
            log_level="INFO",
        )

        self.generate_solution_number = 0
        self.results_logger = Learning_results_logger(cfg, self.cs)

        if self.cfg.execute_config_name in ["lcbench", "nb301"]:
            from mf_gym.lcbench import MF_bench_gym

            self.mf_gym = MF_bench_gym(cfg)
        if self.cfg.execute_config_name in ["nb201"]:
            from mf_gym.nb201 import NB201

            self.mf_gym = NB201(cfg)

    def clac_objective(self, configuration, fidelity, **kwargs):
        self.generate_solution_number += 1
        candidate = configuration.get_dictionary()
        candidate.update({"solution_number": self.generate_solution_number})
        fidelity = math.ceil(fidelity)

        s = time.time()
        for current_fidelity in range(1, fidelity + 1):
            # Generated execute command
            execute_command, obtained_objective_value_info = generate_execute_command(
                self.cfg,
                candidate,
                current_fidelity,
                self.generate_solution_number,
                self.generate_solution_number,
            )
            # Run one step of objective function
            if self.cfg.execute_config_name in ["classification", "segmentation"]:
                run_dnns_one_step(
                    self.cfg, execute_command, obtained_objective_value_info
                )
            elif self.cfg.execute_config_name in ["lcbench", "nb301", "nb201"]:
                run_lcbench_one_step(
                    self.cfg, self.mf_gym, obtained_objective_value_info
                )

            # Get objective function value
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

        res = {
            "fitness": float(objective_value),
            "cost": float(time.time() - s),
            "info": {
                "budget": int(fidelity),
                "time": float(time.time() - s),
            },
        }
        self.results_logger.dump_all_optimization_result()
        return res

    def run(self):

        trajectory, runtime, history = self.dehb.run(
            fevals=self.cfg.sampler.n_trials,
            verbose=True,
            save_intermediate=True,
        )
        self.results_logger.show_results()

        with open(self.cfg.out_dir + "trajectory.pkl", "wb") as fh:
            pickle.dump(trajectory, fh)

        with open(self.cfg.out_dir + "runtime.pkl", "wb") as fh:
            pickle.dump(runtime, fh)

        with open(self.cfg.out_dir + "history.pkl", "wb") as fh:
            pickle.dump(history, fh)
