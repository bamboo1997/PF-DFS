import glob
import json
import os
import pickle
import shutil
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm


def suggest_host(cfg):
    return (
        f"{cfg.server.host_ip_1}."
        + f"{cfg.server.host_ip_2}."
        + f"{cfg.server.host_ip_3}."
        + f"{cfg.server.host_ip_4}"
    )


def suggest_run_id(cfg):
    if cfg.execute_config_name in ["classification", "segmentation"]:
        run_id = f"{cfg.sampler.name}_{cfg.dataset.name}_{cfg.network.name}_r_seed_{cfg.default.r_seed:04}"
    elif cfg.execute_config_name in ["lcbench", "nb301", "nb201"]:
        run_id = f"{cfg.sampler.name}_{cfg.objective.name}_{cfg.objective.instance}_r_seed_{cfg.default.r_seed:04}"
    return run_id


class PF_DFS(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def calc_regression(self, f_eval_list, current_fidelity):
        if self.cfg.sampler.regression_model == "linear":
            # calc linear regression
            p1 = f_eval_list[-1]
            p2 = f_eval_list[-2]
            a = p1 - p2
            b = p1

            y_linear = [
                a * i + b
                for i in range(1, (self.cfg.default.epochs - current_fidelity) + 1)
            ]
            y_linear = list(f_eval_list) + y_linear
            return y_linear

        elif self.cfg.sampler.regression_model == "modified_linear":
            # calc modi linear regression
            if len(set(f_eval_list)) == 1 or f_eval_list[-1] != f_eval_list[-2]:
                # calc linear regression
                p1 = f_eval_list[-1]
                p2 = f_eval_list[-2]
                a = p1 - p2
                b = p1
                y_linear_modi = [
                    a * i + b
                    for i in range(1, (self.cfg.default.epochs - current_fidelity) + 1)
                ]
                y_linear_modi = list(f_eval_list) + y_linear_modi
            else:
                # calc modified linear regression
                p1 = f_eval_list[-1]
                p2 = sorted(set(f_eval_list))[1]
                p2_index = [i for i, x in enumerate(f_eval_list) if x == p2][0]
                a = (p1 - p2) / (current_fidelity - p2_index)
                b = p1
                y_linear_modi = [
                    a * i + b
                    for i in range(1, (self.cfg.default.epochs - current_fidelity) + 1)
                ]
                y_linear_modi = list(f_eval_list) + y_linear_modi
            return y_linear_modi
        else:
            raise ValueError(
                "Please select regression model, 'linear' or 'modified_linear'."
            )

    def check_condition(
        self, previous_ranking_list, current_ranking_list, regression_ranking_list
    ):
        if regression_ranking_list is not None:
            # with regression model
            if np.all(current_ranking_list == previous_ranking_list) and np.all(
                current_ranking_list == regression_ranking_list
            ):
                return True
        else:
            # without regression model
            if np.all(current_ranking_list == previous_ranking_list):
                return True
        return False

    def select_fidelity_level(
        self,
        objective_value_list_for_pruning,
        current_fidelity,
    ):
        previous_objective_value_list = objective_value_list_for_pruning[-2]
        current_objective_value_list = objective_value_list_for_pruning[-1]

        current_rankings = np.argsort(current_objective_value_list)
        previous_rankings = np.argsort(previous_objective_value_list)

        if str(self.cfg.sampler.regression_model) != "none":
            objective_function_array = np.array(objective_value_list_for_pruning)
            regression_objective_value_list = []
            regression_curve_list = []
            for pop_id in range(len(current_rankings)):
                objective_value_list_per_pop = objective_function_array[:, pop_id]
                regression_objective_value = self.calc_regression(
                    objective_value_list_per_pop,
                    current_fidelity,
                )
                regression_curve_list.append(regression_objective_value)
                regression_objective_value_list.append(regression_objective_value[-1])
            regression_rankings = np.argsort(regression_objective_value_list)
        else:
            regression_rankings = None
            regression_curve_list = None

        return (
            self.check_condition(
                previous_rankings, current_rankings, regression_rankings
            ),
            regression_curve_list,
        )


def suggest_out_dir_path(cfg, generation, pop_id):
    if cfg.execute_config_name in ["lcbench", "nb201", "nb301"]:
        OUT_DIR = (
            f"{os.environ['SGE_LOCALDIR']}/"
            + cfg.out_dir
            + "details_log/"
            + f"generation_{generation}/"
            + f"pop_id_{pop_id}/"
        )
    else:
        OUT_DIR = (
            cfg.out_dir
            + "details_log/"
            + f"generation_{generation}/"
            + f"pop_id_{pop_id}/"
        )
    return OUT_DIR


def generate_execute_command(
    cfg,
    candidate,
    current_fidelity,
    generation,
    pop_id,
    custom_generation=None,
    custom_fidelity=None,
):
    execute_command = [
        "python",
        "src/train_one_step.py",
        cfg.execute_config_name,
    ]
    # Add override command
    execute_command += cfg.override_cmd

    # Add out_dir
    if custom_generation is None and custom_fidelity is None:
        OUT_DIR = suggest_out_dir_path(cfg, generation, pop_id)
        execute_command += [f"out_dir={OUT_DIR}"]
        obtained_objective_value_info = {
            "candidate": candidate,
            "path": OUT_DIR + "output.csv",
            "generation": generation,
            "current_fidelity": current_fidelity,
            "pop_id": pop_id,
            "sampling_id": f"{generation:03}-{current_fidelity:03}",
        }
        # Add dnn dataloader seed
        execute_command += [
            "custom_seed.value="
            + str(candidate["solution_number"] + current_fidelity + cfg.default.r_seed),
        ]
    else:
        OUT_DIR = suggest_out_dir_path(cfg, custom_generation, pop_id)
        execute_command += [f"out_dir={OUT_DIR}"]
        obtained_objective_value_info = {
            "candidate": candidate,
            "path": OUT_DIR + "output.csv",
            "generation": custom_generation,
            "current_fidelity": custom_fidelity,
            "pop_id": pop_id,
            "sampling_id": f"{generation:03}-{current_fidelity:03}",
        }
        # Add dnn dataloader seed
        execute_command += [
            "custom_seed.value="
            + str(candidate["solution_number"] + custom_fidelity + cfg.default.r_seed),
        ]
    os.makedirs(OUT_DIR, exist_ok=True)

    # Add hyperprameter configuration
    hp_commands = []
    for hp_name in list(candidate.keys())[:-1]:
        if hp_name in ["lr", "momentum", "weight_decay", "lr_decay"]:
            hp_commands.append(f"optimizer.hp.{hp_name}={candidate[hp_name]}")
        elif hp_name in ["affine_p", "hflip_p", "brightness_p", "cutout_p"]:
            hp_commands.append(f"dataset.augment.hp.{hp_name}={candidate[hp_name]}")
    execute_command += hp_commands

    return execute_command, obtained_objective_value_info


def submit_commands(para_list):
    node_id, execute_command, obtained_objective_value_info = para_list
    execute_command_str = " "
    for c in execute_command:
        execute_command_str += c + " "

    RUN = False
    # The lowest fidelity level (i.e., epochs 1)
    if not os.path.exists(obtained_objective_value_info["path"]):
        RUN = True
    # If there is no minimum fidelity level
    else:
        results_list = list(
            pd.read_csv(obtained_objective_value_info["path"])["objective"].values
        )
        if len(results_list) < obtained_objective_value_info["current_fidelity"]:
            RUN = True
    if RUN:
        subprocess.run(
            [
                "ssh",
                "-p",
                "2299",
                str(node_id),
                f"cd {os.getcwd()};"
                + "source ~/miniconda3/bin/activate pf-dfs;"
                + execute_command_str,
            ]
        )


class Learning_results_logger(object):
    def __init__(self, cfg, cs) -> None:
        self.cfg = cfg
        self.cs = cs
        self.history_X = []
        self.history_F = []
        self.learning_result = []
        self.maximum_fidelity_objective_list = []
        self.current_fidelity_objective_list = []

        if self.cfg.sampler.pruning:
            if self.cfg.default.resource == "ABCI":
                self.regression_output_dir = (
                    f"{os.environ['SGE_LOCALDIR']}/" + cfg.out_dir + "regression_log/"
                )
            else:
                self.regression_output_dir = self.cfg.out_dir + "regression_log/"
            os.makedirs(self.regression_output_dir, exist_ok=True)

    def show_results(self) -> None:
        # Save solutions
        if self.cfg.sampler.name in ["CMAES"]:
            X = pd.DataFrame(
                np.array(self.history_X),
                columns=[k for k in self.cs.get_hyperparameter_names()],
            )
            X.to_csv(self.cfg.out_dir + "history_X.csv", index=False)
            # Save objective value
            F = pd.DataFrame(np.array(self.history_F), columns=["objective"])
            F.to_csv(self.cfg.out_dir + "history_F.csv", index=False)
            print(pd.concat([X, F], axis=1))

        # Save all optimization results
        pd.DataFrame(
            np.array(self.learning_result),
            columns=["epochs", "objective", "generation", "pop_id", "sampling_id"],
        ).to_csv(self.cfg.out_dir + "history_opt.csv", index=False)

        # Compression of detailed log data
        if self.cfg.execute_config_name in ["lcbench", "nb201", "nb301"]:
            subprocess.run(
                [
                    "tar",
                    "-I",
                    "pigz",
                    "-cf",
                    "details_log.tar.gz",
                    "details_log/",
                ],
                cwd=f"{os.environ['SGE_LOCALDIR']}/" + self.cfg.out_dir,
            )
            subprocess.run(
                [
                    "cp",
                    f"{os.environ['SGE_LOCALDIR']}/"
                    + self.cfg.out_dir
                    + "details_log.tar.gz",
                    self.cfg.out_dir,
                ]
            )
        else:
            subprocess.run(
                ["tar", "-I", "pigz", "-cf", "details_log.tar.gz", "details_log/"],
                cwd=self.cfg.out_dir,
            )
            shutil.rmtree(self.cfg.out_dir + "details_log/")

        # Save the regression curve to predict the objective value of the maximum fidelity level.
        if self.cfg.sampler.pruning:
            regression_log_files = sorted(glob.glob(self.regression_output_dir + "*"))
            regression_log = {}
            for file_path in tqdm(regression_log_files, leave=False):
                generation = file_path.split("/")[-1].split("_")[1]
                fidelity = file_path.split("/")[-1].split("_")[-1].split(".")[0]
                with open(file_path, "rb") as f:
                    log = pickle.load(f)
                regression_log[f"generation_{generation}_fidelity_{fidelity}"] = log
            with open(self.cfg.out_dir + "regression_all_log.pickle", "wb") as f:
                pickle.dump(regression_log, f)
        # shutil.rmtree(self.regression_output_dir)

        # save maximum fidelity level results
        if len(self.maximum_fidelity_objective_list) > 0 and len(
            self.current_fidelity_objective_list
        ):
            with open(self.cfg.out_dir + "objective_list.pickle", "wb") as f:
                pickle.dump(
                    {
                        "current_fidelity_objective_list": self.current_fidelity_objective_list,
                        "mamximum_fidelity_objective_list": self.maximum_fidelity_objective_list,
                    },
                    f,
                )

    def save_learning_results(
        self, learned_epoch, objective_value, generation, pop_id, sampling_id
    ):
        self.learning_result.append(
            [
                learned_epoch,
                objective_value,
                generation,
                pop_id,
                sampling_id,
            ]
        )

    def save_regression_curve(
        self, regression_curve_list, generation, current_fidelity
    ):
        with open(
            self.regression_output_dir
            + f"generation_{generation}_fidelity_{current_fidelity}.pickle",
            "wb",
        ) as f:
            pickle.dump(regression_curve_list, f)

    def save_solutions(self, solutions):
        for x in solutions:
            self.history_X.append(x)

    def save_objectives(self, objective_values):
        for f in objective_values:
            self.history_F.append(float(f))

    def save_current_maximum_objective(
        self, current_objective_list, maximum_objective_list
    ):
        self.current_fidelity_objective_list.append(current_objective_list)
        self.maximum_fidelity_objective_list.append(maximum_objective_list)

    def dump_all_optimization_result(self):
        # Save all optimization results
        pd.DataFrame(
            np.array(self.learning_result),
            columns=["epochs", "objective", "generation", "pop_id", "sampling_id"],
        ).to_csv(self.cfg.out_dir + "history_opt.csv", index=False)

    def get_all_learning_results(self):
        return pd.DataFrame(
            np.array(self.learning_result),
            columns=["epochs", "objective", "generation", "pop_id", "sampling_id"],
        )
