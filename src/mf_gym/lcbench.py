from yahpo_gym import BenchmarkSet, list_scenarios, local_config


class MF_bench_gym(object):
    # This implementation is constructed by referred to below notebook
    # https://github.com/slds-lmu/yahpo_gym/blob/main/yahpo_gym/notebooks/using_yahpo_gym.ipynb
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg

        local_config.init_config()
        local_config.set_data_path(self.cfg.objective.dir_path)

        # Check scenario name
        self.scenario_name = self.check_scenario_name(self.cfg.objective.name)
        # Check instance
        self.scenario_instance = self.cfg.objective.instance
        # Build benchmark
        self.bench = BenchmarkSet(
            scenario=self.scenario_name,
            instance=str(self.scenario_instance),
            multithread=False,
        )

        self.fidelity_name = self.check_fidelity_name(self.cfg.objective.fidelity_name)
        self.r_seed = self.cfg.default.r_seed
        self.target_name = self.cfg.objective.target_name

        self.cs = self.bench.get_opt_space(drop_fidelity_params=True, seed=self.r_seed)

    def check_fidelity_name(self, fidelity_name):
        fidelity_space = self.bench.get_fidelity_space()
        if fidelity_name not in fidelity_space.get_hyperparameter_names():
            raise ValueError(
                f"Please select in as follows {fidelity_space.get_hyperparameter_names()}"
            )
        self.fidelity_lower = fidelity_space.get_hyperparameter(fidelity_name).lower
        self.fidelity_upper = fidelity_space.get_hyperparameter(fidelity_name).upper
        return fidelity_name

    def check_scenario_name(self, scenario_name):
        if scenario_name not in list_scenarios():
            raise ValueError(f"Please select in as follows {list_scenarios()}")
        return scenario_name

    def get_fidelity_info(self):
        return {"min": self.fidelity_lower, "max": self.fidelity_upper}

    def __call__(self, config, fidelity_level):
        # Check incumbent fidelity level
        if not (
            self.fidelity_lower <= fidelity_level
            and fidelity_level <= self.fidelity_upper
        ):
            raise ValueError(
                f"The incumbent fidelity level is out of fidelity space [{self.fidelity_lower}, {self.fidelity_upper}]"
            )

        # If the set scenario is LCBench, add the OpenML task id to the config
        if self.scenario_name == "lcbench":
            config.update({"OpenML_task_id": str(self.scenario_instance)})
            # smoothing results between 1 and the selected fidelity level
            # return objective value, runtime, and incumbent fidelity level
            VAL_ACC = 1e-8
            VAL_CE = 1e8
            VAL_B_ACC = 1e-8
            TEST_CE = 1e8
            TEST_B_ACC = 1e-8
            lc_results = [None for _ in range(fidelity_level)]
            for e in range(1, fidelity_level + 1):
                config.update({self.fidelity_name: e})
                results = self.bench.objective_function(
                    config, seed=self.r_seed, multithread=False
                )[0]

                if results["val_accuracy"] > VAL_ACC:
                    VAL_ACC = results["val_accuracy"]
                if results["val_cross_entropy"] < VAL_CE:
                    VAL_CE = results["val_cross_entropy"]
                if results["val_balanced_accuracy"] > VAL_B_ACC:
                    VAL_B_ACC = results["val_balanced_accuracy"]
                if results["test_cross_entropy"] < TEST_CE:
                    TEST_CE = results["test_cross_entropy"]
                if results["test_balanced_accuracy"] > TEST_B_ACC:
                    TEST_B_ACC = results["test_balanced_accuracy"]
                lc_results[e - 1] = 100 - VAL_ACC
            results.update({"val_accuracy": VAL_ACC})
            results.update({"val_cross_entropy": VAL_CE})
            results.update({"val_balanced_accuracy": VAL_B_ACC})
            results.update({"test_cross_entropy": TEST_CE})
            results.update({"test_balanced_accuracy": TEST_B_ACC})

            if self.target_name == "val_accuracy":
                return {
                    "objective": results[self.target_name],
                    "cost": results["time"],
                    "fidelity_level": fidelity_level,
                    "learning_results": lc_results,
                }
            elif self.target_name == "val_error_rate":
                return {
                    "objective": 100 - results["val_accuracy"],
                    "cost": results["time"],
                    "fidelity_level": fidelity_level,
                    "learning_results": lc_results,
                }
        elif self.scenario_name == "nb301":
            VAL_ACC = 1e-8
            # get learning curve
            lc_results = [1 for _ in range(fidelity_level)]

            config.update({self.fidelity_name: fidelity_level})
            results = self.bench.objective_function(config, seed=self.r_seed)[0]
            if self.target_name == "val_accuracy":
                return {
                    "objective": results[self.target_name],
                    "cost": results["runtime"],
                    "fidelity_level": fidelity_level,
                    "learning_results": lc_results,
                }
            elif self.target_name == "val_error_rate":
                return {
                    "objective": 100 - results["val_accuracy"],
                    "cost": results["runtime"],
                    "fidelity_level": fidelity_level,
                    "learning_results": lc_results,
                }
        else:
            raise ValueError(f"The setting of {self.scenario_name} is none")
