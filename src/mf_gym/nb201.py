import ConfigSpace as CS
import sys
import time

from naslib import utils as naslib_util
from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from hpbandster.core.worker import Worker


class NB201(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.target_name = self.cfg.objective.target_name
        self.check_dataset()
        self.define_CS()

        # NASlib init
        sys.argv = ["main.py"]
        args = naslib_util.parse_args()
        args.config_file = "NASLib/naslib/defaults/darts_defaults.yaml"
        args.opts = []
        self.naslib_config = naslib_util.get_config_from_args(
            args=args, config_type="nas"
        )
        self.naslib_config.dataset = self.cfg.objective.instance
        self.dataset_api = naslib_util.get_dataset_api(
            self.naslib_config.search_space, self.naslib_config.dataset
        )

        self.search_space = NasBench201SearchSpace()
        self.search_space.instantiate_model = False

    def check_dataset(self):
        if self.cfg.objective.instance not in ["cifar10", "cifar100", "ImageNet16-120"]:
            raise ValueError(
                "Please select instance from [cifar10, cifar100, ImageNet16-120]"
            )

    def define_CS(self):
        self.cs = CS.ConfigurationSpace(seed=self.cfg.default.r_seed)
        # Setting fidelity (i.e., training epochs)
        self.fidelity_lower = 1
        self.fidelity_upper = 200

        # Operations: ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
        # as convert [0, 1, 2, 3, 4]
        self.cs.add_hyperparameters(
            [
                CS.CategoricalHyperparameter(
                    "edge_1", [0, 1, 2, 3, 4], default_value=0
                ),
                CS.CategoricalHyperparameter(
                    "edge_2", [0, 1, 2, 3, 4], default_value=0
                ),
                CS.CategoricalHyperparameter(
                    "edge_3", [0, 1, 2, 3, 4], default_value=0
                ),
                CS.CategoricalHyperparameter(
                    "edge_4", [0, 1, 2, 3, 4], default_value=0
                ),
                CS.CategoricalHyperparameter(
                    "edge_5", [0, 1, 2, 3, 4], default_value=0
                ),
                CS.CategoricalHyperparameter(
                    "edge_6", [0, 1, 2, 3, 4], default_value=0
                ),
            ]
        )

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

        self.search_space.set_spec([config[key] for key in config.keys()])
        # smoothing results between 1 and the selected fidelity level
        # return objective value, runtime, and incumbent fidelity level

        val_acc_raw_list = self.search_space.query(
            Metric.VAL_ACCURACY,
            self.naslib_config.dataset,
            dataset_api=self.dataset_api,
            epoch=self.fidelity_upper,
            full_lc=True,
        )
        train_time = self.search_space.query(
            Metric.TRAIN_TIME,
            self.naslib_config.dataset,
            dataset_api=self.dataset_api,
        )
        train_time = train_time * (fidelity_level / self.fidelity_upper)

        val_acc_list = []
        val_error_rate_list = []
        for epoch in range(fidelity_level):
            VAL_ACC = val_acc_raw_list[epoch]
            VAL_ERROR = 100 - VAL_ACC
            if epoch == 0:
                val_acc_list.append(VAL_ACC)
                val_error_rate_list.append(VAL_ERROR)
            else:
                if VAL_ACC > val_acc_list[-1]:
                    val_acc_list.append(VAL_ACC)
                    val_error_rate_list.append(VAL_ERROR)
                else:
                    val_acc_list.append(val_acc_list[-1])
                    val_error_rate_list.append(val_error_rate_list[-1])
        lc_results = [1 for _ in range(fidelity_level)]
        if self.target_name == "val_accuracy":
            return {
                "objective": val_acc_list[-1],
                "cost": train_time,
                "fidelity_level": fidelity_level,
                "learning_results": lc_results,
            }
        elif self.target_name == "val_error_rate":
            return {
                "objective": val_error_rate_list[-1],
                "cost": train_time,
                "fidelity_level": fidelity_level,
                "learning_results": lc_results,
            }
