import glob
import os
import sys

from omegaconf import OmegaConf


def setup_config():
    args = sys.argv

    config_file_name = args[1]
    config_file_path = f"./src/conf/{config_file_name}.yaml"
    if os.path.exists(config_file_path):
        cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"

    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args_list=args[2:]))
    if "out_dir" not in cfg:
        python_file_path = args[0].split("/")[-1]
        if "train" in python_file_path:
            if config_file_name == "unet":
                NETWORK = f"{cfg.network.name}_{cfg.network.encoder}/"
            else:
                NETWORK = f"{cfg.network.name}/"
            output_dir_path = (
                f"{cfg.default.dir_name}/"
                + NETWORK
                + f"{cfg.dataset.name}/"
                + f"r_seed_{cfg.default.r_seed}/"
            )

        elif "search" in python_file_path:
            if cfg.sampler.name in ["CMAES", "PBIL"]:
                SAMPLER = cfg.sampler.name + "_pruning_" + str(cfg.sampler.pruning)
            else:
                SAMPLER = cfg.sampler.name

            if config_file_name in ["classification", "segmentation"]:
                output_dir_path = (
                    f"{cfg.default.dir_name}/"
                    + f"{cfg.network.name}/"
                    + f"{cfg.dataset.name}/"
                    + f"{SAMPLER}/"
                    + f"r_seed_{cfg.default.r_seed}/"
                )
            elif config_file_name in ["lcbench", "nb201", "nb301"]:
                output_dir_path = (
                    f"{cfg.default.dir_name}/"
                    + f"{cfg.objective.name}/"
                    + f"instance_{cfg.objective.instance}/"
                    + f"{SAMPLER}/"
                    + f"r_seed_{cfg.default.r_seed}/"
                )
            else:
                raise ValueError("No config file.")
        else:
            raise ValueError("Please select python file, 'train.py or search.py'.")
        version_num = len(sorted(glob.glob(output_dir_path + "*")))
        output_dir_path += f"version_{version_num}/"
    else:
        output_dir_path = f"{cfg.out_dir}"

    os.makedirs(output_dir_path, exist_ok=True)

    out_dir_comp = {"out_dir": output_dir_path}
    cfg = OmegaConf.merge(cfg, out_dir_comp)

    config_name_comp = {"execute_config_name": config_file_name}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    config_name_comp = {"override_cmd": args[2:]}
    cfg = OmegaConf.merge(cfg, config_name_comp)

    with open(output_dir_path + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    return cfg


def update_cfg(cfg):
    with open(cfg.out_dir + "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)


def override_original_config(cfg):
    config_file_path = f"src/conf/{cfg.execute_config_name}.yaml"
    if os.path.exists(config_file_path):
        original_cfg = OmegaConf.load(config_file_path)
    else:
        raise "No YAML file !!!"
    return OmegaConf.merge(original_cfg, cfg)
