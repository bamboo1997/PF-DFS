import os
import subprocess
from multiprocessing import Pool


def run_copy_dataset(para_list):
    cfg, node_id = para_list

    subprocess.run(
        [
            "ssh",
            "-p",
            "2299",
            str(node_id),
            f"cd {os.getcwd()};"
            + "source ~/miniconda3/bin/activate pf-dfs;"
            + f"python src/copy_dataset.py {cfg.dataset.name}",
        ]
    )


def copy_dataset(cfg):
    with open(os.environ["SGE_JOB_HOSTLIST"], "r") as f:
        available_node_list = [str(n[:-1]) for n in f.readlines()]
    para_list = [[cfg, node_id] for node_id in available_node_list]
    P = Pool(len(available_node_list))
    P.map(run_copy_dataset, para_list)
    P.close()


def suggest_sampler(cfg):
    if cfg.execute_config_name in ["classification", "segmentation"]:
        copy_dataset(cfg)

    if cfg.sampler.name == "RandomSearch":
        from sampler.randomsearch import RandomSearchSampler

        sampler = RandomSearchSampler(cfg)
    if cfg.sampler.name == "CMAES":
        from sampler.cma import CMAESSampler

        sampler = CMAESSampler(cfg)
    elif cfg.sampler.name == "PBIL":
        from sampler.pbil import PBILSampler

        sampler = PBILSampler(cfg)
    elif cfg.sampler.name == "Hyperband":
        from sampler.hyperband import HyperBandSampler

        sampler = HyperBandSampler(cfg)
    elif cfg.sampler.name == "DEHB":
        from sampler.dehb import DEHBSampler

        sampler = DEHBSampler(cfg)
    return sampler
