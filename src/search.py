import os
from omegaconf import DictConfig, OmegaConf
import time

from common import setup_config
from sampler.master import suggest_sampler


def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if not os.path.exists(cfg.out_dir + "finish.txt"):
        s = time.time()
        sampler = suggest_sampler(cfg)

        sampler.run()

        with open(cfg.out_dir + "finish.txt", "w") as f:
            f.write(str(time.time() - s))


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)
