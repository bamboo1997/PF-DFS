import ConfigSpace as CS


def get_vgg_configuration(cfg):
    # For HB family
    config_space = CS.ConfigurationSpace(seed=cfg.default.r_seed)
    config_space.add_hyperparameter(
        CS.Float("lr", bounds=[1e-4, 0.1], log=True, default=1e-2)
    )
    config_space.add_hyperparameter(
        CS.Float("momentum", bounds=[0.8, 1.0], log=False, default=0.9)
    )
    config_space.add_hyperparameter(
        CS.Float("weight_decay", bounds=[5e-6, 5e-2], log=True, default=5e-4)
    )
    config_space.add_hyperparameter(
        CS.Float("lr_decay", bounds=[0.0001, 0.1], log=False, default=0.01)
    )

    # For CMA-ES
    search_space = {
        "lr": {
            "Type": "continuous",
            "Range": [1, 4],
            "Wrapper": lambda x: 0.1**x,
        },
        "momentum": {
            "Type": "continuous",
            "Range": [0.8, 1.0],
            "Wrapper": lambda x: x,
        },
        "weight_decay": {
            "Type": "continuous",
            "Range": [2, 6],
            "Wrapper": lambda x: 5.0 * (0.1**x),
        },
        "lr_decay": {
            "Type": "continuous",
            "Range": [0.0001, 0.1],
            "Wrapper": lambda x: x,
        },
    }
    return config_space, search_space


def get_unet_configuration(cfg):
    # For HB family
    config_space = CS.ConfigurationSpace(seed=cfg.default.r_seed)
    config_space.add_hyperparameter(
        CS.Float("lr", bounds=[1e-4, 0.1], log=True, default=1e-2)
    )
    config_space.add_hyperparameter(
        CS.Float("momentum", bounds=[0.8, 1.0], log=False, default=0.9)
    )
    config_space.add_hyperparameter(
        CS.Float("weight_decay", bounds=[5e-6, 5e-2], log=True, default=5e-4)
    )
    config_space.add_hyperparameter(
        CS.Float("lr_decay", bounds=[0.0001, 0.1], log=False, default=0.01)
    )
    config_space.add_hyperparameter(
        CS.Float("affine_p", bounds=[0.0, 1.0], log=False, default=0.3)
    )
    config_space.add_hyperparameter(
        CS.Float("hflip_p", bounds=[0.0, 1.0], log=False, default=0.3)
    )
    config_space.add_hyperparameter(
        CS.Float("brightness_p", bounds=[0.0, 1.0], log=False, default=0.3)
    )
    config_space.add_hyperparameter(
        CS.Float("cutout_p", bounds=[0.0, 1.0], log=False, default=0.3)
    )

    # For CMA-ES
    search_space = {
        "lr": {
            "Type": "continuous",
            "Range": [1, 4],
            "Wrapper": lambda x: 0.1**x,
        },
        "momentum": {
            "Type": "continuous",
            "Range": [0.8, 1.0],
            "Wrapper": lambda x: x,
        },
        "weight_decay": {
            "Type": "continuous",
            "Range": [2, 6],
            "Wrapper": lambda x: 5.0 * (0.1**x),
        },
        "lr_decay": {
            "Type": "continuous",
            "Range": [0.0001, 0.1],
            "Wrapper": lambda x: x,
        },
        "affine_p": {
            "Type": "continuous",
            "Range": [0.0, 1.0],
            "Wrapper": lambda x: x,
        },
        "hflip_p": {
            "Type": "continuous",
            "Range": [0.0, 1.0],
            "Wrapper": lambda x: x,
        },
        "brightness_p": {
            "Type": "continuous",
            "Range": [0.0, 1.0],
            "Wrapper": lambda x: x,
        },
        "cutout_p": {
            "Type": "continuous",
            "Range": [0.0, 1.0],
            "Wrapper": lambda x: x,
        },
    }
    return config_space, search_space


def get_lcbench_configuration(cfg):
    from mf_gym.lcbench import MF_bench_gym

    mf_gym = MF_bench_gym(cfg)
    config_space = CS.ConfigurationSpace(seed=cfg.default.r_seed)
    # remove openml id
    config_space.add_hyperparameters(
        [
            mf_gym.cs.get_hyperparameter(para_name)
            for para_name in mf_gym.cs.get_hyperparameter_names()[1:]
        ]
    )

    # Raw search space info
    # batch_size, Type: UniformInteger, Range: [16, 512], Default: 91, on log-scale
    # learning_rate, Type: UniformFloat, Range: [0.00010000000000000009, 0.10000000000000002], Default: 0.0031622777, on log-scale
    # max_dropout, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
    # max_units, Type: UniformInteger, Range: [64, 1024], Default: 256, on log-scale
    # momentum, Type: UniformFloat, Range: [0.1, 0.99], Default: 0.545
    # num_layers, Type: UniformInteger, Range: [1, 5], Default: 3
    # weight_decay, Type: UniformFloat, Range: [1e-05, 0.1], Default: 0.050005
    # For CMA-ES
    search_space = {
        "batch_size": {
            "Type": "continuous",
            "Range": [4, 9],
            "Wrapper": lambda x: 2**x,
        },
        "learning_rate": {
            "Type": "continuous",
            "Range": [1, 4],
            "Wrapper": lambda x: 0.1**x,
        },
        "max_dropout": {
            "Type": "continuous",
            "Range": [0, 1.0],
            "Wrapper": lambda x: x,
        },
        "max_units": {
            "Type": "continuous",
            "Range": [6, 10],
            "Wrapper": lambda x: 2**x,
        },
        "momentum": {
            "Type": "continuous",
            "Range": [0.1, 0.99],
            "Wrapper": lambda x: x,
        },
        "num_layers": {
            "Type": "integer",
            "Range": [0, 1],
            "Wrapper": lambda x: x,
        },
        "weight_decay": {
            "Type": "continuous",
            "Range": [1e-5, 0.1],
            "Wrapper": lambda x: x,
        },
    }
    return config_space, search_space


def get_nb201_configuration(cfg):
    from mf_gym.nb201 import NB201

    mf_gym = NB201(cfg)
    config_space = mf_gym.cs

    return config_space, None


def get_nb301_configuration(cfg):
    from mf_gym.lcbench import MF_bench_gym

    mf_gym = MF_bench_gym(cfg)
    config_space = mf_gym.cs

    return config_space, None


def suggest_search_space(cfg):
    if cfg.execute_config_name == "classification":
        config_space, search_space = get_vgg_configuration(cfg)
    elif cfg.execute_config_name == "segmentation":
        config_space, search_space = get_unet_configuration(cfg)
    elif cfg.execute_config_name == "lcbench":
        config_space, search_space = get_lcbench_configuration(cfg)
    elif cfg.execute_config_name == "nb201":
        config_space, search_space = get_nb201_configuration(cfg)
    elif cfg.execute_config_name == "nb301":
        config_space, search_space = get_nb301_configuration(cfg)

    return config_space, search_space
