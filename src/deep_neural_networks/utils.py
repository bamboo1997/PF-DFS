import os
import random
import subprocess

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torchvision
from deep_neural_networks.dataset_labels import suggest_sementation_classes
import glob
from timm.scheduler import CosineLRScheduler
from deep_neural_networks.models.naive_cnn import NaiveCNN


def setup_device(cfg):
    if torch.cuda.is_available():
        device = cfg.default.device_id
        if not cfg.default.deterministic:
            torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    return device


def fixed_r_seed(cfg):
    random.seed(cfg.custom_seed.value)
    np.random.seed(cfg.custom_seed.value)
    torch.manual_seed(cfg.custom_seed.value)
    torch.cuda.manual_seed(cfg.custom_seed.value)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def suggest_resize_trainsform(cfg):
    resize_transform = []

    if cfg.dataset.letterbox:
        resize_transform.append(A.LongestMaxSize(max_size=cfg.dataset.image_size))
        resize_transform.append(
            A.PadIfNeeded(
                min_width=cfg.dataset.image_size,
                min_height=cfg.dataset.image_size,
                value=(0, 0, 0),
                border_mode=cv2.BORDER_CONSTANT,
            )
        )
    else:
        resize_transform.append(
            A.Resize(width=cfg.dataset.image_size, height=cfg.dataset.image_size)
        )

    return resize_transform


def suggest_dataset_root_dir(cfg):
    if cfg.default.resource in ["ABCI"]:
        dataset_path = f"{os.environ['SGE_LOCALDIR']}/datasets/{cfg.dataset.name}/"
    elif cfg.default.resource == "local":
        dataset_path = f"./datasets/{cfg.dataset.name}/"
    else:
        raise ValueError("Resource type is selectd from 'ABCI' or 'local'")

    return dataset_path


def copy_datasets(cfg) -> None:
    if cfg.default.resource in ["ABCI"]:
        if not os.path.exists(
            f"{os.environ['SGE_LOCALDIR']}/datasets/{cfg.dataset.name}/"
        ):
            print("Copy dataaset to SGE_LOCALDIR !!")
            os.makedirs(f"{os.environ['SGE_LOCALDIR']}/datasets/", exist_ok=True)
            copy_dataset_path = f"./datasets/{cfg.dataset.name}.tar.gz"

            # Copy dataset to local storage
            subprocess.run(
                [
                    "cp",
                    copy_dataset_path,
                    f"{os.environ['SGE_LOCALDIR']}/datasets/",
                ]
            )
            # extract dataset images
            subprocess.run(
                [f"tar -I  pigz -xf {cfg.dataset.name}.tar.gz"],
                cwd=f"{os.environ['SGE_LOCALDIR']}/datasets/",
                shell=True,
            )


def suggest_classification(cfg):
    if cfg.default.resource == "ABCI":
        numof_classes = len(
            glob.glob(
                f"{os.environ['SGE_LOCALDIR']}/datasets/{cfg.dataset.name}/train/*"
            )
        )
    elif cfg.default.resource == "local":
        numof_classes = len(glob.glob(f"./datasets/{cfg.dataset.name}/train/*"))
    else:
        raise ValueError("Resource type is selectd from 'ABCI' or 'local'.")
    numof_class = {"dataset": {"numof_classes": numof_classes}}
    return OmegaConf.merge(cfg, numof_class)


def suggest_segmentation(cfg):
    segment_labels = suggest_sementation_classes(cfg)
    numof_class = {"dataset": {"numof_classes": len(segment_labels)}}
    return OmegaConf.merge(cfg, numof_class)


def suggest_network(cfg):
    if cfg.execute_config_name == "classification":
        cfg = suggest_classification(cfg)
        if cfg.network.name == "naive_cnn":
            model = NaiveCNN(cfg)

        elif cfg.network.name == "vgg16_bn":
            model = torchvision.models.vgg16_bn(weights=None)
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.classifier[0] = nn.Linear(512 * 1 * 1, 4096)
            model.classifier[6] = nn.Linear(4096, cfg.dataset.numof_classes)
        elif cfg.network.name == "resnet18":
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT
            )
            model.fc = nn.Linear(512, cfg.dataset.numof_classes)
        elif cfg.network.name == "mobilenet_v2":
            model = torchvision.models.mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
            )
            model.classifier[1] = nn.Linear(1280, cfg.dataset.numof_classes)

    elif cfg.execute_config_name == "segmentation":
        cfg = suggest_segmentation(cfg)
        if cfg.network.name == "UNet":
            model = smp.Unet(
                encoder_name=cfg.network.encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=cfg.dataset.numof_classes,
            )
    return cfg, model


def setup_learner(cfg, model_states_path, device):

    if os.path.exists(model_states_path):
        # Load previous state
        previous_model_info = torch.load(model_states_path)
        previous_model_state = previous_model_info["model_state_dict"]
        previous_optimizer_state = previous_model_info["optimizer_state_dict"]
        previous_scheduler_state_dict = previous_model_info["scheduler_state_dict"]
        previous_learned_epoch = previous_model_info["epoch"]
    else:
        previous_learned_epoch = 0

    # Set up model
    cfg, model = suggest_network(cfg)
    model.to(device)
    if cfg.default.DP and cfg.default.n_gpus > 1:
        model = torch.nn.DataParallel(
            model, device_ids=[i for i in range(cfg.default.n_gpus)]
        )
    if os.path.exists(model_states_path):
        # Load previous model weights
        model.load_state_dict(previous_model_state)

    # Set up optimizer
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.optimizer.hp.lr,
        momentum=cfg.optimizer.hp.momentum,
        weight_decay=cfg.optimizer.hp.weight_decay,
        nesterov=True,
    )
    if os.path.exists(model_states_path):
        # Load previous optimizer state
        optimizer.load_state_dict(previous_optimizer_state)

    # Set up scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.default.epochs,
        eta_min=cfg.optimizer.hp.lr * cfg.optimizer.hp.lr_decay,
    )

    if os.path.exists(model_states_path):
        # Load previous scheduler state
        scheduler.load_state_dict(previous_scheduler_state_dict)

    return cfg, model, optimizer, scheduler, previous_learned_epoch


def suggest_loss_func(cfg, device):
    if cfg.network.loss_func.name == "CE":
        loss_func = nn.CrossEntropyLoss().to(device)
    return loss_func


def save_learner(cfg, model, optimizer, scheduler, current_epoch):
    weight_dir_path = cfg.out_dir + "weights/"
    os.makedirs(weight_dir_path, exist_ok=True)

    save_file_path = weight_dir_path + "latest_epochs.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": current_epoch,
            "cfg": cfg,
        },
        save_file_path,
    )
