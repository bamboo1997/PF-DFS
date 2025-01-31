import glob
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from deep_neural_networks.utils import (
    copy_datasets,
    suggest_dataset_root_dir,
    suggest_resize_trainsform,
)


class DatasetLoader(Dataset):
    def __init__(self, cfg, img_file_path_list, transform=None):
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        self.image_paths = [p["img_path"] for p in img_file_path_list]
        self.image_labels = [p["class"] for p in img_file_path_list]

        if self.cfg.dataset.cache:
            self.manager = Manager()
            self.cache_img = self.manager.dict()
            print("load image")
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as e:
                for i in tqdm(list(np.arange(len(self.image_paths))), leave=False):
                    e.submit(self.pre_load_data, index=i)

    def pre_load_data(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.cache_img[index] = image

    def __getitem__(self, index):
        if self.cfg.dataset.cache:
            image = self.cache_img[index]
        else:
            image = cv2.imread(self.image_paths[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transform_output = self.transform(image=image)
            image = transform_output["image"]

        image = torch.from_numpy(image).permute(2, 0, 1)
        return {"image": image, "label": self.image_labels[index]}

    def __len__(self):
        return len(self.image_paths)


def suggest_transform(cfg):
    resize_transform = suggest_resize_trainsform(cfg)

    train_transform = A.Compose(
        resize_transform
        + [
            A.PadIfNeeded(
                min_height=cfg.dataset.image_size + 4,
                min_width=cfg.dataset.image_size + 4,
                value=(0, 0, 0),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(height=cfg.dataset.image_size, width=cfg.dataset.image_size),
            A.Affine(p=cfg.dataset.augment.hp.affine_p),
            A.HorizontalFlip(p=cfg.dataset.augment.hp.hflip_p),
            A.RandomBrightnessContrast(p=cfg.dataset.augment.hp.brightness_p),
            A.Cutout(
                p=cfg.dataset.augment.hp.cutout_p,
                num_holes=1,
                max_w_size=cfg.dataset.image_size // 4,
                max_h_size=cfg.dataset.image_size // 4,
            ),
            A.Normalize(),
        ]
    )
    val_transform = A.Compose(resize_transform + [A.Normalize(p=1.0)])
    test_transform = A.Compose(resize_transform + [A.Normalize(p=1.0)])
    return train_transform, val_transform, test_transform


def suggest_img_file_path(dataset_path, dir_path, class_names, phase):
    if len(dir_path) == 0:
        if phase in ["train", "val"]:
            raise ValueError(f"No Images of {phase} !!!")
        else:
            warnings.warn("No Images of Test !!!")
            return []
    else:
        img_file_path_list = []
        for class_index, class_name in enumerate(class_names):
            img_paths = [
                {"img_path": p, "class": class_index}
                for p in sorted(glob.glob(dataset_path + f"{phase}/{class_name}/*"))
            ]
            img_file_path_list += img_paths
        return img_file_path_list


def suggest_dataset_file_path(cfg, dataset_path):
    train_dir_list = sorted(glob.glob(dataset_path + "train/*"))
    val_dir_list = sorted(glob.glob(dataset_path + "val/*"))
    test_dir_list = sorted(glob.glob(dataset_path + "test/*"))

    class_names = [p.split("/")[-1] for p in train_dir_list]

    train_img_file_list = suggest_img_file_path(
        dataset_path, train_dir_list, class_names, "train"
    )
    val_img_file_list = suggest_img_file_path(
        dataset_path, val_dir_list, class_names, "val"
    )
    test_img_file_list = suggest_img_file_path(
        dataset_path, test_dir_list, class_names, "test"
    )

    return train_img_file_list, val_img_file_list, test_img_file_list


def suggest_dataset(cfg):
    copy_datasets(cfg)
    dataset_path = suggest_dataset_root_dir(cfg)

    train_img_file_list, val_img_file_list, test_img_file_list = (
        suggest_dataset_file_path(cfg, dataset_path)
    )

    train_transform, val_transform, test_transform = suggest_transform(cfg)

    train_dataset = DatasetLoader(cfg, train_img_file_list, train_transform)
    val_dataset = DatasetLoader(cfg, val_img_file_list, val_transform)
    test_dataset = DatasetLoader(cfg, test_img_file_list, test_transform)

    return train_dataset, val_dataset, test_dataset
