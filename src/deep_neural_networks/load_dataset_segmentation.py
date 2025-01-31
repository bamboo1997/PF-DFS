import glob
import os
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

    def __init__(self, cfg, image_file_path_list, anno_file_path_list, transform):
        super().__init__()
        self.cfg = cfg
        self.image_file_path_list = image_file_path_list
        self.anno_file_path_list = anno_file_path_list
        self.dataset_length = len(self.anno_file_path_list)
        self.transform = transform

        # cache of dataset image
        if self.cfg.dataset.cache:
            self.manager = Manager()
            self.cache_img = self.manager.dict()
            self.cache_anno = self.manager.dict()
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as e:
                for i in tqdm(list(np.arange(self.dataset_length)), leave=False):
                    e.submit(self.pre_load_data, index=i)

    def __len__(self) -> int:
        return len(self.image_file_path_list)

    def pre_load_data(self, index):
        image = cv2.cvtColor(
            cv2.imread(self.image_file_path_list[index]), cv2.COLOR_BGR2RGB
        )
        mask = cv2.cvtColor(
            cv2.imread(self.anno_file_path_list[index]), cv2.COLOR_BGR2GRAY
        )
        self.cache_img[index] = image
        self.cache_anno[index] = mask

    def __getitem__(self, index):
        if self.cfg.dataset.cache:
            image = self.cache_img[index]
            mask = self.cache_anno[index]
        else:
            image = cv2.cvtColor(
                cv2.imread(self.image_file_path_list[index]), cv2.COLOR_BGR2RGB
            )
            mask = cv2.cvtColor(
                cv2.imread(self.anno_file_path_list[index]), cv2.COLOR_BGR2GRAY
            )

        transformed_output = self.transform(image=image, mask=mask)
        image = transformed_output["image"].astype(np.float32).transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        mask = transformed_output["mask"]
        mask = torch.from_numpy(mask).float()

        return {"image": image, "label": mask}


def suggest_dataset_file_path(dataset_path):
    train_image_file_list = sorted(glob.glob(dataset_path + "train/rgb/*"))
    train_anno_file_list = sorted(glob.glob(dataset_path + "train/label/*"))
    assert len(train_image_file_list) == len(
        train_anno_file_list
    ), "No match between number of images and number of annotations."

    val_image_file_list = sorted(glob.glob(dataset_path + "val/rgb/*"))
    val_anno_file_list = sorted(glob.glob(dataset_path + "val/label/*"))
    assert len(val_image_file_list) == len(
        val_anno_file_list
    ), "No match between number of images and number of annotations."

    test_image_file_list = sorted(glob.glob(dataset_path + "test/rgb/*"))
    test_anno_file_list = sorted(glob.glob(dataset_path + "test/label/*"))

    return (
        train_image_file_list,
        train_anno_file_list,
        val_image_file_list,
        val_anno_file_list,
        test_image_file_list,
        test_anno_file_list,
    )


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
    val_transform = A.Compose(resize_transform + [A.Normalize()])
    test_transform = A.Compose(resize_transform + [A.Normalize()])

    return train_transform, val_transform, test_transform


def suggest_dataset(cfg):
    copy_datasets(cfg)
    dataset_path = suggest_dataset_root_dir(cfg)

    (
        train_image_path_list,
        train_anno_path_list,
        val_image_path_list,
        val_anno_path_list,
        test_image_path_list,
        test_anno_path_list,
    ) = suggest_dataset_file_path(dataset_path)

    train_transform, val_transform, test_transform = suggest_transform(cfg)
    train_dataset = DatasetLoader(
        cfg, train_image_path_list, train_anno_path_list, train_transform
    )
    val_dataset = DatasetLoader(
        cfg, val_image_path_list, val_anno_path_list, val_transform
    )
    test_dataset = DatasetLoader(
        cfg, test_image_path_list, test_anno_path_list, test_transform
    )

    return train_dataset, val_dataset, test_dataset
