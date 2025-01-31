import os
import glob
import torch
import torchvision
from tqdm import tqdm
import random


def train_val():
    random.seed(234)
    train_dataset = torchvision.datasets.CIFAR100(
        root="datasets/", train=True, download=True
    )

    train_dataset_dict = {}
    for i in range(100):
        train_dataset_dict[str(i)] = []

    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        train_dataset_dict[str(label)].append(img)

    for label in tqdm(range(100)):
        numbers = list(range(500))
        random.shuffle(numbers)

        train_set = numbers[:400]
        output_dir_path = f"datasets/cifar100/train/{label}/"
        os.makedirs(output_dir_path, exist_ok=True)
        for train_id in train_set:
            train_dataset_dict[str(label)][train_id].save(
                output_dir_path + f"image_{train_id:05}.jpg"
            )

        val_set = numbers[400:]
        output_dir_path = f"datasets/cifar100/val/{label}/"
        os.makedirs(output_dir_path, exist_ok=True)
        for val_id in val_set:
            train_dataset_dict[str(label)][val_id].save(
                output_dir_path + f"image_{val_id:05}.jpg"
            )


def test():
    test_dataset = torchvision.datasets.CIFAR100(
        root="datasets/", train=False, download=True
    )
    for img, label in tqdm(test_dataset, leave=False):
        output_dir_path = f"datasets/cifar100/test/{label}/"
        os.makedirs(output_dir_path, exist_ok=True)
        img_num = len(glob.glob(output_dir_path + "*")) + 1
        img.save(output_dir_path + f"image_{img_num:05}.jpg")


if __name__ == "__main__":
    train_val()
    test()
