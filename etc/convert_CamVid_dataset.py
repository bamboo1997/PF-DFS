import glob
import os
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


def rgb_to_index(img, path):
    color2index = {
        (0, 0, 0): 0,
        (128, 128, 128): 1,
        (0, 128, 64): 2,
        (128, 0, 0): 2,
        (64, 192, 0): 2,
        (64, 0, 64): 2,
        (192, 0, 128): 2,
        (192, 192, 128): 3,
        (0, 0, 64): 3,
        (128, 64, 128): 4,
        (128, 0, 192): 4,
        (192, 0, 64): 4,
        (0, 0, 192): 5,
        (64, 192, 128): 5,
        (128, 128, 192): 5,
        (128, 128, 0): 6,
        (192, 192, 0): 6,
        (192, 128, 128): 7,
        (128, 128, 64): 7,
        (0, 64, 64): 7,
        (64, 64, 128): 8,
        (64, 0, 128): 9,
        (64, 128, 192): 9,
        (192, 128, 192): 9,
        (192, 64, 128): 9,
        (128, 64, 64): 9,
        (64, 64, 0): 10,
        (192, 128, 64): 10,
        (64, 0, 192): 10,
        (64, 128, 64): 10,
        (0, 128, 192): 11,
        (192, 0, 192): 11,
    }
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    bar = tqdm(total=width * height, leave=False)
    for w in range(width):
        for h in range(height):
            b, g, r = img[h, w, :]
            if (r, g, b) in color2index.keys():
                m_lable[h, w, :] = color2index[(r, g, b)]
            else:
                m_lable[h, w, :] = 0
            bar.update(1)
    bar.close()
    return m_lable


def run_parallel_segmentation(para_list):
    output_img_dir_path, output_label_dir_path, img_path, anno_img_path = para_list

    subprocess.run(["cp", img_path, output_img_dir_path + img_path.split("/")[-1]])
    subprocess.run(
        ["cp", anno_img_path, output_label_dir_path + anno_img_path.split("/")[-1]]
    )


def submit_segmentation(output_dir_path, img_paths, anno_img_paths):
    output_img_dir_path = output_dir_path + "rgb/"
    os.makedirs(output_img_dir_path, exist_ok=True)
    output_label_dir_path = output_dir_path + "label/"
    os.makedirs(output_label_dir_path, exist_ok=True)

    para_list = [
        [output_img_dir_path, output_label_dir_path, img_path, anno_img_path]
        for img_path, anno_img_path in zip(img_paths, anno_img_paths)
    ]
    p = Pool(os.cpu_count() * 2)
    p.map(run_parallel_segmentation, para_list)


if __name__ == "__main__":
    input_dir_path = "./datasets/CamVid_org/"

    for phase in ["train", "val", "test"]:
        output_dir_path = f"./datasets/CamVid/{phase}/"
        os.makedirs(output_dir_path, exist_ok=True)

        img_file_path_list = sorted(glob.glob(input_dir_path + f"{phase}/*"))
        anno_file_path_list = sorted(glob.glob(input_dir_path + f"{phase}annot/*"))

        submit_segmentation(output_dir_path, img_file_path_list, anno_file_path_list)
