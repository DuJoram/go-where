import glob
import json
import os
from math import floor
from typing import Tuple, Callable

import numpy as np
import torch
import torchvision.transforms.functional
import tqdm
from PIL import Image
from torch.utils.data import Dataset


class SynthGo(Dataset):
    def __init__(
        self,
        dataset_path: str,
        test_split: bool = False,
        grid_size: int = 8,
        transform: Callable = None,
        target_transform: Callable = None,
        augmentation_dataset: Dataset = None,
        shuffle_augmentation_dataset: bool = True,
    ):
        self._dataset_path = dataset_path
        self._grid_size = grid_size
        self._transform = transform
        self._target_transform = target_transform

        train_processed_path = os.path.join(dataset_path, "processed", "train")
        test_processed_path = os.path.join(dataset_path, "processed", "test")

        os.makedirs(train_processed_path, exist_ok=True)
        os.makedirs(test_processed_path, exist_ok=True)

        self._size = len(
            glob.glob(
                os.path.join(self._dataset_path, "test" if test_split else "train")
                + "/*.png"
            )
        )
        self._processed_path = (
            test_processed_path if test_split else train_processed_path
        )

        self.preprocess(
            augmentation_dataset=augmentation_dataset,
            shuffle_augmentation_dataset=shuffle_augmentation_dataset,
        )

    def image_coord_to_grid(
        self, image_x: float, image_y: float
    ) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        cell_x = min(self._grid_size - 1, int(floor(image_x * self._grid_size)))
        cell_y = min(self._grid_size - 1, int(floor(image_y * self._grid_size)))

        grid_x = image_x * self._grid_size - cell_x
        grid_y = image_y * self._grid_size - cell_y
        return (cell_x, cell_y), (grid_x, grid_y)

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int):
        sample = np.load(os.path.join(self._processed_path, f"{idx:04d}.npz"))
        image = sample["image"]
        target = sample["target"]

        image = torch.FloatTensor(image).permute(2, 0, 1) / 127.5 - 1
        target = torch.FloatTensor(target).permute(2, 0, 1)
        return image, target

    def preprocess(
        self,
        augmentation_dataset: Dataset = None,
        shuffle_augmentation_dataset: bool = False,
    ):
        for split in ["train", "test"]:
            data_path = os.path.join(self._dataset_path, split)
            size = len(glob.glob(data_path + "/*.png"))

            augmentation_dataset_indices = None
            if augmentation_dataset is not None:
                augmentation_dataset_indices = np.arange(len(augmentation_dataset))
                if shuffle_augmentation_dataset:
                    augmentation_dataset_indices = np.random.permutation(
                        augmentation_dataset_indices
                    )

            for idx in tqdm.trange(size):
                image_path = os.path.join(data_path, f"{idx:04d}_image.png")
                target_path = os.path.join(data_path, f"{idx:04d}_label.json")

                processed_file = os.path.join(
                    self._dataset_path, "processed", split, f"{idx:04d}.npz"
                )

                if os.path.exists(processed_file):
                    continue

                image = Image.open(image_path).convert("RGBA")
                mask = torch.FloatTensor(np.array(image))[:, :, -1] / 255
                image = image.convert("RGB")

                with open(target_path) as f:
                    corners = json.load(f)["corners"]

                target = np.zeros((self._grid_size, self._grid_size, 3))
                for x, y in corners:
                    (cell_x, cell_y), (grid_x, grid_y) = self.image_coord_to_grid(
                        image_x=x, image_y=(1 - y)
                    )
                    target[cell_y, cell_x] = [1, grid_y, grid_x]

                if augmentation_dataset is not None:
                    background_image = augmentation_dataset[
                        augmentation_dataset_indices[idx]
                    ][0]
                    background_image_squared = (
                        torchvision.transforms.functional.center_crop(
                            background_image, min(background_image.shape[1:])
                        )
                    )
                    background_image_resized = torchvision.transforms.functional.resize(
                        background_image_squared, image.size[-1]
                    )

                    mask = mask[:, :, None]
                    image = np.uint8(
                        mask * np.array(image)
                        + (1 - mask)
                        * (
                            background_image_resized.permute(1, 2, 0).numpy() * 127.5
                            + 127.5
                        )
                    )
                    np.savez(processed_file, image=image, target=target)
