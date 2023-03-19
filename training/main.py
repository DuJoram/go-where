import argparse
import datetime
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from fcnet64 import FCNet64
from synthgo import SynthGo
from trainer import Trainer


def train_val_split(
    dataset: Dataset,
    validation_set_size: int = None,
    shuffle=False,
) -> Tuple[Subset, Subset]:
    train_set_size = len(dataset) - validation_set_size
    indices = np.arange(len(dataset))
    if shuffle:
        indices = np.random.permutation(indices)
    train_set_indices = indices[:train_set_size]
    validation_set_indices = indices[train_set_size:]

    train_dataset = Subset(
        dataset,
        train_set_indices,
    )
    validation_dataset = Subset(
        dataset,
        validation_set_indices,
    )
    return (
        train_dataset,
        validation_dataset,
    )


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for training, evaluation and testing")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument(
        "--probability-loss-fn",
        type=str,
        choices=["bce", "mse"],
        default="bce",
        help="Loss to use for the probability",
    )
    parser.add_argument("--probability-loss-weigth", type=float, default=1, help="Weight on the probability loss")
    parser.add_argument(
        "--location-loss-fn",
        type=str,
        choices=["l1", "l2", "mse"],
        default="mse",
        help="Loss function on the location predictions",
    )
    parser.add_argument("--location-loss-weight", type=float, default=1, help="Weight on the location loss")
    parser.add_argument("--validation-interval", type=int, default=5, help="Evaluate every n epochs")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save training checkpoint every n epochs")
    parser.add_argument(
        "--load-from-checkpoint", type=str, default=None, help="Path to checkpoint, e.g. 'logs/checkpoints/001337'"
    )

    args = parser.parse_args()

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            #            torchvision.transforms.Grayscale(),
            torchvision.transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
            ),
            #            torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0, 0.3)),
            torchvision.transforms.Normalize(
                mean=0.5,
                std=0.5,
            ),
        ]
    )

    caltech256 = torchvision.datasets.Caltech256(
        root="data",
        transform=transform,
        download=True,
    )

    dataset = SynthGo(
        "SynthGo",
        grid_size=8,
        test_split=False,
        augmentation_dataset=caltech256,
        shuffle_augmentation_dataset=True,
        transform=transform,
        target_transform=lambda x: torch.FloatTensor(x).permute(2, 1, 0),
    )

    (
        train_dataset,
        validation_dataset,
    ) = train_val_split(
        dataset=dataset,
        validation_set_size=100,
    )

    test_dataset = SynthGo(
        "SynthGo",
        grid_size=8,
        test_split=True,
        augmentation_dataset=caltech256,
        shuffle_augmentation_dataset=True,
        transform=transform,
        target_transform=lambda x: torch.FloatTensor(x).permute(2, 1, 0),
    )

    log_train_images = [train_dataset[idx] for idx in range(5)]
    log_test_images = [test_dataset[idx] for idx in range(5)]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        pin_memory_device="cuda",
        # prefetch_factor=2,
        # persistent_workers=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        # num_workers=8,
        # pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        # num_workers=8,
        # pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True,
    )

    net = FCNet64(
        in_channels=3,
        out_channels=3,
    )

    trainer = Trainer(
        model=net,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        log_train_images=log_train_images,
        log_test_images=log_test_images,
        log_dir=f"logs/{timestamp}/",
        log_every_n_steps=1,
        validation_every_n_epoch=args.validation_interval,
        save_every_n_steps=args.checkpoint_interval,
        device="gpu",
    )

    if args.load_from_checkpoint is not None:
        trainer.load(args.load_from_checkpoint)

    trainer.fit(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
    )


if __name__ == "__main__":
    main()
