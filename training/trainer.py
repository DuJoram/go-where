import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Tuple, Union

import cv2 as cv
import torch
import torchvision
import torchvision.utils
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def target_to_image_coords(
    image_shape: torch.Size,
    cell_size: int,
    cell_x: Union[int, torch.Tensor],
    cell_y: Union[int, torch.Tensor],
    target_coords: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_x = torch.clip(
        torch.round(cell_x * cell_size + target_coords[1, cell_y, cell_x] * cell_size),
        0,
        image_shape[2],
    ).int()
    image_y = torch.clip(
        torch.round(cell_y * cell_size + target_coords[0, cell_y, cell_x] * cell_size),
        0,
        image_shape[1],
    ).int()
    return image_x, image_y


@dataclass
class HParams:
    cell_size: int
    learning_rate: float
    loss_p: str
    loss_p_weight: float
    loss_loc: str
    loss_loc_weight: float
    max_epochs: int
    validation_every_n_epoch: int
    save_every_n_steps: int
    global_step: int


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_p: str = "bce",
        loss_p_weight: float = 1,
        loss_loc: str = "mse",
        loss_loc_weight: float = 1,
        learning_rate=1e-3,
        max_epochs: int = 100,
        cell_size: int = 64,
        log_train_images: List[Tuple[torch.FloatTensor]] = None,
        log_test_images: List[Tuple[torch.FloatTensor]] = None,
        log_dir: str = "logs/",
        log_every_n_steps: int = 1,
        validation_every_n_epoch: int = 1,
        save_every_n_steps: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self._model = model
        self._log_train_images = log_train_images
        self._log_test_images = log_test_images

        self._hparams = HParams(
            cell_size=cell_size,
            learning_rate=learning_rate,
            loss_p=loss_p,
            loss_p_weight=loss_p_weight,
            loss_loc=loss_loc,
            loss_loc_weight=loss_loc_weight,
            max_epochs=max_epochs,
            validation_every_n_epoch=validation_every_n_epoch,
            save_every_n_steps=save_every_n_steps,
            global_step=0,
        )

        if device == "gpu":
            device = "cuda"

        self._device = torch.device(device)
        self._model.to(self._device)

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._hparams.learning_rate)

        self._validation_scalar_log_buffer = dict()
        self.global_step = 0

        self._start_epoch = 0

        self._log_dir = log_dir
        self._checkpoints_dir = os.path.join(log_dir, "checkpoints")

    @property
    def global_step(self):
        return self._hparams.global_step

    @global_step.setter
    def global_step(self, global_step):
        self._hparams.global_step = global_step

    def fit(
        self,
        train_dataloader: DataLoader = None,
        validation_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
    ):
        assert train_dataloader is not None
        num_training_steps = len(train_dataloader)
        num_validation_steps = len(validation_dataloader)

        num_test_samples = len(test_dataloader) if test_dataloader is not None else None

        os.makedirs(self._checkpoints_dir, exist_ok=True)
        self._logger = SummaryWriter(log_dir=self._log_dir)

        if self._hparams.loss_p == "mse":
            self._loss_p = lambda pred, target: nn.functional.mse_loss(torch.sigmoid(pred), target)
        elif self._hparams.loss_p == "bce":
            self._loss_p = nn.functional.binary_cross_entropy_with_logits

        if self._hparams.loss_loc in ["mse", "l2"]:
            self._loss_loc = nn.functional.mse_loss
        elif self._hparams.loss_loc == "l1":
            self._loss_loc = nn.functional.l1_loss

        self._model.train()
        for epoch in range(self._start_epoch, self._hparams.max_epochs):
            self._model.train()
            self._mode = "train"
            for idx, (images, targets) in tqdm.tqdm(
                enumerate(train_dataloader),
                desc=f"Training Epoch {epoch}/{self._hparams.max_epochs}",
                total=num_training_steps,
            ):
                self.global_step = epoch * num_training_steps + idx
                images = images.to(self._device)
                targets = targets.to(self._device)

                self._optimizer.zero_grad()
                loss = self.training_step(images, targets, batch_idx=idx)
                loss.backward()
                self._optimizer.step()

            if ((epoch + 1) % self._hparams.validation_every_n_epoch) == 0:
                self._model.eval()
                self._mode = "validation"

                with torch.no_grad():
                    for idx, (images, targets) in tqdm.tqdm(
                        enumerate(validation_dataloader),
                        desc=f"Validation Epoch {epoch}/{self._hparams.max_epochs}",
                        total=num_validation_steps,
                    ):
                        images = images.to(self._device)
                        targets = targets.to(self._device)
                        self.validation_step(images, targets, batch_idx=idx)

                    self.validation_epoch_end(self._validation_scalar_log_buffer)

                    for key, values in self._validation_scalar_log_buffer.items():
                        mean_value = torch.tensor(values).mean(dim=0)
                        self._logger.add_scalar(key, mean_value, global_step=self.global_step)

            if (self._hparams.save_every_n_steps is not None) and (
                ((epoch + 1) % self._hparams.save_every_n_steps) == 0
            ):
                self.save(epoch=epoch)

            self._logger.flush()
            self._model.train()
            self._mode = "train"

    def log_scalar(self, tag: str = None, value: Any = None):
        if self._mode == "train":
            self._logger.add_scalar(tag, value, global_step=self.global_step)
        elif self._mode == "validation":
            if tag not in self._validation_scalar_log_buffer:
                self._validation_scalar_log_buffer[tag] = list()

            self._validation_scalar_log_buffer[tag].append(value)

    def log_image(self, tag: str = None, image: Any = None):
        self._logger.add_image(tag, image, global_step=self.global_step)

    def training_step(self, images, targets, batch_idx) -> torch.Tensor:
        loss, loss_p, loss_loc = self.step_common(images, targets, log_prefix="training")
        return loss

    def validation_step(self, images, targets, batch_idx):
        return self.step_common(images, targets, log_prefix="validation")

    def validation_epoch_end(self, scalars):
        def arg_top4_2d(values):
            indices_flat = torch.sort(values.flatten())[1][-4:]
            xs = indices_flat // values.shape[-1]
            ys = indices_flat % values.shape[-1]
            xs = xs.flatten()
            ys = ys.flatten()
            return xs, ys

        prefix_images = [
            ("train", self._log_train_images),
            ("test", self._log_test_images),
        ]

        cam = cv.VideoCapture(0)
        if cam.isOpened():
            cam.set(3, 1280)
            cam.set(4, 720)
            ret, frame = cam.read()
            if ret:
                frame = torchvision.transforms.functional.normalize(
                    torch.FloatTensor(frame).to(self._device).permute(2, 0, 1) / 255,
                    mean=0.5,
                    std=0.5,
                )[[2, 1, 0]]

                x_pad = (self._hparams.cell_size - frame.shape[2] % self._hparams.cell_size) % self._hparams.cell_size
                y_pad = (self._hparams.cell_size - frame.shape[1] % self._hparams.cell_size) % self._hparams.cell_size
                pad_left = math.ceil(x_pad / 2)
                pad_right = x_pad - pad_left
                pad_top = math.ceil(y_pad / 2)
                pad_bottom = y_pad - pad_top
                frame = torchvision.transforms.functional.pad(frame, padding=[pad_left, pad_top, pad_right, pad_bottom])
                prefix_images.append(("live", [(frame, None)]))
            cam.release()

        for prefix, log_images in prefix_images:
            if log_images is None:
                continue

            for idx, (image, target) in enumerate(log_images):
                image = image.to(self._device)
                if target is not None:
                    target = target.to(self._device)
                    target_p = target[0]
                    target_coord = target[1:]

                prediction = self._model(image.unsqueeze(0)).squeeze()

                prediction_p = prediction[0]
                prediction_coord = prediction[1:]

                log_image = torch.round(image * 127.5 + 127.5).type(torch.uint8)
                top4_x, top4_y = arg_top4_2d(prediction_p)
                arg_top4 = torch.stack([top4_x, top4_y], dim=-1)
                for cell_y in range(prediction_p.shape[0]):
                    for cell_x in range(prediction_p.shape[1]):
                        if target is not None and target_p[cell_y, cell_x] > 0.5:
                            image_x, image_y = target_to_image_coords(
                                image.shape,
                                cell_size=self._hparams.cell_size,
                                cell_x=cell_x,
                                cell_y=cell_y,
                                target_coords=target_coord,
                            )
                            log_image = torchvision.utils.draw_keypoints(
                                log_image,
                                keypoints=torch.FloatTensor([[[image_x, image_y]]]).to(self._device),
                                colors="green",
                                radius=4,
                            )

                        image_x, image_y = target_to_image_coords(
                            image.shape,
                            cell_size=self._hparams.cell_size,
                            cell_x=cell_x,
                            cell_y=cell_y,
                            target_coords=prediction_coord,
                        )

                        log_image = torchvision.utils.draw_keypoints(
                            log_image,
                            keypoints=torch.FloatTensor([[[image_x, image_y]]]).to(self._device),
                            colors="blue"
                            if torch.any(
                                torch.all(
                                    arg_top4 == torch.tensor([cell_y, cell_x]).to(self._device),
                                    dim=-1,
                                )
                            )
                            else "red",
                            radius=torch.round(torch.sigmoid(prediction_p[cell_y, cell_x]) * 5 + 1).int(),
                        )

                self.log_image(
                    f"{prefix}/image_{idx}",
                    log_image,
                )

    def test_step(self, images, targets, targets_idx):
        return self.step_common(images, targets, log_prefix="test")

    def test_epoch_end(self, scalars):
        pass

    def step_common(
        self, images, targets, log_prefix: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self._model(images)

        loss_p, loss_loc = self.losses(prediction=prediction, target=targets)
        loss = self._hparams.loss_p_weight * loss_p + self._hparams.loss_loc_weight * loss_loc

        if log_prefix is not None:
            self.log_scalar(f"{log_prefix}/loss", loss)
            self.log_scalar(f"{log_prefix}/loss_p", loss_p)
            self.log_scalar(f"{log_prefix}/loss_loc", loss_loc)

        return loss, loss_p, loss_loc

    def losses(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_p = target[:, 0]
        target_locs = target[:, 1:]

        prediction_p = prediction[:, 0]
        prediction_locs = prediction[:, 1:]

        loss_p = self._loss_p(prediction_p, target_p)
        locs_mask = target_p.squeeze().bool()

        loss_loc = self._loss_loc(
            prediction_locs.permute(0, 2, 3, 1)[locs_mask],
            target_locs.permute(0, 2, 3, 1)[locs_mask],
        )

        return loss_p, loss_loc

    def save(self, epoch: int):
        save_dir = os.path.join(self._checkpoints_dir, f"{epoch:06d}")
        os.makedirs(save_dir, exist_ok=True)
        hparams_file = os.path.join(save_dir, "hparams.json")
        model_pt = os.path.join(save_dir, "model.pt")
        optimizer_pt = os.path.join(save_dir, "optimizer.pt")

        model_mode = self._model.training

        self._model.eval()
        model_state = self._model.state_dict()
        torch.save(model_state, model_pt)
        self._model.train(mode=model_mode)

        optimizer_state = self._optimizer.state_dict()
        torch.save(optimizer_state, optimizer_pt)

        with open(hparams_file, "w") as f:
            json.dump(asdict(self._hparams), f)

    def load(self, checkpoint_path: str):
        assert os.path.exists(checkpoint_path)
        checkpoint_path = os.path.normpath(checkpoint_path)
        hparams_file = os.path.join(checkpoint_path, "hparams.json")
        model_pt = os.path.join(checkpoint_path, "model.pt")
        optimizer_pt = os.path.join(checkpoint_path, "optimizer.pt")

        self._checkpoints_dir, epoch_str = os.path.split(checkpoint_path)
        self._log_dir = os.path.split(self._checkpoints_dir)[0]

        # The epoch from the folder name has already been completed, so we need to start from the next one.
        self._start_epoch = int(epoch_str) + 1

        assert os.path.exists(hparams_file) and os.path.exists(model_pt) and os.path.exists(optimizer_pt)

        model_mode = self._model.training
        self._model.eval()
        self._model.load_state_dict(torch.load(model_pt))
        self._model.train(mode=model_mode)

        self._optimizer.load_state_dict(torch.load(optimizer_pt))

        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        self._hparams = HParams(**hparams)
