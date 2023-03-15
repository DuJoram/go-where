import math
from typing import Tuple, Optional, List, Union, Any

import cv2 as cv
import torch
import torchvision
import torchvision.utils
import tqdm
from torch import optim, nn
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


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_p_weight: float = 1,
        loss_loc_weight: float = 1,
        learning_rate=1e-3,
        max_epochs: int = 100,
        cell_size: int = 64,
        log_train_images: List[Tuple[torch.FloatTensor]] = None,
        log_test_images: List[Tuple[torch.FloatTensor]] = None,
        log_dir: str = "logs/",
        log_every_n_steps: int = 1,
        validation_every_n_epoch: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        self._model = model
        self._log_train_images = log_train_images
        self._log_test_images = log_test_images
        self._cell_size = cell_size
        self._learning_rate = learning_rate
        self._loss_p_weight = loss_p_weight
        self._loss_loc_weight = loss_loc_weight
        self._max_epochs = max_epochs

        self._validation_every_n_epoch = validation_every_n_epoch

        if device == "gpu":
            device = "cuda"
        self._device = torch.device(device)
        self._model.to(self._device)

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

        self._validation_scalar_log_buffer = dict()
        self._logger = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

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

        self._model.train()
        for epoch in range(self._max_epochs):
            self._model.train()
            self._mode = "train"
            for idx, (images, targets) in tqdm.tqdm(
                enumerate(train_dataloader),
                desc=f"Training Epoch {epoch}/{self._max_epochs}",
                total=num_training_steps,
            ):
                self.global_step = epoch * num_training_steps + idx
                images = images.to(self._device)
                targets = targets.to(self._device)

                self._optimizer.zero_grad()
                loss = self.training_step(images, targets, batch_idx=idx)
                loss.backward()
                self._optimizer.step()

            if ((epoch + 1) % self._validation_every_n_epoch) == 0:
                self._model.eval()
                self._mode = "validation"

                with torch.no_grad():
                    for idx, (images, targets) in tqdm.tqdm(
                        enumerate(validation_dataloader),
                        desc=f"Validation Epoch {epoch}/{self._max_epochs}",
                        total=num_validation_steps,
                    ):
                        images = images.to(self._device)
                        targets = targets.to(self._device)
                        self.validation_step(images, targets, batch_idx=idx)

                    self.validation_epoch_end(self._validation_scalar_log_buffer)

                    for key, values in self._validation_scalar_log_buffer.items():
                        mean_value = torch.tensor(values).mean(dim=0)
                        self._logger.add_scalar(
                            key, mean_value, global_step=self.global_step
                        )

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
        loss, loss_p, loss_loc = self.step_common(
            images, targets, log_prefix="training"
        )
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

                x_pad = (
                    self._cell_size - frame.shape[2] % self._cell_size
                ) % self._cell_size
                y_pad = (
                    self._cell_size - frame.shape[1] % self._cell_size
                ) % self._cell_size
                pad_left = math.ceil(x_pad / 2)
                pad_right = x_pad - pad_left
                pad_top = math.ceil(y_pad / 2)
                pad_bottom = y_pad - pad_top
                frame = torchvision.transforms.functional.pad(
                    frame, padding=[pad_left, pad_top, pad_right, pad_bottom]
                )
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
                                cell_size=self._cell_size,
                                cell_x=cell_x,
                                cell_y=cell_y,
                                target_coords=target_coord,
                            )
                            log_image = torchvision.utils.draw_keypoints(
                                log_image,
                                keypoints=torch.FloatTensor([[[image_x, image_y]]]).to(
                                    self._device
                                ),
                                colors="green",
                                radius=4,
                            )

                        image_x, image_y = target_to_image_coords(
                            image.shape,
                            cell_size=self._cell_size,
                            cell_x=cell_x,
                            cell_y=cell_y,
                            target_coords=prediction_coord,
                        )

                        log_image = torchvision.utils.draw_keypoints(
                            log_image,
                            keypoints=torch.FloatTensor([[[image_x, image_y]]]).to(
                                self._device
                            ),
                            colors="blue"
                            if torch.any(
                                torch.all(
                                    arg_top4
                                    == torch.tensor([cell_y, cell_x]).to(self._device),
                                    dim=-1,
                                )
                            )
                            else "red",
                            radius=torch.round(
                                torch.sigmoid(prediction_p[cell_y, cell_x]) * 5 + 1
                            ).int(),
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
        loss = self._loss_p_weight * loss_p + self._loss_loc_weight * loss_loc

        if log_prefix is not None:
            self.log_scalar(f"{log_prefix}/loss", loss)
            self.log_scalar(f"{log_prefix}/loss_p", loss_p)
            self.log_scalar(f"{log_prefix}/loss_loc", loss_loc)

        return loss, loss_p, loss_loc

    def losses(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_p = target[:, 0]
        target_locs = target[:, 1:]

        prediction_p = prediction[:, 0]
        prediction_locs = prediction[:, 1:]

        loss_p = nn.functional.binary_cross_entropy_with_logits(prediction_p, target_p)
        locs_mask = target_p.squeeze().bool()
        loss_loc = nn.functional.mse_loss(
            prediction_locs.permute(0, 2, 3, 1)[locs_mask],
            target_locs.permute(0, 2, 3, 1)[locs_mask],
        )

        return loss_p, loss_loc
