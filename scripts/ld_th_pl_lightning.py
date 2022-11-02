from pathlib import Path
from collections import OrderedDict

from pyexpat import model

from rich.progress import track

import numpy as np
import pandas as pd

import albumentations as A

import ternausnet.models

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import ld_dataset as ldd
import ld_plot as ldp
import ld_image as ldi
import ld_th_pl_lightning as ldpl


g_device = (
    "mps"
    if torch.backends.mps.is_built() is True
    else "cuda"
    if torch.backends.cuda.is_built()
    else "cpu"
)


csv_version_overview_path = Path("../notebooks/lightning_logs").joinpath(
    "versions_overview.csv"
)


def create_model(model_name, device):
    model = getattr(ternausnet.models, model_name)(pretrained=True)
    model = model.to(device)
    return model


class LeafDiskSegmentation(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        num_workers: int,
        max_epochs: int,
        accumulate_grad_batches: int,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        train_augmentations,
        val_augmentations,
        selected_device: str = None,
    ):
        super().__init__()
        # hparams
        self.batch_size = batch_size
        self.selected_device = (
            selected_device if selected_device is not None else g_device
        )
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        # dataframes
        self.train_data = train_data
        self.val_data = val_data
        # albumentations
        self.train_augmentations = train_augmentations
        self.val_augmentations = val_augmentations
        # Model
        self.unet = getattr(ternausnet.models, "UNet11")(
            pretrained=True,
        ).to(g_device)
        self.loss = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def forward(self, x):
        embedding = self.unet(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            ldd.LeafDeafSegmentationDataset(
                df_img=self.train_data,
                transform=self.train_augmentations,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ldd.LeafDeafSegmentationDataset(
                df_img=self.val_data,
                transform=self.val_augmentations,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        y = y.long()
        loss = self.loss(logits, y.unsqueeze(1).float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        y = y.long()
        loss = self.loss(logits, y.unsqueeze(1).float())
        self.log("val_loss", loss)
        return loss

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        y = y.long()
        loss = self.loss(logits, y.unsqueeze(1).float())
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


class LeafDiskSegmentationPredictor(object):
    def __init__(
        self,
        model=None,
        device: str = None,
        prediction_threshold=0.5,
    ) -> None:
        self._model = model
        self._device = g_device if device is None else device
        self._prediction_threshold = prediction_threshold
        self._trainer = Trainer(
            accelerator="gpu" if self._device == "mps" else "cpu",
            logger=False,
        )

    def predict_image(self, image_path, return_probabilities: bool = False):
        image = ldd.open_image(str(image_path))
        original_size = tuple(image.shape[:2])

        self._model.eval()
        predictions = self._trainer.predict(
            self._model,
            DataLoader(
                ldd.LeafDeafSegmentationInferenceDataset(
                    image_list=[image],
                    transform=self._model.val_augmentations,
                    return_image_size=False,
                ),
                batch_size=1,
                num_workers=0,
            ),
        )

        probabilities = torch.sigmoid(predictions[0].squeeze(1))
        if return_probabilities is True:
            return A.resize(
                probabilities.float().squeeze(0).squeeze(0).cpu().numpy(),
                height=original_size[0],
                width=original_size[1],
            )

        predicted_mask = (
            (probabilities >= self._prediction_threshold)
            .float()
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        predicted_mask = A.resize(
            predicted_mask,
            height=original_size[0],
            width=original_size[1],
        )
        predicted_mask[predicted_mask != 0] = 255

        return predicted_mask.astype(np.uint8)

    @property
    def model(self):
        return model

    @model.setter
    def model(self, new_model):
        self._model = new_model


def train_ld_segmenter(model: LeafDiskSegmentation, trainer: Trainer):
    trainer.fit()


def update_overviews(df_test_images):
    if csv_version_overview_path.is_file():
        version_overview = pd.read_csv(csv_version_overview_path).reset_index(drop=True)
    else:
        version_overview = pd.DataFrame()

    checkpoints = [
        chk
        for fld in Path("lightning_logs").glob("*")
        for chk in Path(f"{fld}/checkpoints").glob("*")
        if fld.is_dir() is True
        and (
            "checkpoint_fileName" not in version_overview.columns
            or str(chk) not in version_overview.checkpoint_fileName.to_list()
        )
    ]

    if checkpoints:
        data = OrderedDict()

        data["epoch"] = []
        data["step"] = []
        data["train_loss"] = []
        data["val_loss"] = []
        data["test_loss"] = []
        data["version"] = []
        data["image_width"] = []
        data["image_height"] = []
        data["checkpoint_fileName"] = []
        data["selected_device"] = []
        data["learning_rate"] = []
        data["batch_size"] = []
        data["accumulate_grad_batches"] = []
        data["max_epochs"] = []
        data["num_workers"] = []

        trainer = Trainer(accelerator="gpu", logger=False)

        for chk in track(checkpoints, description="Testing models"):
            print(chk)
            try:
                model = LeafDiskSegmentation.load_from_checkpoint(chk)

                for key, chunk in zip(
                    data.keys(),
                    [chunk.split("=")[1] for chunk in chk.stem.split("-")],
                ):
                    data[key].append(chunk)

                resizer = model.val_augmentations[0]
                data["image_width"].append(resizer.width)
                data["image_height"].append(resizer.height)
                data["version"].append(chk.parent.parent.stem)

                test_result = trainer.test(
                    model,
                    DataLoader(
                        ldd.LeafDeafSegmentationDataset(
                            df_img=df_test_images,
                            transform=model.val_augmentations,
                        ),
                        batch_size=1,
                        num_workers=1,
                        pin_memory=True,
                    ),
                )
                data["test_loss"].append(list(test_result[0].values())[0])
                for k in [
                    "selected_device",
                    "learning_rate",
                    "batch_size",
                    "accumulate_grad_batches",
                    "max_epochs",
                    "num_workers",
                ]:
                    if hasattr(model, k):
                        data[k].append(getattr(model, k))
                    else:
                        data[k].append(None)
                data["checkpoint_fileName"].append(chk)
            except Exception:
                pass

        try:
            version_overview = pd.concat([version_overview, pd.DataFrame(data=data)])
        except:
            for k in data:
                print(f"{k}: {len(data[k])}")
        else:
            version_overview.to_csv(csv_version_overview_path, index=False)

    version_overview = version_overview.sort_values(
        ["val_loss", "train_loss", "test_loss", "epoch", "step"]
    ).reset_index(drop=True)

    return version_overview
