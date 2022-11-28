from pathlib import Path

from rich.console import Console

import albumentations as A

from sklearn.model_selection import train_test_split

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from pytorch_lightning.callbacks import RichProgressBar

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import ld_dataset as ldd
import ld_th_pl_lightning as ldpl
import check_dataset_consistency as cdc

import click


console = Console()


@click.command()
@click.option("--batch_size", default=8, help="Batch size")
@click.option("--image_size_factor", default=14, help="Resizer image height size")
@click.option("--max_epoch", default=400, help="Max epoch count")
@click.option("--train_count", default=1, help="Training will be run X times")
def train_model(batch_size, image_size_factor, max_epoch, train_count):
    console.print("Starting Trainings")
    console.print("______________________________________________________")
    # Check dataset
    cdc.check_dataset(
        root_folder=ldd.root_folder.joinpath("ld_sheets"),
        images_folder_name="images",
        masks_folder_name="masks",
    )

    # Build trainino dataset
    df_train_images = ldd.build_items_dataframe(
        images_folder=ldd.train_images_folder,
        masks_folder=ldd.train_masks_folder,
    )

    for i in range(train_count):
        console.print("______________________________________________________")
        console.print(f"Starting training {i+1}/{train_count}")

        train, test = train_test_split(
            df_train_images, test_size=0.3, stratify=df_train_images["year"]
        )
        test, val = train_test_split(test, test_size=0.5, stratify=test["year"])

        # assert (image_size_factor / 2 == 0, "image_size_factor should not be odd")

        img_width = 32 * image_size_factor
        img_height = 32 * image_size_factor

        # assert img_width / img_height == 1.5

        alb_resizer = [A.Resize(height=img_height, width=img_width)]

        train_transformers_list = alb_resizer + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(p=0.33),
            A.CLAHE(p=0.33),
        ]

        to_tensor = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        model = ldpl.LeafDiskSegmentation(
            batch_size=batch_size,
            selected_device=ldpl.g_device,
            learning_rate=0.001,
            max_epochs=max_epoch,
            num_workers=0,
            train_augmentations=A.Compose(train_transformers_list + to_tensor),
            train_data=train,
            val_augmentations=A.Compose(alb_resizer + to_tensor),
            val_data=val,
            accumulate_grad_batches=3,
        )

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=model.max_epochs,
            log_every_n_steps=5,
            callbacks=[
                RichProgressBar(),
                EarlyStopping(
                    monitor="val_loss", mode="min", patience=15, min_delta=0.0005
                ),
                DeviceStatsMonitor(),
                ModelCheckpoint(
                    save_top_k=3,
                    monitor="val_loss",
                    auto_insert_metric_name=True,
                    filename="{epoch}-{step}-{train_loss:.5}-{val_loss:.5f}",
                ),
            ],
            accumulate_grad_batches=model.accumulate_grad_batches,
        )
        trainer.fit(model)


if __name__ == "__main__":
    train_model()
