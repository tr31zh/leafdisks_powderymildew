from pathlib import Path
from re import I, L

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import cv2

from torch.utils.data import Dataset

root_folder: Path = Path(__file__).parent.parent.joinpath("data_in", "images")

train_images_folder = root_folder.joinpath("ld_sheets", "images")
train_masks_folder = root_folder.joinpath("ld_sheets", "masks")

una_images_folder = root_folder.joinpath("ld_tmp")
wot_images_folder = root_folder.joinpath("wot")


def _get_mask_path_from_image(image_name, masks_folder):
    mask_path = masks_folder.joinpath(image_name + ".png")
    if mask_path.is_file() is False:
        mask_path = masks_folder.joinpath(image_name + ".bmp")
    if mask_path.is_file() is False:
        return None
    return mask_path


def _get_image_path_from_mask(mask_name, images_folder):
    image_path = images_folder.joinpath(mask_name + ".jpg")
    return False if image_path.is_file() is False else image_path


def check_items_consistency(images_folder, masks_folder=None) -> pd.DataFrame:

    if isinstance(images_folder, Path) is False:
        images_folder = Path(images_folder)
    if masks_folder is not None and isinstance(masks_folder, Path) is False:
        masks_folder = Path(masks_folder)

    df_dict = {}
    df_dict["miss_mask"] = []
    df_dict["miss_image"] = []
    for image_path in images_folder.glob("*"):
        if _get_mask_path_from_image(image_path.stem, masks_folder) is None:
            df_dict["miss_mask"].append(image_path)
    for mask_path in masks_folder.glob("*"):
        if _get_image_path_from_mask(mask_path.stem, images_folder) is None:
            df_dict["miss_image"].append(image_path)

    return df_dict


def build_items_dataframe(images_folder, masks_folder=None) -> pd.DataFrame:
    if isinstance(images_folder, Path) is False:
        images_folder = Path(images_folder)
    if masks_folder is not None and isinstance(masks_folder, Path) is False:
        masks_folder = Path(masks_folder)

    df_dict = {}
    if masks_folder is not None:
        df_dict["image_name"] = []
        df_dict["image_path"] = []
        df_dict["mask_path"] = []
        for image_path in tqdm(images_folder.glob("*"), desc="Parsing images folder"):
            if cv2.imread(str(image_path)) is None:
                continue
            image_name = image_path.stem
            mask_path = _get_mask_path_from_image(image_name, masks_folder)
            if mask_path is None or cv2.imread(str(mask_path)) is None:
                continue
            df_dict["image_name"].append(image_name)
            df_dict["image_path"].append(str(image_path))
            df_dict["mask_path"].append(str(mask_path))
        return pd.DataFrame(df_dict).assign(
            year=lambda x: x.image_name.str.lower()
            .str.split(pat="exp", expand=True)[1]
            .str.split(pat="dm", expand=True)[0]
            .astype(int)
        )
    else:
        df_dict["image_name"] = []
        df_dict["image_path"] = []
        for image_path in tqdm(images_folder.glob("*"), desc="Parsing images folder"):
            if cv2.imread(str(image_path)) is None:
                continue
            df_dict["image_name"].append(image_path.name)
            df_dict["image_path"].append(str(image_path))
        return pd.DataFrame(df_dict).assign("mask_path", None)


def get_image_name_at(idx: int, df: pd.DataFrame) -> str:
    return df.image_name.to_list()[0]


def get_mask_path(image_name: str, df: pd.DataFrame):
    try:
        mask_path = df[df.image_name == image_name].mask_path.to_list()[0]
    except:
        return None
    else:
        return mask_path


def get_image_path(image_name, df: pd.DataFrame):
    try:
        image_path = df[df.image_name == image_name].image_path.to_list()[0]
    except:
        return None
    else:
        return image_path


def open_mask(path, df: pd.DataFrame = None):
    mask_path = path if df is None else get_mask_path(image_name=path, df=df)
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask[mask == 0.0] = 0.0
        mask[mask != 0.0] = 1.0
        return mask
    else:
        return None


def open_image(path, df: pd.DataFrame = None):
    return cv2.cvtColor(
        cv2.imread(path if df is None else get_image_path(image_name=path, df=df)),
        cv2.COLOR_BGR2RGB,
    )


def open_image_and_mask(key, df: pd.DataFrame):
    if isinstance(key, str):
        return open_image(key, df), open_mask(key, df)
    if isinstance(key, int):
        filename = get_image_name_at(key, df)
        return open_image(filename, df), open_mask(filename, df)
    return None, None


class LeafDeafSegmentationDataset(Dataset):
    def __init__(self, df_img: pd.DataFrame, transform=None):
        self.transform = transform
        self.df_img = df_img

    def __len__(self):
        return self.df_img.shape[0]

    def __getitem__(self, idx):
        image, mask = open_image_and_mask(key=idx, df=self.df_img)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


class LeafDeafSegmentationInferenceDataset(Dataset):
    def __init__(self, image_list, transform, dataframe=None, return_image_size=True):
        self.transform = transform
        self.image_list = image_list
        self.dataframe = dataframe
        self.return_image_size = return_image_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        if isinstance(image, np.ndarray) is False:
            image = open_image(self.image_list[idx])
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        if self.return_image_size is True:
            return image, original_size
        else:
            return image


toto = "toto"
