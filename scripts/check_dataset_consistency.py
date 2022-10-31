from pathlib import Path

import click

from rich import print
from rich.progress import track

DEFAULT_ROOT_FOLDER = "./data_in/images/ld_sheets" if True else "./"


def out_to(txt: str):
    print(txt)


@click.command()
@click.option("--root_folder", default=DEFAULT_ROOT_FOLDER, help="Path to root folder")
@click.option(
    "--images_folder_name",
    default="images",
    help="Name of the images folder",
)
@click.option("--masks_folder_name", default="masks", help="Name of the masks folder")
def check_dataset(root_folder, images_folder_name, masks_folder_name):
    out_to(f"Current working folder {str(Path.cwd())}")
    out_to("Check that images and masks folders exist")
    p_root_folder = Path(root_folder)
    good_images = p_root_folder.joinpath(images_folder_name).is_dir()
    good_maskss = p_root_folder.joinpath(masks_folder_name).is_dir()
    if good_images is False:
        out_to(
            f"images folder {images_folder_name} not found in root folder {p_root_folder}"
        )
    if good_maskss is False:
        out_to(
            f"images folder {masks_folder_name} not found in root folder {p_root_folder}"
        )
    if good_images is False or good_maskss is False:
        out_to("Folder error, exiting")
        out_to("---> Exit")
        return
    out_to("___________________________")

    img_ext = []
    for image_path in track(
        p_root_folder.joinpath(images_folder_name).glob("*"),
        description="Checking image extensions",
    ):
        ext = image_path.suffix
        if ext and ext not in img_ext:
            img_ext.append(ext)
    if len(img_ext) == 0:
        out_to("No images found")
        out_to("---> Exit")
        return
    elif len(img_ext) == 1:
        out_to(f"All images have the extension: {img_ext[0]}")
    else:
        out_to("There multiple images extensions:")
        for e in img_ext:
            out_to(e)
    out_to("___________________________")

    msk_ext = []
    for image_path in track(
        p_root_folder.joinpath(masks_folder_name).glob("*"),
        description="Checking mask extensions",
    ):
        ext = image_path.suffix
        if ext and ext not in msk_ext:
            msk_ext.append(ext)
    if len(msk_ext) == 0:
        out_to("No masks found")
        out_to("---> Exit")
        return
    elif len(msk_ext) == 1:
        out_to(f"All masks have the extension: {msk_ext[0]}")
    else:
        out_to("There multiple masks extensions:")
        for e in msk_ext:
            out_to(e)
    out_to("___________________________")

    image_names = [f.stem for f in p_root_folder.joinpath(images_folder_name).glob("*")]
    mask_names = [f.stem for f in p_root_folder.joinpath(masks_folder_name).glob("*")]

    missing_masks = []
    for image_name in track(
        image_names,
        description="Checking missing masks",
    ):
        if image_name not in mask_names:
            missing_masks.append(image_name)
    if missing_masks:
        out_to("Missing masks for:")
        [out_to(mm) for mm in missing_masks]
    else:
        out_to("No missing masks")
    out_to("___________________________")

    missing_images = []
    for image_name in track(
        mask_names,
        description="Checking missing masks",
    ):
        if image_name not in image_names:
            missing_images.append(image_name)
    if missing_images:
        out_to("Missing images for:")
        [out_to(mm) for mm in missing_images]
    else:
        out_to("No missing images")
    out_to("___________________________")


if __name__ == "__main__":
    check_dataset()
