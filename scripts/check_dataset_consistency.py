from distutils import extension
from pathlib import Path
import subprocess
import sys
import json


def install_pkg(pkg):
    try:
        __import__(pkg[0])
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg[1]])


[
    install_pkg(pkg)
    for pkg in [
        ("click", "click"),
        ("rich", "rich"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
    ]
]

from PIL import Image

import numpy as np

import click

from rich.console import Console
from rich.progress import track

# DEFAULT_ROOT_FOLDER = "./data_in/images/ld_sheets"
DEFAULT_ROOT_FOLDER = "./"

console = Console()


def out_to(txt: str, style=None):
    console.print(txt, style=style)


def exit_error(path, report, error_message=None):
    if error_message is not None:
        out_to(error_message, style="red")
    out_to("---> Exit", style="red")
    if path.is_dir() is True:
        write_report(path=path, report=report)


def write_report(path, report):
    out_to(f"Process report in {str(path.joinpath('report.json'))}")
    json.dump(
        report,
        open(str(path.joinpath("report.json")), "w"),
        indent=4,
    )


def check_dataset(root_folder, images_folder_name, masks_folder_name):
    out_to("___________________________")
    out_to(f"Current working folder {str(Path.cwd())}")
    out_to("___________________________")
    out_to("Check that images and masks folders exist")

    report = {}

    p_root_folder = Path(root_folder)
    if p_root_folder.is_dir() is False:
        exit_error(
            path=p_root_folder,
            report=report,
            error_message="Root folder does not exist",
        )
        return
    good_images = p_root_folder.joinpath(images_folder_name).is_dir()
    good_masks = p_root_folder.joinpath(masks_folder_name).is_dir()
    report["good_images"] = good_images
    report["good_masks"] = good_masks
    if good_images is False:
        out_to(
            f"images folder {images_folder_name} not found in root folder {p_root_folder}"
        )
    if good_masks is False:
        out_to(
            f"images folder {masks_folder_name} not found in root folder {p_root_folder}"
        )
    if good_images is False or good_masks is False:
        exit_error(path=p_root_folder, report=report)
        return
    else:
        out_to("-> Folders OK")
    out_to("___________________________")

    def check_extensions(fld_path, data_type: str, extensions) -> bool:
        dt_ext = []
        for image_path in track(
            [i for i in p_root_folder.joinpath(fld_path).glob("*")],
            description=f"Checking {data_type} extensions",
            total=1000,
        ):
            if ".DS_Store" in str(image_path):
                continue
            ext = image_path.suffix
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(str(image_path.resolve()))
            if ext and ext not in dt_ext:
                dt_ext.append(ext)

        if len(dt_ext) == 0:
            exit_error(
                path=p_root_folder, report=report, error_message="No extensions fount"
            )
            return False
        elif len(dt_ext) == 1:
            out_to(f"All {data_type}s have the extension: {dt_ext[0]}")
        else:
            out_to(f"There multiple {data_type}s extensions:", style="red")
            for e in dt_ext:
                out_to(f"- {e}")
        out_to("___________________________")
        return True

    extensions = {
        "images": {},
        "masks": {},
    }
    if (
        check_extensions(
            fld_path=images_folder_name,
            data_type="image",
            extensions=extensions["images"],
        )
        is False
    ):
        return
    if (
        check_extensions(
            fld_path=masks_folder_name,
            data_type="mask",
            extensions=extensions["masks"],
        )
        is False
    ):
        return
    report["extensions"] = extensions

    image_names = [f.stem for f in p_root_folder.joinpath(images_folder_name).glob("*")]
    mask_names = [f.stem for f in p_root_folder.joinpath(masks_folder_name).glob("*")]

    def check_missing(present_list, look_for_list, data_type, error_list: list):
        missing_ = []
        for p in track(present_list, description=f"Checking missing {data_type}s"):
            if p not in look_for_list:
                error_list.append(p)
                missing_.append(p)
        if missing_:
            out_to(f"Missing {data_type}s for:", style="red")
            [out_to(mm) for mm in missing_]
        else:
            out_to(f"No missing {data_type}s")
        out_to("___________________________")

    report["missing_masks"] = []
    report["missing_images"] = []
    check_missing(
        image_names,
        mask_names,
        data_type="image",
        error_list=report["missing_masks"],
    )
    check_missing(
        mask_names,
        image_names,
        data_type="mask",
        error_list=report["missing_images"],
    )

    def check_metadata(data_path, data_type, report_data):
        for image_path in track(
            [i for i in p_root_folder.joinpath(data_path).glob("*")],
            description=f"Checking {data_type} formats",
        ):
            if ".DS_Store" in str(image_path):
                continue
            try:
                img = np.array(Image.open(str(image_path)))
            except Exception as e:
                report_data["load_fails"].append(str(image_path))
                continue
            if str(img.shape) not in report_data["shapes"]:
                report_data["shapes"][str(img.shape)] = []
            report_data["shapes"][str(img.shape)].append(str(image_path))

            if str(img.dtype) not in report_data["formats"]:
                report_data["formats"][str(img.dtype)] = []
            report_data["formats"][str(img.dtype)].append(str(image_path))

        if len(report_data["load_fails"]) == 0:
            out_to("All images loaded succesfully")
        else:
            out_to("Failed to load:", style="red")
            for img in report_data["load_fails"]:
                out_to(f"- {img}")

        shapes = list(set(list(report_data["shapes"].keys())))
        if len(shapes) == 0:
            exit_error(
                path=p_root_folder, report=report, error_message="Empty images only"
            )
            return
        elif len(shapes) == 1:
            out_to(f"All images have the same shape: {shapes[0]}")
        else:
            out_to("Multiple shapes detected:", style="red")
            for shape in shapes:
                out_to(f"- {shape}")

        formats = list(set(list(report_data["formats"].keys())))
        if len(formats) == 0:
            exit_error(path=p_root_folder, report=report, error_message="No formats")
            return
        elif len(formats) == 1:
            sfmt = formats[0]
            if sfmt == "uint8":
                out_to(f"Single format detected: {sfmt}")
            else:
                out_to(f"Wrong format detected: {sfmt}", style="red")
        else:
            out_to("Multiple formats detected:", style="red")
            for format in formats:
                out_to(f"- {format}")

    report["metatdata"] = {
        "images": {
            "shapes": {},
            "formats": {},
            "load_fails": [],
        },
        "masks": {
            "shapes": {},
            "formats": {},
            "load_fails": [],
        },
    }
    check_metadata(
        data_path=images_folder_name,
        data_type="image",
        report_data=report["metatdata"]["images"],
    )
    check_metadata(
        data_path=masks_folder_name,
        data_type="mask",
        report_data=report["metatdata"]["masks"],
    )


@click.command()
@click.option(
    "--root_folder",
    default=DEFAULT_ROOT_FOLDER,
    help="Path to root folder",
)
@click.option(
    "--images_folder_name",
    default="images",
    help="Name of the images folder, 'images' by default",
)
@click.option(
    "--masks_folder_name",
    default="masks",
    help="Name of the masks folder, 'masks' by default",
)
def _check_dataset(root_folder, images_folder_name, masks_folder_name):
    return check_dataset(
        root_folder=root_folder,
        images_folder_name=images_folder_name,
        masks_folder_name=masks_folder_name,
    )


if __name__ == "__main__":
    _check_dataset()
