import copy

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import albumentations as A
from albumentations.pytorch import ToTensorV2

import ld_image as ldi
import ld_dataset as ldd


def _update_axis(axis, image, title=None, fontsize=18, remove_axis=True):
    axis.imshow(image, origin="upper")
    if title is not None:
        axis.set_title(title, fontsize=fontsize)
    if remove_axis is True:
        axis.set_axis_off()


def visualize_image(image, title=None, fontsize=18, remove_axis=True, figsize=(8, 8)):
    _, ax = plt.subplots(1, 1, figsize=figsize)
    _update_axis(
        axis=ax,
        image=image,
        title=title,
        fontsize=fontsize,
        remove_axis=remove_axis,
    )
    plt.tight_layout()
    plt.show()


def visualize_item(image, mask, direction="tb", figsize=(8, 8)):
    if direction == "tb":
        _, ax = plt.subplots(2, 1, figsize=figsize)
    else:
        _, ax = plt.subplots(1, 2, figsize=figsize)
    _update_axis(ax[0], image=image)
    _update_axis(ax[1], mask)

    plt.tight_layout()
    plt.show()


def visualize_augmented_item(image, mask, original_image=None, original_mask=None):
    _, ax = plt.subplots(2, 2, figsize=(8, 8))
    _update_axis(ax[0, 0], image=original_image, title="Original image")
    _update_axis(ax[0, 1], image=original_mask, title="Original mask")
    _update_axis(ax[1, 0], image=image, title="Transformed image")
    _update_axis(ax[1, 1], image=mask, title="Transformed mask")

    plt.tight_layout()
    plt.show()


def display_image_grid(df_img, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = df_img.shape[0]
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2 * rows))
    for i, image_filename in enumerate(df_img.image_name):
        image, mask = ldd.open_image_and_mask(image_filename, df_img)

        _update_axis(ax[i, 0], image=image, title=image_filename)
        _update_axis(ax[i, 1], image=mask, title="Ground truth mask")

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            _update_axis(ax[i, 2], image=predicted_mask, title="Predicted mask")
    plt.tight_layout()
    plt.show()


def visualize_augmented_dataset_item(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose(
        [t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))]
    )
    _, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 4 * samples))
    for i in range(samples):
        image, mask = dataset[idx + i]
        _update_axis(ax[i, 0], image=image, title=f"Image {image.shape}")
        _update_axis(ax[i, 1], image=mask, title=f"Mask {mask.shape}")
    plt.tight_layout()
    plt.show()


def show_masked_image(image, mask, luma=0.3, width=10, height=8):
    figure(figsize=(width, height), dpi=80)
    plt.imshow(ldi.apply_mask(image=image, mask=mask, bckg_luma=luma), origin="upper")
    plt.gray()
    plt.axis("off")
    plt.tight_layout()
    plt.show()
