from typing import Any, Union

import numpy as np
import cv2

from sklearn.cluster import MeanShift
from scipy.interpolate import splprep, splev

import albumentations as A


C_BLACK = (0, 0, 0)
C_BLUE = (255, 0, 0)
C_BLUE_VIOLET = (226, 43, 138)
C_CABIN_BLUE = (209, 133, 67)
C_CYAN = (255, 255, 0)
C_DIM_GRAY = (105, 105, 105)
C_FUCHSIA = (255, 0, 255)
C_GREEN = (0, 128, 0)
C_LIGHT_STEEL_BLUE = (222, 196, 176)
C_LIME = (0, 255, 0)
C_MAROON = (0, 0, 128)
C_ORANGE = (80, 127, 255)
C_PURPLE = (128, 0, 128)
C_RED = (0, 0, 255)
C_SILVER = (192, 192, 192)
C_TEAL = (128, 128, 0)
C_WHITE = (255, 255, 255)
C_YELLOW = (0, 255, 255)


def to_RGB(BGR):
    return tuple(reversed(BGR))


def ensure_odd(
    i: int,
    min_val: Union[None, int] = None,
    max_val: Union[None, int] = None,
) -> int:
    """Transforms an odd number into pair number by adding one
    Arguments:
        i {int} -- number
    Returns:
        int -- Odd number
    """
    if (i > 0) and (i % 2 == 0):
        i += 1
    if min_val is not None:
        return max(i, min_val)
    if max_val is not None:
        return min(i, max_val)
    return i


def get_morphology_kernel(size: int, shape: int):
    """Builds morphology kernel
    :param size: kernel size, must be odd number
    :param shape: select shape of kernel
    :return: Morphology kernel
    """
    return cv2.getStructuringElement(shape, (size, size))


def open(
    image: Any,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
    rois: tuple = (),
    proc_times: int = 1,
):
    """Morphology - Open wrapper
    Arguments:
        image {numpy array} -- Source image
        kernel_size {int} -- kernel size
        kernel_shape {int} -- cv2 constant
        roi -- Region of Interrest
        proc_times {int} -- iterations
    Returns:
        numpy array -- opened image
    """
    morph_kernel = get_morphology_kernel(kernel_size, kernel_shape)
    if rois:
        result = image.copy()
        for roi in rois:
            r = roi.as_rect()
            result[r.top : r.bottom, r.left : r.right] = cv2.morphologyEx(
                result[r.top : r.bottom, r.left : r.right],
                cv2.MORPH_OPEN,
                morph_kernel,
                iterations=proc_times,
            )
    else:
        result = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, morph_kernel, iterations=proc_times
        )
    return result


def close(
    image: Any,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
    rois: tuple = (),
    proc_times: int = 1,
):
    """Morphology - Close wrapper
    Arguments:
        image {numpy array} -- Source image
        kernel_size {int} -- kernel size
        kernel_shape {int} -- cv2 constant
        roi -- Region of Interest
        proc_times {int} -- iterations
    Returns:
        numpy array -- closed image
    """
    morph_kernel = get_morphology_kernel(kernel_size, kernel_shape)
    if rois:
        result = image.copy()
        for roi in rois:
            r = roi.as_rect()
            result[r.top : r.bottom, r.left : r.right] = cv2.morphologyEx(
                result[r.top : r.bottom, r.left : r.right],
                cv2.MORPH_CLOSE,
                morph_kernel,
                iterations=proc_times,
            )
    else:
        result = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, morph_kernel, iterations=proc_times
        )
    return result


def dilate(
    image: Any,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
    rois: tuple = (),
    proc_times: int = 1,
):
    """Morphology - Dilate wrapper
    Arguments:
        image {numpy array} -- Source image
        kernel_size {int} -- kernel size
        kernel_shape {int} -- cv2 constant
        roi -- Region of Interrest
        proc_times {int} -- iterations
    Returns:
        numpy array -- dilated image
    """
    morph_kernel = get_morphology_kernel(kernel_size, kernel_shape)
    if rois:
        result = image.copy()
        for roi in rois:
            if roi is not None:
                r = roi.as_rect()
                result[r.top : r.bottom, r.left : r.right] = cv2.dilate(
                    result[r.top : r.bottom, r.left : r.right],
                    morph_kernel,
                    iterations=proc_times,
                )
    else:
        result = cv2.dilate(image, morph_kernel, iterations=proc_times)
    return result


def erode(
    image: Any,
    kernel_size: int = 3,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
    rois: tuple = (),
    proc_times: int = 1,
):
    """Morphology - Erode wrapper
    Arguments:
        image {numpy array} -- Source image
        kernel_size {int} -- kernel size
        kernel_shape {int} -- cv2 constant
        roi -- Region of Interrest
        proc_times {int} -- iterations
    Returns:
        numpy array -- eroded image
    """
    morph_kernel = get_morphology_kernel(kernel_size, kernel_shape)
    if rois:
        result = image.copy()
        for roi in rois:
            if roi is not None:
                r = roi.as_rect()
                result[r.top : r.bottom, r.left : r.right] = cv2.erode(
                    result[r.top : r.bottom, r.left : r.right],
                    morph_kernel,
                    iterations=proc_times,
                )
    else:
        result = cv2.erode(image, morph_kernel, iterations=proc_times)
    return result


class ContourWrapper(object):
    def __init__(self, contour) -> None:
        self.contour = contour
        M = cv2.moments(contour)
        self.cx = M["m10"] / M["m00"]
        self.cy = M["m01"] / M["m00"]
        self.col = None
        self.row_index = None
        self.left, self.top, self.width, self.height = cv2.boundingRect(contour)
        self._area = None

    def __str__(self) -> str:
        return f"(x:{int(self.cx)}, y:{int(self.cy)}) {self.area} {self.row}{self.col}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def index(self):
        return f"{self.row}{self.col}"

    @property
    def max_size(self):
        return max(self.width, self.height)

    @property
    def row(self):
        return chr(self.row_index + 65) if self.row_index is not None else "-"

    @property
    def area(self):
        if self._area is None:
            self._area = cv2.contourArea(self.contour)
        return self._area

    @property
    def row_color(self):
        return (
            to_RGB(C_YELLOW)
            if self.row == "A"
            else to_RGB(C_ORANGE)
            if self.row == "B"
            else to_RGB(C_RED)
            if self.row == "C"
            else to_RGB(C_FUCHSIA)
        )

    @property
    def col_color(self):
        return (
            to_RGB(C_TEAL)
            if self.col == 1
            else to_RGB(C_CABIN_BLUE)
            if self.col == 2
            else to_RGB(C_BLUE_VIOLET)
            if self.col == 3
            else to_RGB(C_BLUE)
            if self.col == 4
            else to_RGB(C_FUCHSIA)
        )

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height


class ContourList(object):
    def __init__(self, mask, external_only: bool = True):
        self._contours = [
            ContourWrapper(c)
            for c in cv2.findContours(
                mask.copy(),
                cv2.RETR_EXTERNAL if external_only is True else cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )[-2:-1][0]
        ]
        self._iter_step = 0

    def __getitem__(self, idx):
        return self._contours[idx]

    def __len__(self):
        return len(self._contours)

    def __iter__(self):
        self._iter_step = -1
        return self

    def __next__(self):
        if self._iter_step < len(self._contours) - 1:
            self._iter_step += 1
            return self._contours[self._iter_step]
        else:
            raise StopIteration

    def update_contours(self, contours):
        self._contours = contours

    def get(self, row: str, col: int) -> ContourWrapper:
        for contour in self._contours:
            if contour.row.lower() == row.lower() and contour.col == col:
                return contour
        else:
            return None

    def max_area(self):
        return sorted(self._contours, key=lambda x: x.area)[-1].area


def apply_mask(image, mask, bckg_luma=0.3, draw_contours: int = -1):
    lum, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    lum = (lum * bckg_luma).astype(np.uint)
    lum[lum >= 255] = 255
    lum = lum.astype(np.uint8)
    background = cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)

    res = cv2.bitwise_or(
        cv2.bitwise_and(background, background, mask=255 - mask),
        cv2.bitwise_and(image, image, mask=mask),
    )
    if draw_contours > 0:
        cv2.drawContours(
            res,
            [c.contour for c in ContourList(mask, external_only=False)],
            -1,
            (255, 0, 255),
            draw_contours,
        )

    return res


def print_contour_threshold(mask, ar_threshiold=1.5, size_thrshold=0.80, canvas=None):
    if canvas is None:
        canvas = np.dstack((mask, mask, mask))
    contours = ContourList(mask)
    max_area = contours.max_area * size_thrshold

    for c in contours:
        colour = (
            to_RGB(C_RED)
            if c.area < max_area
            else to_RGB(C_ORANGE)
            if c.width / c.height > ar_threshiold or c.height / c.width > ar_threshiold
            else to_RGB(C_GREEN)
        )
        cv2.drawContours(canvas, [c.contour], 0, colour, -1)
    return canvas


def clean_contours(
    mask,
    ar_threshiold=1.5,
    size_thrshold=0.80,
    kernel_size=0,
    kernel_shape: int = cv2.MORPH_ELLIPSE,
    open_count=0,
    erode_count=0,
):
    contours = ContourList(mask)

    # Remove non circular contours
    contours.update_contours(
        [
            c
            for c in contours
            if c.width / c.height < ar_threshiold
            and c.height
            or c.width < ar_threshiold
        ]
    )
    if len(contours) == 0:
        return mask

    # Remove large or small contours
    threshold_area = sorted(contours, key=lambda x: x.area)[-1].area * size_thrshold
    contours.update_contours([c for c in contours if c.area > threshold_area])

    mask = cv2.drawContours(
        np.zeros_like(mask), [c.contour for c in contours], -1, (255), -1
    )
    if kernel_size == 0 or open_count == 0 or erode_count == 0:
        return mask
    else:
        return erode(
            open(
                image=mask,
                kernel_size=kernel_size,
                kernel_shape=kernel_shape,
                proc_times=open_count,
            ),
            kernel_size=kernel_size,
            kernel_shape=kernel_shape,
            proc_times=erode_count,
        )


def print_single_contour(
    mask,
    contours: ContourList,
    row,
    col,
    canvas=None,
    rect_thickness=4,
):
    if canvas is None:
        canvas = np.dstack((mask, mask, mask))

    fnt = (cv2.FONT_HERSHEY_SIMPLEX, 0.6)
    fnt_scale = 1.5
    fnt_thickness = 3

    c = contours.get(row=row, col=col)
    if c is None:
        return canvas
    padding = 10
    cv2.rectangle(
        canvas,
        (c.left - padding, c.top - padding),
        (c.right + padding, c.bottom + padding),
        (0, 255, 255),
        rect_thickness,
    )
    cv2.circle(canvas, (int(c.left), int(c.top)), 40, c.col_color, 80)
    cv2.circle(canvas, (int(c.left), int(c.top)), 20, c.row_color, 40)
    cv2.putText(
        canvas,
        str(c.index),
        (int(c.left) - 30, int(c.top) + 10),
        fnt[0],
        fnt_scale,
        (0, 0, 0),
        fnt_thickness,
    )

    return canvas


def print_contours_indexes(mask, contours, canvas=None):
    if canvas is None:
        canvas = np.dstack((mask, mask, mask))

    fnt = (cv2.FONT_HERSHEY_SIMPLEX, 0.6)
    fnt_scale = 1.5
    fnt_thickness = 3

    for c in contours:
        cv2.circle(canvas, (int(c.left), int(c.top)), 40, c.col_color, 80)
        cv2.circle(canvas, (int(c.left), int(c.top)), 20, c.row_color, 40)
        cv2.putText(
            canvas,
            str(c.index),
            (int(c.left) - 30, int(c.top) + 10),
            fnt[0],
            fnt_scale,
            (0, 0, 0),
            fnt_thickness,
        )

    return canvas


def get_leaf_disk(image, contours, row, col, mask=None, padding=0) -> ContourWrapper:
    c = contours.get(row=row, col=col)
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=erode(mask, kernel_size=15))
    res = A.crop(
        image,
        c.left - padding,
        c.top - padding,
        c.right + padding,
        c.bottom + padding,
    )
    return res


def index_contours(mask) -> list:
    contours = ContourList(mask=mask)

    X = [[c.cx, 1] for c in contours]
    ms = MeanShift(bandwidth=100, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_

    labels = max(labels) - labels
    labels_unique = np.unique(labels)

    for contour, label in zip(contours, labels):
        contour.col = label

    if len(labels_unique < 4):
        prev_min_min = 0
        inc_labels = [[i, 0] for i in range(len(labels_unique))]
        for label in labels_unique:
            cur_lbl_contours = [c for c in contours if c.col == label]
            max_width = sorted(cur_lbl_contours, key=lambda x: x.max_size)[-1].max_size
            min_left = (
                sorted(cur_lbl_contours, key=lambda x: x.cx)[0].cx - max_width / 2
            )
            i = 1
            while min_left - prev_min_min > i * 1.1 * max_width:
                inc_labels[label][1] += 1
                i += 1
            prev_min_min = min_left + max_width

        for pos, inc in reversed(inc_labels):
            labels[labels >= pos] += inc

        for contour, label in zip(contours, labels):
            contour.col = label

        labels_unique = np.unique(labels)

    for label in labels_unique:
        label_contour = sorted(
            [c for c in contours if c.col == label], key=lambda x: x.cy
        )
        for i, label_column in enumerate(label_contour):
            label_column.row_index = i

    for contour in contours:
        contour.col += 1

    return contours
