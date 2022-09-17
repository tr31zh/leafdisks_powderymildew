import numpy as np

import cv2

from sklearn.cluster import MeanShift, estimate_bandwidth


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
        return f"(x:{int(self.cx)}, y:{int(self.cy)}) {self.row}{self.col}"

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


def apply_mask(image, mask, bckg_luma=0.3):
    lum, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    lum = (lum * bckg_luma).astype(np.uint)
    lum[lum >= 255] = 255
    lum = lum.astype(np.uint8)
    background = cv2.cvtColor(cv2.merge((lum, a, b)), cv2.COLOR_LAB2BGR)

    return cv2.bitwise_or(
        cv2.bitwise_and(background, background, mask=255 - mask),
        cv2.bitwise_and(image, image, mask=mask),
    )


def print_contour_threshold(mask, threshold=0.8):
    contours = [
        ContourWrapper(c)
        for c in cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
            -2:-1
        ][0]
    ]
    max_area = sorted(contours, key=lambda x: x.area)[-1].area * threshold

    canvas = np.dstack((np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)))

    for contour in contours:
        cv2.drawContours(
            canvas,
            [contour.contour],
            0,
            to_RGB(C_RED) if contour.area < max_area else to_RGB(C_GREEN),
            -1,
        )
    return canvas


def print_contours_indexs(mask, contours, canvas=None):
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


def index_contours(mask, threshold=0.8) -> list:
    contours = [
        ContourWrapper(c)
        for c in cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
            -2:-1
        ][0]
    ]
    max_area = sorted(contours, key=lambda x: x.area)[-1].area * threshold
    contours = sorted(
        [c for c in contours if c.area > max_area],
        key=lambda x: cv2.minAreaRect(x.contour),
    )

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
            if min_left - prev_min_min > 1.1 * max_width:
                inc_labels[label][1] += 1
            prev_min_min = min_left + max_width

        for pos, inc in reversed(inc_labels):
            labels[labels >= pos] += inc

    for label in labels_unique:
        label_contour = sorted(
            [c for c in contours if c.col == label], key=lambda x: x.cy
        )
        for i, label_column in enumerate(label_contour):
            label_column.row_index = i

    for contour in contours:
        contour.col += 1

    return contours
