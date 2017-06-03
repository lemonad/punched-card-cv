"""
Proof of concept reading of punched cards using photos of
punched cards and OpenCV.


Jonas Nockert, 2017.

"""
from math import copysign, sqrt
from sys import argv, exit

import cv2
import numpy as np

from charmap_ibm_029 import charmap


def find_corner(thresh, box, x_direction, y_direction, corner_len):
    """Figure out which corner is the upper left corner.

    The upper left corner of a punched card is cut diagonally and not
    rounded. By checking how much background color there is in each
    corner, we can figure out which corner is the upper left one.

    """
    # Corner 1.
    pts = np.array([box[0],
                    box[0] - x_direction * corner_len,
                    box[0] - x_direction * corner_len - y_direction * corner_len,
                    box[0] - y_direction * corner_len],
                   np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.polylines(imcolor, [pts], True, (0, 255, 255), 5)
    c1 = getBlackCount(thresh, pts)

    # Corner 2.
    pts = np.array([box[1],
                    box[1] + x_direction * corner_len,
                    box[1] + x_direction * corner_len - y_direction * corner_len,
                    box[1] - y_direction * corner_len],
                   np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.polylines(imcolor, [pts], True, (0, 255, 255), 5)
    c2 = getBlackCount(thresh, pts)

    # Corner 3.
    pts = np.array([box[2],
                    box[2] + x_direction * corner_len,
                    box[2] + x_direction * corner_len + y_direction * corner_len,
                    box[2] + y_direction * corner_len],
                   np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.polylines(imcolor, [pts], True, (0, 255, 255), 5)
    c3 = getBlackCount(thresh, pts)

    # Corner 4.
    pts = np.array([box[3],
                    box[3] - x_direction * corner_len,
                    box[3] - x_direction * corner_len + y_direction * corner_len,
                    box[3] + y_direction * corner_len],
                   np.int32)
    pts = pts.reshape((-1,1,2))
    # cv2.polylines(imcolor, [pts], True, (0, 255, 255), 5)
    c4 = getBlackCount(thresh, pts)

    max_c = max(c1, c2, c3, c4)

    if max_c == c1:
        return (box[0], box[1], box[2], box[3])
    elif max_c == c2:
        return (box[1], box[2], box[3], box[0])
    elif max_c == c3:
        return (box[2], box[3], box[0], box[1])
    else:
        return (box[3], box[0], box[1], box[2])


def getBlackCount(thresh, pts):
    """Get number of black (punched) pixels for a specific hole."""
    (rows, cols) = thresh.shape
    maskframe = np.full((rows, cols), 255, dtype=np.uint8)
    cv2.fillConvexPoly(maskframe, pts, 0)
    corner = thresh.copy()
    cv2.bitwise_or(corner, 255, corner, mask=maskframe)
    corner = 255 - corner
    return cv2.countNonZero(corner)


def readcard(image_path):
    """Read a punch card from a photo."""
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cardim = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(cardim, kernel, iterations = 1)
    ret, thresh = cv2.threshold(dilation, 127, 255, 0)

    im2, contours, hierarchy = cv2.findContours(thresh,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(thresh, contours, -1, (0,0,0), 3)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    dim1 = sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2))
    dim2 = sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2))
    width = max(dim1, dim2)
    height = min(dim1, dim2)

    (rows, cols) = thresh.shape
    maskframe = np.full((rows, cols), 255, dtype=np.uint8)
    cv2.fillConvexPoly(maskframe, box, 0)
    cv2.bitwise_or(thresh, 255, thresh, mask=maskframe)

    imcolor = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    x_direction = (box[0] - box[1]) / dim1
    y_direction = (box[0] - box[3]) / dim2
    # TODO Remove hard coded 5% corner side length.
    corner_len = width * 0.05

    (upper_left, upper_right, lower_right, lower_left) = find_corner(
        thresh, box, x_direction, y_direction, corner_len)

    cv2.circle(imcolor, (upper_left[0], upper_left[1]), 20, (0, 255, 0), 5)

    line = ""
    # TODO Remove hard coded offsets for punch card holes.
    upper_delta_x = (upper_right - upper_left) / 84.9
    lower_delta_x = (lower_right - lower_left) / 84.9
    left_delta_y = (upper_left - lower_left) / 13.1
    right_delta_y = (upper_right - lower_right) / 13.1
    for i in range(80):
        ul = upper_left + upper_delta_x * 3.0 + upper_delta_x * float(i)
        ll = lower_left + lower_delta_x * 3.0 + lower_delta_x * float(i)
        cv2.line(imcolor,
                 (int(ul[0]), int(ul[1])),
                 (int(ll[0]), int(ll[1])),
                 (255, 0))

        holes = [False] * 12
        for jr in range(12):
            j = 11 - jr
            loc = ll + (ul - ll) / 13.15 * 1.12 + (ul - ll) / 13.15 * float(j)
            x = int(loc[0])
            y = int(loc[1])
            cv2.circle(imcolor,
                       (x, y),
                       2, (255, 0, 255), 1)
            hole = 255 - thresh[y-2:y+2, x-2:x+2]
            holes[j] = cv2.countNonZero(hole) > 1

        bits = 0
        for b in range(12):
            if holes[b]:
                bits += pow(2, b)

        if not bits:
            line += " "
        else:
            line += charmap[str(bits)]

    cv2.drawContours(imcolor, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('image', imcolor)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return line


if __name__ == "__main__":
    if len(argv) <= 1:
        print("Usage: {:s} [image]".format(argv[0]))
    else:
        print(readcard(argv[1]))
