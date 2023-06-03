"""
Proof of concept reading of punched cards using photos of
punched cards and OpenCV.

Jonas Nockert, 2017.


1. Find upper left corner (the diagonally cut corner) by checking
   which corner that has most background pixels.

2. Now that we know the orientation of the card, use a grid approach
   to go through each potentially punched position to see if it is
   punched or not, again by counting background pixels. Each column
   forms a bit pattern.

3. Match this bit pattern with a fixed 64 character map (different
   punched cards can have different maps).

"""
from math import sqrt
from sys import argv

import cv2
import numpy as np

from charmap_ibm_029 import charmap


class ImageError(Exception):
    """Raised when image could not be processed."""


def line_intersection(p0, p1, q0, q1):
    """Returns intersection point if lines intersect, otherwise None."""
    vp = p1 - p0
    vq = q1 - q0

    vpx = vp[0][0]
    vpy = vp[0][1]
    vqx = vq[0][0]
    vqy = vq[0][1]

    delta_x = (p0 - q0)[0][0]
    delta_y = (p0 - q0)[0][1]
    p0x = p0[0][0]
    p0y = p0[0][1]

    det = (-vqx * vpy + vpx * vqy)
    if abs(det) < 1e-20:
        return None

    s = (-vpy * (delta_x) + vpx * (delta_y)) / det
    t = (vqx * (delta_y) - vqy * (delta_x)) / det

    return np.array((p0x + (t * vpx), p0y + (t * vpy)))


def find_corner (thresh):
    """Figure out which corner is the upper left corner.

    The upper left corner of a punched card is cut diagonally and not
    rounded. By checking how much background color there is in each
    corner, we can figure out which corner is the upper left one.

    """
    # Find minimum polygon that surrounds card.
    contours, hierarchy = cv2.findContours(thresh.copy(),
                                                cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_NONE)
    # Approximate bounding box to a small number of points. Since the cards
    # have three rounded corners and one cut corner, the extra points will
    # be used to approximate this and the four four longest line segments
    # are the sides of the card.
    d = 0
    while True:
        d = d + 1;
        approx = cv2.approxPolyDP(contours[0], d, True);
        if len(approx) <= 8:
            break
    # cv2.drawContours(imcolor, [approx], 0, (0, 0, 255), 2)

    # Find the four longest corners.
    lines = []
    approx_len = len(approx)
    for i in range(approx_len):
        next_i = (i + 1) % approx_len
        dist = np.linalg.norm(approx[i] - approx[next_i])
        lines.append([dist, approx[i], approx[next_i]])
    lines.sort(reverse=True)

    corner_pts = []
    corner_pts.append(line_intersection(lines[0][1], lines[0][2],
                                        lines[2][1], lines[2][2]))
    corner_pts.append(line_intersection(lines[0][1], lines[0][2],
                                        lines[3][1], lines[3][2]))
    corner_pts.append(line_intersection(lines[1][1], lines[1][2],
                                        lines[2][1], lines[2][2]))
    corner_pts.append(line_intersection(lines[1][1], lines[1][2],
                                        lines[3][1], lines[3][2]))

    # Order points upper left, upper right, lower right, lower left.
    x_ordered = sorted(corner_pts, key=lambda corner: corner[0])
    left_corners = x_ordered[:2]
    right_corners = x_ordered[2:]
    xy_ordered = sorted(left_corners, key=lambda corner: corner[1])
    c1 = np.array((xy_ordered[0][0], xy_ordered[0][1]), dtype=np.int32)
    c4 = np.array((xy_ordered[1][0], xy_ordered[1][1]), dtype=np.int32)
    xy_ordered = sorted(right_corners, key=lambda corner: corner[1])
    c2 = np.array((xy_ordered[0][0], xy_ordered[0][1]), dtype=np.int32)
    c3 = np.array((xy_ordered[1][0], xy_ordered[1][1]), dtype=np.int32)

    # cv2.circle(imcolor, (int(c1[0]), int(c1[1])), 10, (255, 255, 0), 1)
    # cv2.circle(imcolor, (int(c2[0]), int(c2[1])), 10, (255, 255, 0), 1)
    # cv2.circle(imcolor, (int(c3[0]), int(c3[1])), 10, (255, 255, 0), 1)
    # cv2.circle(imcolor, (int(c4[0]), int(c4[1])), 10, (255, 255, 0), 1)
    # pts = np.vstack((c1, c2, c3, c4))
    # cv2.polylines(imcolor, np.int32([pts]), True, (0,0,255), 1)

    # TODO Remove hard coded 5% corner side length.
    corner_len = max(np.linalg.norm(c2 - c1), np.linalg.norm(c4 - c1)) * 0.05
    corner_pixel_count = [0] * 4

    vc1 = (c2 - c1) / np.linalg.norm(c2 - c1) * corner_len
    vc2 = (c4 - c1) / np.linalg.norm(c4 - c1) * corner_len
    pts = np.array((c1, c1 + vc1, c1 + vc1 + vc2, c1 + vc2), dtype=np.int32)
    # cv2.polylines(imcolor, np.int32([pts]), True, (0,0,255), 1)
    corner_pixel_count[0] = get_corner_black_pixel_count(thresh, pts)

    vc1 = (c1 - c2) / np.linalg.norm(c1 - c2) * corner_len
    vc2 = (c3 - c2) / np.linalg.norm(c3 - c2) * corner_len
    pts = np.array((c2, c2 + vc1, c2 + vc1 + vc2, c2 + vc2), dtype=np.int32)
    # cv2.polylines(imcolor, [pts], True, (0,0,255), 1)
    corner_pixel_count[1] = get_corner_black_pixel_count(thresh, pts)

    vc1 = (c2 - c3) / np.linalg.norm(c2 - c3) * corner_len
    vc2 = (c4 - c3) / np.linalg.norm(c4 - c3) * corner_len
    pts = np.array((c3, c3 + vc1, c3 + vc1 + vc2, c3 + vc2), dtype=np.int32)
    # cv2.polylines(imcolor, np.int32([pts]), True, (0,0,255), 1)
    corner_pixel_count[2] = get_corner_black_pixel_count(thresh, pts)

    vc1 = (c3 - c4) / np.linalg.norm(c3 - c4) * corner_len
    vc2 = (c1 - c4) / np.linalg.norm(c1 - c4) * corner_len
    pts = np.array((c4, c4 + vc1, c4 + vc1 + vc2, c4 + vc2), dtype=np.int32)
    # cv2.polylines(imcolor, np.int32([pts]), True, (0,0,255), 1)
    corner_pixel_count[3] = get_corner_black_pixel_count(thresh, pts)

    # The corner with most black pixels is the upper left corner.
    max_c = max(corner_pixel_count)

    # Return upper left, upper right, lower right, lower left corners.
    if max_c == corner_pixel_count[0]:
        return (c1, c2, c3, c4)
    elif max_c == corner_pixel_count[1]:
        return (c2, c3, c4, c1)
    elif max_c == corner_pixel_count[2]:
        return (c3, c4, c1, c2)
    else:
        return (c4, c1, c2, c3)


def get_corner_black_pixel_count (thresh, pts):
    """Get number of black pixels for a specific corner.

    The corner with the most black pixels is the diagonally cut
    upper left corner.
    """
    (rows, cols) = thresh.shape
    maskframe = np.full((rows, cols), 255, dtype=np.uint8)
    cv2.fillConvexPoly(maskframe, pts, 0)
    corner = thresh.copy()
    cv2.bitwise_or(corner, 255, corner, mask=maskframe)
    corner = 255 - corner
    return cv2.countNonZero(corner)


def transform(thresh, upper_left, upper_right, lower_right, lower_left):
    """Perspective transform image."""
    contours, hierarchy = cv2.findContours(thresh,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.float32(box)

    x_ordered = sorted(box, key=lambda corner: corner[0])
    left_corners = x_ordered[:2]
    right_corners = x_ordered[2:]
    xy_ordered = sorted(left_corners, key=lambda corner: corner[1])
    c1 = np.array((xy_ordered[0][0], xy_ordered[0][1]), dtype=np.float32)
    c4 = np.array((xy_ordered[1][0], xy_ordered[1][1]), dtype=np.float32)
    xy_ordered = sorted(right_corners, key=lambda corner: corner[1])
    c2 = np.array((xy_ordered[0][0], xy_ordered[0][1]), dtype=np.float32)
    c3 = np.array((xy_ordered[1][0], xy_ordered[1][1]), dtype=np.float32)

    w = np.linalg.norm(c2 - c1)
    h = np.linalg.norm(c4 - c1)

    src = np.array((upper_left, upper_right, lower_right, lower_left),
                   dtype=np.float32)
    dst = np.array(([0, 0], [w, 0], [w, h], [0, h]), dtype=np.float32)

    transformation = cv2.getPerspectiveTransform(src, dst)
    imdst = cv2.warpPerspective(thresh, transformation, (w, h))
    return imdst


def read_card(image_path):
    """Read a punched card from a photo."""
    cardim = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Dilate and threshold in order to try and remove text and
    # other information on the card
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(cardim, kernel, iterations = 1)
    erodation = cv2.erode(dilation, kernel, iterations = 1)
    ret, thresh = cv2.threshold(erodation, 127, 255, 0)

    (upper_left, upper_right, lower_right, lower_left) = find_corner(thresh)
    thresh_m = transform(thresh, upper_left, upper_right, lower_right,
                         lower_left)
    (height, width) = thresh_m.shape
    # imcolor = cv2.cvtColor(thresh_m, cv2.COLOR_GRAY2BGR)

    line = ""
    # TODO Remove hard coded offsets for punch card holes.
    delta_x = width / 84.5
    for i in range(80):
        x = int(delta_x * 2.8 + delta_x * float(i))
        # cv2.line(imcolor, (x, 0), (x, height), (255, 0, 0))

        holes = [False] * 12
        for jr in range(12):
            j = 11 - jr
            y = int(height * 0.08 + height / 12.90 * float(j))
            # cv2.circle(imcolor, (x, y), 2, (255, 0, 255), 1)
            hole = 255 - thresh_m[y-2:y+2, x-2:x+2]
            holes[jr] = cv2.countNonZero(hole) > 1

        bits = 0
        for b in range(12):
            if holes[b]:
                bits += pow(2, b)

        if not bits:
            line += " "
        elif str(bits) not in charmap:
            print("Error: column 0x{:x} not in character map.".format(bits))
        else:
            line += charmap[str(bits)]

    # cv2.imshow('image', imcolor)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return line


if __name__ == "__main__":
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    if len(argv) <= 1:
        print("Usage: {:s} [image]".format(argv[0]))
    else:
        print(read_card(argv[1]))
