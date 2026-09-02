"""Perspective correction: rectify a tilted/angled plate crop to a
fronto-parallel rectangle before handing it to OCR. YOLO only gives an
axis-aligned bounding box, not the plate's actual corners, so this finds
the corners itself via classical CV (no model, no training data needed) -
grayscale -> edges -> contours -> approximate to 4 points -> 4-point
perspective warp. Falls back to the original crop, untouched, whenever it
can't find a confident quadrilateral - this should only ever be able to
help, never make a crop worse than not running it at all.
"""

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 arbitrary points as (top-left, top-right, bottom-right,
    bottom-left) - required for a correct perspective warp (getPerspectiveTransform
    needs src/dst points in matching order, not just matching identity).

    Standard trick: top-left has the smallest x+y sum, bottom-right the
    largest; top-right has the smallest y-x difference, bottom-left the
    largest. Works regardless of what order the 4 points originally came in.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def find_plate_corners(crop: np.ndarray):
    """Attempts to find the plate's 4 corners within a YOLO-cropped region.
    Returns an ordered (4, 2) float32 array, or None if nothing confident
    was found (e.g. contour didn't approximate to ~4 points, or the
    candidate region doesn't plausibly cover most of the crop - a plate
    detector's crop should mostly *be* the plate, so a "quadrilateral"
    covering a small corner of the image is more likely noise than the
    actual plate boundary).
    """
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    best_quad = None
    best_area = 0

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        # The plate detector already cropped tightly to the plate, so the
        # real plate boundary should cover most of the crop - reject small
        # quads (stray edges/glare/etc.) rather than warp to noise.
        if area < 0.3 * crop_area:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    if best_quad is None:
        return None

    return order_points(best_quad.reshape(4, 2).astype("float32"))


def rectify_plate(crop: np.ndarray) -> np.ndarray:
    """Warps `crop` to a fronto-parallel rectangle using its detected
    corners. Returns the original crop, unmodified, if no confident
    quadrilateral was found - correction should only ever help, never
    hurt, so "can't find corners" degrades to "do nothing" rather than
    guessing.
    """
    corners = find_plate_corners(crop)
    if corners is None:
        return crop

    (tl, tr, br, bl) = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    target_w = int(max(width_top, width_bottom))
    target_h = int(max(height_left, height_right))
    if target_w <= 0 or target_h <= 0:
        return crop

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(crop, matrix, (target_w, target_h))
