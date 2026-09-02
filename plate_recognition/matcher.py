


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU between two axis-aligned boxes, each ((x1, y1), (x2, y2)) with
    corners in *arbirtrary* order (not assumed top-left/bottom-right - same
    gotcha as carBox)."""
    ax1, ax2 = sorted([box_a[0][0], box_a[1][0]])
    ay1, ay2 = sorted([box_a[0][1], box_a[1][1]])
    bx1, bx2 = sorted([box_b[0][0], box_b[1][0]])
    by1, by2 = sorted([box_b[0][1], box_b[1][1]])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)  # 0 if boxes don't overlap

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def _licpoly_to_bbox(lic_poly: list[tuple]) -> tuple:
    """Collapse a (possibly tilted) 4-point licPoly into an axis-aligned box,
    matching the fidelity of our own predicted boxes (also axis-aligned,
    from YOLO's .xyxy) - true 4-point polygon IoU would be false precision
    when one side of the comparison was never more than a rectangle."""
    xs = [p[0] for p in lic_poly]
    ys = [p[1] for p in lic_poly]
    return (min(xs), min(ys)), (max(xs), max(ys))


def match_predicted_to_ground_truth(
    predicted_tracks: dict[int, dict[int, tuple]],  # {track_id: {frame_num: (pt1, pt2)}}, normalized [0,1]
    frame_annotations: dict[int, dict],  # load_frame_annotations() output
    iou_threshold: float = 0.5,
) -> dict[int, str | None]:
    """For each predicted track_id, find the LSV-LP car_id (from licPoly,
    not carBox) whose plate box overlaps it most across shared frames.
    Returns {predicted_track_id: matched_car_id or None}.

    Aggregation: mean IoU across every frame both tracks share, not "any
    single frame clears the threshold" - a single lucky/unlucky frame
    shouldn't decide a match either way when there are many shared frames
    to average over.

    iou_threshold applies to that mean, not per-frame - the same 0.5 the
    LSV-LP paper itself uses to call a detection a true positive (Section
    5.3), reused here to keep "this is the same box" consistent with how
    the dataset's own creators define it.

    Many-to-one is allowed on purpose: if our tracker ever fragments one
    real vehicle into two predicted track_ids, both legitimately match the
    same ground-truth car, and forcing a strict one-to-one assignment would
    throw away a real signal (each fragment gets its own shot at reading
    the plate) rather than better reflect reality.
    """
    # Every ground-truth car's plate box, indexed by frame, so we can look
    # up "what box did car X have on frame N" without rescanning per pair.
    gt_boxes: dict[str, dict[int, tuple]] = {}
    for frame_num, cars in frame_annotations.items():
        for car_id, info in cars.items():
            gt_boxes.setdefault(car_id, {})[frame_num] = _licpoly_to_bbox(info["licPoly"])

    matches: dict[int, str | None] = {}
    for track_id, pred_frames in predicted_tracks.items():
        best_car_id = None
        best_score = 0.0

        for car_id, car_frames in gt_boxes.items():
            shared_frames = pred_frames.keys() & car_frames.keys()
            if not shared_frames:
                continue  # never co-occurred - can't be the same plate

            mean_iou = sum(
                compute_iou(pred_frames[f], car_frames[f]) for f in shared_frames
            ) / len(shared_frames)

            if mean_iou > best_score:
                best_score = mean_iou
                best_car_id = car_id

        matches[track_id] = best_car_id if best_score >= iou_threshold else None

    return matches