"""Accuracy evaluation harness: runs the full pipeline across the evaluation
corpus (LSV-LP subset + acar.mp4/acar2.mp4), matches predicted tracks to
ground truth (LSV-LP only - acar videos have no annotations to score
against), and computes exact-match accuracy with aggregation on vs. off.
See ROADMAP.md "Phase 1 — CV pipeline" for the design.
"""

import json
import runpy

import cv2

from loader import build_ground_truth_tracks, load_frame_annotations, normalize_plate_text
from matcher import match_predicted_to_ground_truth

_module = runpy.run_path("number-plate-recognition.py", run_name="not_main")
LPR = _module["LPR"]


def process_video(model_path: str, source, cap_backend=None, max_frames: int = None):
    """Runs full per-frame detection+OCR across one video with a *fresh*
    LPR instance (so track IDs and vote history never leak between videos).

    Returns:
      predicted_boxes: {frame_num: {track_id: (pt1, pt2)}}, normalized
        [0,1] - the shape match_predicted_to_ground_truth expects.
      raw_text_by_track: {track_id: (text, confidence) from the *last*
        frame that track was seen} - the "aggregation off" baseline: what
        a system with no memory would report as of the last frame it saw
        the track, since that's the natural behavior without stabilize_text.
      stable_text_by_track: {track_id: stabilize_text's final majority-vote
        result} - the "aggregation on" result.
      total_frames: frames actually read.
    """
    lpr = LPR(model_path=model_path)
    cap = cv2.VideoCapture(source, cap_backend) if cap_backend is not None else cv2.VideoCapture(source)

    # {track_id: {frame_num: box}} - matcher.py's match_predicted_to_ground_truth
    # expects this nesting order, NOT {frame_num: {track_id: box}}. Track IDs and
    # frame numbers are both small ints, so getting this backwards doesn't crash -
    # it silently matches on coincidental numeric collisions between two unrelated
    # ID spaces instead of real frame overlap. (Found exactly this bug for real:
    # see ROADMAP.md.)
    predicted_boxes: dict = {}
    raw_text_by_track = {}
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if max_frames and frame_num > max_frames:
            break
        fh, fw = frame.shape[:2]

        boxes, track_ids = lpr.detect_plates(frame)
        for bbox, track_id in zip(boxes, track_ids):
            if track_id is None:
                continue
            track_id = int(track_id)
            x1, y1, x2, y2 = bbox
            predicted_boxes.setdefault(track_id, {})[frame_num] = (
                (x1 / fw, y1 / fh), (x2 / fw, y2 / fh)
            )

            text, confidence = lpr.extract_text(frame, bbox)
            raw_text_by_track[track_id] = (normalize_plate_text(text), confidence)
            lpr.stabilize_text(track_id, text, confidence)

    cap.release()

    stable_text_by_track = {}
    for track_id in raw_text_by_track:
        # Empty-text call doesn't add a vote, just returns the current majority -
        # reuses the already-tested stabilize_text logic instead of reaching
        # into lpr.plate_history directly.
        stable_text_by_track[track_id] = normalize_plate_text(lpr.stabilize_text(track_id, "", 0.0))

    return predicted_boxes, raw_text_by_track, stable_text_by_track, frame_num


def evaluate_lsvlp_video(model_path: str, category: str, video_id: str):
    """Runs one LSV-LP video and scores it against its own ground truth."""
    source = f"../test/videos/{category}/{video_id}/frame%d.jpg"
    predicted_boxes, raw_text_by_track, stable_text_by_track, total_frames = process_video(
        model_path, source, cap_backend=cv2.CAP_IMAGES
    )

    frame_annotations = load_frame_annotations(category, video_id)
    ground_truth = build_ground_truth_tracks(category, video_id)
    matches = match_predicted_to_ground_truth(predicted_boxes, frame_annotations)

    scored = []
    for track_id, car_id in matches.items():
        if car_id is None:
            continue
        gt_text = ground_truth.get(car_id, {}).get("text")
        if gt_text is None:
            continue  # no usable ground truth for this car (every reading was '#')

        raw_text, _ = raw_text_by_track.get(track_id, ("", 0.0))
        stable_text = stable_text_by_track.get(track_id, "")

        scored.append({
            "track_id": track_id,
            "car_id": car_id,
            "ground_truth": gt_text,
            "raw_text": raw_text,
            "stable_text": stable_text,
            "raw_correct": raw_text == gt_text,
            "stable_correct": stable_text == gt_text,
        })

    return {
        "total_frames": total_frames,
        "total_tracks": len(raw_text_by_track),
        "scored_tracks": scored,
    }


def run_full_evaluation(
    model_path: str,
    subset_path: str = "lsvlp_video_subset.json",
    extra_videos: list = None,
    limit_per_category: int = None,
):
    """Runs the full evaluation corpus: every LSV-LP video in subset_path
    (scored against ground truth), plus extra_videos - e.g. acar.mp4/
    acar2.mp4 - counted toward frames/tracks but not scored (no ground
    truth exists for them).
    """
    subset = json.load(open(subset_path))
    all_scored = []
    total_frames = 0
    total_tracks = 0

    for category, video_ids in subset.items():
        ids = video_ids[:limit_per_category] if limit_per_category else video_ids
        for video_id in ids:
            result = evaluate_lsvlp_video(model_path, category, video_id)
            total_frames += result["total_frames"]
            total_tracks += result["total_tracks"]
            all_scored.extend(result["scored_tracks"])
            print(
                f"{category}/{video_id}: frames={result['total_frames']} "
                f"tracks={result['total_tracks']} scored={len(result['scored_tracks'])}"
            )

    for source in (extra_videos or []):
        _boxes, raw_text_by_track, _stable, frames = process_video(model_path, source)
        total_frames += frames
        total_tracks += len(raw_text_by_track)
        print(f"{source}: frames={frames} tracks={len(raw_text_by_track)} (no ground truth to score)")

    n = len(all_scored)
    raw_accuracy = sum(s["raw_correct"] for s in all_scored) / n if n else 0.0
    stable_accuracy = sum(s["stable_correct"] for s in all_scored) / n if n else 0.0

    return {
        "total_frames": total_frames,
        "total_tracks": total_tracks,
        "scored_tracks": n,
        "raw_accuracy": raw_accuracy,
        "stable_accuracy": stable_accuracy,
        "all_scored": all_scored,
    }
