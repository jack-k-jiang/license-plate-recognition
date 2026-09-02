import glob
import json
import os
import re
from collections import Counter

TEST_ROOT = os.path.join(os.path.dirname(__file__), "..", "test")



def load_frame_annotations(category: str, video_id: str) -> dict[int, dict]:
    """Parse every frame{n}.json for one video, in frame-number order.
    Returns {frame_num: {car_id: {"carBox": ..., "licPoly": ..., "licNumber": ...}}}.
    Empty {} frames are valid (no plate visible) - keep them, don't skip."""

    jdir = os.path.join(TEST_ROOT,"jsons",category, video_id)
    paths = glob.glob(os.path.join(jdir, "frame*.json"))

    frames = {}
    for path in paths:
        # "frame123.json" -> 123. Don't sort the *strings* - "frame10" would 
        # sort before "frame2" lexicographically, which is wrong.

        match = re.match(r"frame(\d+)\.json$", os.path.basename(path))
        frame_num = int(match.group(1))

        with open(path) as f:
            frames[frame_num] = json.load(f)

    return frames

def normalize_plate_text(text: str) -> str:
    """Keeps only 0-9A-Z, uppercased: fast-plate-ocr's alphabet has neither
    Chinese characters nor punctuation like licNumber's "-" separator, so
    anything else can never be matched and shouldn't be compared. Used on
    both ground truth (here) and predicted text (in the accuracy harness) so
    both sides of a comparison go through the exact same normalization."""
    return re.sub(r'[^0-9A-Za-z]', '', text).upper()


def denormalize(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[int, int]]:
    """Convert LSV-LP's [0,1]-normalized points to pixel coordinates for one frame."""
    denormalized = []

    for t in points:
        x_norm = t[0]
        y_norm = t[1]
        x_pixel = round(width * x_norm)
        y_pixel = round(height * y_norm)
        denormalized.append((x_pixel, y_pixel))

    return denormalized

def build_ground_truth_tracks(category: str, video_id: str) -> dict[str, dict]:
    """Collapse per-frame annotations into one entry per (category, video_id, car_id).

    Returns {car_id: {
        "text": <majority-vote alphanumeric plate text, or None if the car
            has no clean (non-'#') reading anywhere in the video>,
        "frames": [every frame_num the car was annotated in, clean or not],
        "n_distinct_readings": <count of distinct clean readings seen>,
        "unanimous": <True if all clean readings agreed, False if they
            didn't, None if there were no clean readings to compare>,
    }}.

    "frames" is unconditional: it's needed later for licPoly-based work
    (perspective correction, matching against a predicted track), and
    licPoly is annotated at a lower legibility bar than licNumber, so
    plenty of frames have usable geometry despite unclear text.

    "text" is resolved by majority vote across clean readings rather than
    "trust the first clean frame": this is human-labeled ground truth, so
    disagreement between two clean readings of the same physical plate is
    far more likely to be a one-off transcription slip than genuine
    ambiguity - majority vote lands on the correct string in the
    overwhelming case, and "unanimous"/"n_distinct_readings" make any
    disagreement visible instead of silently smoothing over it.
    """

    frames = load_frame_annotations(category, video_id)
    tracks: dict[str, dict] = {}

    for num, frame in frames.items():
        for carId, idInfo in frame.items():
            entry = tracks.setdefault(carId, {"frames": [], "readings": []})
            entry["frames"].append(num)

            licNumber = idInfo["licNumber"]
            if "#" in licNumber:
                continue

            entry["readings"].append(normalize_plate_text(licNumber))

    ground_truth = {}
    for carId, entry in tracks.items():
        readings = entry["readings"]
        if readings:
            counts = Counter(readings)
            text, _ = counts.most_common(1)[0]
            n_distinct = len(counts)
            unanimous = n_distinct == 1
        else:
            text = None
            n_distinct = 0
            unanimous = None

        ground_truth[carId] = {
            "text": text,
            "frames": entry["frames"],
            "n_distinct_readings": n_distinct,
            "unanimous": unanimous,
        }

    return ground_truth