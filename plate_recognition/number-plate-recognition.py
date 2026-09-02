from collections import Counter, defaultdict

import cv2
import torch
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from perspective import rectify_plate

# Minimum mean per-character confidence for a reading to be trusted enough to vote.
MIN_OCR_CONFIDENCE = 0.4
# How many of the most recent readings per tracked plate to keep for the majority vote.
HISTORY_SIZE = 15

class LPR:
    """License Plate Recognition using Ultralytics YOLO and Fast Plate OCR.

    This class handles license plate detection using a YOLO model and text extraction
    using Fast Plate OCR. It supports both image and video streams for a real-time inference.

    Single-frame OCR reads are noisy (motion blur, lighting, a slightly-off crop can flip a
    character), so each detected plate is tracked across frames and its displayed text is the
    majority vote over its recent readings rather than whatever the current frame alone says.
    This trades a bit of latency (a new plate takes a few frames to "settle") for stability.

    Attributes:
        model (YOLO): The YOLO model for license plate detection and tracking.
        reader (LicensePlateRecognizer): The OCR reader instance for text recognition.
        device (torch.device): Computation device (CPU or CUDA)
        imgsz (int): Resolution YOLO resizes frames to internally before detecting.
        plate_history (dict[int, Counter]): Recent OCR readings per tracked plate ID.
    """

    def __init__(self, model_path: str = "yolo26n.pt", imgsz: int = 960, reader: LicensePlateRecognizer = None):
        """Initializes the LPR system

        imgsz: resolution YOLO internally resizes frames to before detecting. The ultralytics
            default (640) is why distant/small plates were going undetected entirely — measured
            ~70% more detections at 1280 vs 640 on this model, at equal confidence. 960 is a
            middle ground between that recall and inference cost; raise it further (matching
            infer_video's max_width, e.g. 1280) if distant plates still aren't showing up, at
            the cost of speed.
        reader: an existing LicensePlateRecognizer to reuse instead of loading a new one. For
            a single LPR instance this doesn't matter, but multiple LPR instances that want to
            pool OCR calls together (e.g. one instance per concurrent stream, batching crops
            from all of them into shared reader.run() calls) need to share one reader rather
            than each loading and holding its own copy of the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.reader = reader or LicensePlateRecognizer("cct-s-v2-global-model", device=self.device.type)
        self.plate_history: dict[int, Counter] = defaultdict(Counter)

    def detect_plates(self, im0: np.ndarray):
        """
           Detects and tracks license plates in an image, assigning each a persistent ID
           across frames so OCR readings can be smoothed over time.
           im0: image
           np.ndarray: N-dimensional array
        """
        results = self.model.track(im0, persist=True, verbose=False, imgsz=self.imgsz, device=self.device.type)
        if not results or results[0].boxes is None:
            return [], []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        track_ids = track_ids.int().cpu().numpy() if track_ids is not None else [None] * len(boxes)
        return boxes, track_ids

    @staticmethod
    def crop_plate(im0: np.ndarray, bbox: np.ndarray):
        """Crops the plate region out of a frame and converts it to the RGB fast-plate-ocr
        expects. Returns None if the box didn't produce a usable crop (e.g. clipped to zero
        size by a resize). Split out from extract_text so callers that want to batch OCR
        calls across multiple crops (rather than one reader.run() per crop) can do the
        cropping here and the batched reader.run() call themselves."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = im0[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    def _read(self, rgb: np.ndarray):
        """Runs OCR on a single crop. Returns (text, confidence)."""
        predictions = self.reader.run(rgb, return_confidence=True)
        if not predictions:
            return "", 0.0
        pred = predictions[0]
        confidence = float(pred.char_probs.mean()) if pred.char_probs is not None else 0.0
        return pred.plate.strip(), confidence

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """Performs OCR on the cropped license plate region, trying perspective
        correction and keeping whichever reading is actually more confident.

        Why not just always use the corrected crop: measured on real data, classical
        contour-based correction only finds a confident quadrilateral ~8% of the time,
        and when it does, it improved OCR confidence in fewer cases than it hurt it
        (7 improved vs. 14 worsened out of 39, in one real test run) - a geometrically
        plausible quadrilateral isn't always the *right* one, and warping to a wrong
        one can distort the text worse than leaving it alone. Rather than trust the
        geometry blindly, this runs OCR on both the original and corrected crop and
        keeps whichever one the OCR model itself is more confident about - strictly
        improve-or-neutral instead of a coin flip, at the cost of a second OCR call
        (cheap - OCR is ~7-10ms/crop, far less than detection).

        Returns (text, confidence), where confidence is the mean per-character probability
        (0.0 if the model doesn't report one, e.g. no plate was decoded).
        """
        rgb = self.crop_plate(im0, bbox)
        if rgb is None:
            return "", 0.0

        text, confidence = self._read(rgb)

        rectified = rectify_plate(rgb)
        if rectified is not rgb:  # rectify_plate returns the same object when it couldn't correct
            rect_text, rect_confidence = self._read(rectified)
            if rect_confidence > confidence:
                text, confidence = rect_text, rect_confidence

        return text, confidence

    def stabilize_text(self, track_id, text: str, confidence: float):
        """Smooths a plate's OCR reading over time using a per-track majority vote.

        Low-confidence or empty reads don't get to vote (they're usually the noisy ones), but
        they still fall back to whatever the best vote so far is, instead of showing nothing.
        Without a track_id (tracking unavailable/lost) it just returns the raw per-frame read.
        """
        if track_id is None:
            return text

        history = self.plate_history[track_id]
        if text and confidence >= MIN_OCR_CONFIDENCE:
            history[text] += 1
            if sum(history.values()) > HISTORY_SIZE:
                # Halve all votes (dropping any that round down to zero) so the vote can still
                # adapt if the reading genuinely changes, instead of being stuck forever.
                for key in list(history):
                    halved = history[key] // 2
                    if halved:
                        history[key] = halved
                    else:
                        del history[key]

        return history.most_common(1)[0][0] if history else text

    def infer_video(
        self,
        source: str = 0,
        output_path: str = None,
        display: bool = True,
        max_width: int = 1280,
        detect_every: int = 2,
    ):
        """Performs real-time LPR on a video

        max_width: frames wider than this are downscaled (aspect-ratio preserved) before
            detection/display/writing. cv2.imshow and VideoWriter get disproportionately slow
            at 4K+ resolutions, which is usually the real bottleneck, not model inference.
            Pass None to keep the source resolution untouched.
        detect_every: run tracking+OCR on every Nth frame; frames in between reuse the last
            known box positions and OCR vote instead of re-running the model. Tracking+OCR
            (not display/encoding) is the actual per-frame bottleneck, so this is what lets
            playback keep up with the source's frame rate. Boxes go slightly stale between
            detections, so raise this back toward 1 if fast-moving plates start drifting out
            of their boxes.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        if max_width and width > max_width:
            height = int(height * (max_width / width))
            width = max_width

        writer = None
        if output_path:
            # H.264 in an mp4 container. "mp4v" (MPEG-4 Part 2) writes fine but macOS's
            # QuickTime/Preview/AVFoundation and Safari won't play it back — "avc1" is what
            # they actually expect.
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Starting LPR video inference... Pres 'q' to quit.")

        last_boxes, last_track_ids = [], []
        frame_idx = 0

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            if max_width and im0.shape[1] > max_width:
                im0 = cv2.resize(im0, (width, height))

            run_detection = frame_idx % detect_every == 0
            frame_idx += 1

            if run_detection:
                boxes, track_ids = self.detect_plates(im0)
                last_boxes, last_track_ids = boxes, track_ids
            else:
                boxes, track_ids = last_boxes, last_track_ids

            ann = Annotator(im0, line_width=4)
            for bbox, track_id in zip(boxes, track_ids):
                if run_detection:
                    text, confidence = self.extract_text(im0, bbox)
                else:
                    text, confidence = "", 0.0  # skip OCR this frame, just look up the current vote
                text = self.stabilize_text(track_id, text, confidence)
                ann.box_label(bbox, label=text, color=colors(17, True))

            if display:
                cv2.imshow("LPR (Press 'q' to exit)", im0)
            if writer:
                writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.relase()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lpr = LPR(model_path="lpr_best.pt")
    lpr.infer_video(source="acar2.mp4", output_path="lpr_output.mp4", display=True)
