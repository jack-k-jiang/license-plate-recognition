from collections import Counter, defaultdict

import cv2
import torch
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

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
        plate_history (dict[int, Counter]): Recent OCR readings per tracked plate ID.
    """

    def __init__(self, model_path: str = "yolo26n.pt"):
        """Initializes the LPR system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.reader = LicensePlateRecognizer("cct-s-v2-global-model", device=self.device.type)
        self.plate_history: dict[int, Counter] = defaultdict(Counter)

    def detect_plates(self, im0: np.ndarray):
        """
           Detects and tracks license plates in an image, assigning each a persistent ID
           across frames so OCR readings can be smoothed over time.
           im0: image
           np.ndarray: N-dimensional array
        """
        results = self.model.track(im0, persist=True, verbose=False)
        if not results or results[0].boxes is None:
            return [], []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        track_ids = track_ids.int().cpu().numpy() if track_ids is not None else [None] * len(boxes)
        return boxes, track_ids

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        """Performs OCR on the cropped license plate region.

        Returns (text, confidence), where confidence is the mean per-character probability
        (0.0 if the model doesn't report one, e.g. no plate was decoded).
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = im0[y1:y2, x1:x2]
        if roi.size == 0:
            return "", 0.0

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        predictions = self.reader.run(rgb, return_confidence=True)
        if not predictions:
            return "", 0.0

        pred = predictions[0]
        confidence = float(pred.char_probs.mean()) if pred.char_probs is not None else 0.0
        return pred.plate.strip(), confidence

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

    def infer_video(self, source: str = 0, output_path: str = None, display: bool=True, max_width: int = 1280):
        """Performs real-time LPR on a video

        max_width: frames wider than this are downscaled (aspect-ratio preserved) before
            detection/display/writing. cv2.imshow and VideoWriter get disproportionately slow
            at 4K+ resolutions, which is usually the real bottleneck, not model inference.
            Pass None to keep the source resolution untouched.
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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Starting LPR video inference... Pres 'q' to quit.")

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            if max_width and im0.shape[1] > max_width:
                im0 = cv2.resize(im0, (width, height))

            boxes, track_ids = self.detect_plates(im0)
            ann = Annotator(im0, line_width=4)
            for bbox, track_id in zip(boxes, track_ids):
                text, confidence = self.extract_text(im0, bbox)
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
