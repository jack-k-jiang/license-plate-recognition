from matcher import compute_iou

class IouTracker:
    """Per-stream, stateful. Call update() once per frame with that frame's
    detected boxes; returns a track_id per box, reusing an existing track_id when a box is the same object as last frame (by IOU)
    , assigning a new one otherwise."""

    def __init__(self, iou_threshold: float =0.3, max_missed: int=5):
        ...
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {} # track_id -> {"box": ..., "missed": 0}
        self.next_id = 1

    # boxes = [[x1, y1, x2, y2], ...] - each entry is one detected plate
    def update(self, boxes: list) -> list[int]:
        if not boxes:
            for ids in self.tracks:
                self.tracks[ids]["missed"] = self.tracks[ids]["missed"]+1
                if (self.tracks[ids]["missed"] > self.max_missed):
                    del self.tracks[ids]

        for box in boxes:
            if not self.tracks:
               self.tracks[self.next_id] = {"box": box, "missed": 0}

            else:
                for ids in self.tracks.keys():
                    box1 = ((box[0], box[1]), (box[2], box[3]))
                    box2 = self.tracks[ids]["box"]

                    iou = compute_iou(box1, box2)
                    if iou < self.iou_threshold:
                        self.next_id+=1
                        self.tracks[self.next_id] = {"box": box, "missed": 0}