"""Async ingest pipeline: per-stream detection/tracking producers feeding a
shared bounded queue, drained by a single OCR worker that batches crops
across streams. See ROADMAP.md "Phase 2 — Backend" for the design (why
detection/tracking stays per-stream and unbatched while OCR is centralized
and batched, why the queue is asyncio.Queue rather than an external broker).
"""

import asyncio
import runpy
import statistics
import time
from dataclasses import dataclass, field

import cv2

import db

# number-plate-recognition.py has a hyphen in its filename, so it can't be
# imported with a normal `import` statement - runpy loads it as a module dict.
_module = runpy.run_path("number-plate-recognition.py", run_name="not_main")
LPR = _module["LPR"]
LicensePlateRecognizer = _module["LicensePlateRecognizer"]


@dataclass
class QueueItem:
    stream_id: str
    track_id: int
    frame_num: int
    crop: object  # RGB np.ndarray, ready for reader.run()
    enqueued_at: float


@dataclass
class RunStats:
    total_frames: int = 0
    total_raw_detections: int = 0
    latencies: list = field(default_factory=list)  # seconds, one per upsert


async def stream_producer(
    lpr: "LPR",
    source,
    stream_id: str,
    queue: asyncio.Queue,
    stats: RunStats,
    max_width: int = 1280,
    target_size: tuple = None,
):
    """Reads frames from `source`, runs per-stream detection+tracking
    (sequential - ByteTrack's persist=True state lives on this stream's own
    `lpr` instance and can't be interleaved with another stream's frames),
    and pushes each detected plate's crop onto the shared bounded queue for
    OCR. `source` is whatever cv2.VideoCapture(*source) accepts, e.g.
    ("acar2.mp4",) or (".../frame%d.jpg", cv2.CAP_IMAGES).

    max_width only ever *downscales* oversized frames (a display-oriented
    default inherited from infer_video) - it never upscales, so it cannot
    guarantee any particular resolution actually reaches the model. For a
    benchmark that specifically claims "1080p streams", pass
    target_size=(1920, 1080) instead: every frame is force-resized to
    exactly that size (both directions), so what the model sees matches
    what gets reported, regardless of the source file's native resolution.

    Every cv2/model call here is a blocking, synchronous call - wrapped in
    asyncio.to_thread so this coroutine yields the event loop while waiting
    on them, instead of stalling the other stream's producer and the OCR
    worker for the whole duration of every frame read/detect.
    """
    cap = await asyncio.to_thread(cv2.VideoCapture, *source)
    frame_num = 0

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                break
            frame_num += 1
            stats.total_frames += 1

            if target_size:
                frame = await asyncio.to_thread(cv2.resize, frame, target_size)
            elif max_width and frame.shape[1] > max_width:
                new_h = int(frame.shape[0] * max_width / frame.shape[1])
                frame = await asyncio.to_thread(cv2.resize, frame, (max_width, new_h))

            boxes, track_ids = await asyncio.to_thread(lpr.detect_plates, frame)
            for bbox, track_id in zip(boxes, track_ids):
                if track_id is None:
                    continue
                crop = lpr.crop_plate(frame, bbox)  # cheap, fine inline (no thread needed)
                if crop is None:
                    continue
                item = QueueItem(stream_id, int(track_id), frame_num, crop, time.time())
                await queue.put(item)  # blocks here if the queue is full - the real backpressure
    finally:
        cap.release()


async def ocr_worker(
    lpr_by_stream: dict,
    reader: "LicensePlateRecognizer",
    queue: asyncio.Queue,
    pool,
    stats: RunStats,
    batch_size: int = 8,
    batch_timeout: float = 0.05,
):
    """Pulls queued crops and forms a micro-batch - up to batch_size items,
    or whatever's arrived within batch_timeout seconds, whichever comes
    first (so a quiet queue doesn't stall waiting to fill a full batch) -
    then makes ONE reader.run() call across the whole batch. That single
    call is the actual "batched CUDA inference": crops pooled from every
    stream currently producing, not one call per stream.

    Each result is routed back to its own stream's LPR instance for
    stabilize_text (the per-track majority vote lives there, keyed by
    track_id, and track_ids from different streams aren't comparable - a
    stream's own LPR instance is the only thing that knows which vote
    history a given track_id's result belongs to), then upserted.
    """
    while True:
        batch = [await queue.get()]
        deadline = time.monotonic() + batch_timeout
        while len(batch) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break

        crops = [item.crop for item in batch]
        predictions = await asyncio.to_thread(reader.run, crops, return_confidence=True)

        for item, pred in zip(batch, predictions):
            confidence = float(pred.char_probs.mean()) if pred.char_probs is not None else 0.0
            text = pred.plate.strip() if pred.plate else ""

            lpr = lpr_by_stream[item.stream_id]
            stable_text = lpr.stabilize_text(item.track_id, text, confidence)

            await db.upsert_vehicle_event(pool, item.stream_id, item.track_id, stable_text, item.frame_num)

            stats.total_raw_detections += 1
            stats.latencies.append(time.time() - item.enqueued_at)

        for _ in batch:
            queue.task_done()


async def run_streams(
    sources: dict,
    label: str,
    model_path: str = "lpr_best.pt",
    queue_maxsize: int = 64,
    target_size: tuple = None,
):
    """Runs every (stream_id -> source) pair in `sources` concurrently
    against one shared bounded queue and OCR worker, records a
    processing_runs row with real aggregate FPS / p95 latency, and returns
    the same numbers for immediate inspection.

    target_size: forwarded to every stream_producer - pass (1920, 1080) for
    a run that needs to honestly back a "1080p streams" claim (see
    stream_producer's docstring for why max_width alone can't guarantee
    this).
    """
    pool = await db.get_pool()
    run_id = await db.start_run(pool, label)

    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
    stats = RunStats()

    # One LPR instance per stream (each needs its own ByteTrack + vote-history
    # state), but only the first one's OCR reader is kept - the rest reuse it,
    # so the model gets loaded once, not once per stream.
    lpr_by_stream = {}
    shared_reader = None
    for stream_id in sources:
        lpr = LPR(model_path=model_path, reader=shared_reader)
        shared_reader = lpr.reader
        lpr_by_stream[stream_id] = lpr

    worker_task = asyncio.create_task(
        ocr_worker(lpr_by_stream, shared_reader, queue, pool, stats)
    )

    t0 = time.time()
    producer_tasks = [
        asyncio.create_task(
            stream_producer(lpr_by_stream[sid], src, sid, queue, stats, target_size=target_size)
        )
        for sid, src in sources.items()
    ]
    await asyncio.gather(*producer_tasks)
    await queue.join()  # wait for the worker to drain everything that's been enqueued
    worker_task.cancel()
    duration = time.time() - t0

    async with pool.acquire() as conn:
        total_events = await conn.fetchval(
            "SELECT count(*) FROM vehicle_events WHERE stream_id = ANY($1)", list(sources.keys())
        )

    aggregate_fps = stats.total_frames / duration if duration > 0 else 0.0
    if len(stats.latencies) >= 2:
        p95_latency_ms = statistics.quantiles(stats.latencies, n=100)[94] * 1000
    elif stats.latencies:
        p95_latency_ms = stats.latencies[0] * 1000
    else:
        p95_latency_ms = 0.0

    await db.finish_run(
        pool, run_id, stats.total_frames, stats.total_raw_detections,
        total_events, aggregate_fps, p95_latency_ms,
    )

    return {
        "run_id": run_id,
        "duration_s": duration,
        "total_frames": stats.total_frames,
        "total_raw_detections": stats.total_raw_detections,
        "total_vehicle_events": total_events,
        "dedup_pct": (
            (stats.total_raw_detections - total_events) / stats.total_raw_detections * 100
            if stats.total_raw_detections else 0.0
        ),
        "aggregate_fps": aggregate_fps,
        "p95_latency_ms": p95_latency_ms,
    }
