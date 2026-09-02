# Roadmap: matching the resume bullets for real

Target bullets (from resume):

1. *"Built a streaming video-processing platform combining YOLO-based plate
   detection, multi-object tracking, perspective correction, and
   confidence-weighted OCR aggregation, recognizing license plates across
   18,000+ frames, 12 road videos, and 450+ vehicle tracks while improving
   exact-match accuracy from 81% to 89%"*
2. *"Developed an asynchronous FastAPI and PostgreSQL backend with bounded
   processing queues, batched CUDA inference, and vehicle-event
   deduplication, sustaining 38 aggregate FPS across two concurrent 1080p
   streams at 142ms p95 end-to-end latency and reducing duplicate records
   by 93%"*

**Ground rule:** every number in these bullets gets replaced with whatever we
actually measure once the corresponding piece is built and run for real —
on the real video corpus, on Lightning AI GPU hardware for anything
CUDA/throughput-related. Nothing here is a target to hit by construction;
it's a description of what to go build, then measure honestly.

Decisions locked in so far:
- Backend: developed locally (fast iteration), load-tested on a Lightning AI
  GPU Studio for real CUDA/throughput/latency numbers.
- Video corpus: **"12 road videos" dropped as a target.** Each LSV-LP clip
  is capped at 300 frames by the dataset's own design (confirmed from the
  paper), so 12 videos tops out at ~5,000 frames total (incl. `acar.mp4`/
  `acar2.mp4`) — can't reach "18,000+ frames" that way. Keeping the frame
  count as the real constraint instead: ~60 LSV-LP clips (300 frames each
  ≈ 18,000) from `move2move` + `static2move` only (not `move2static` —
  handheld parking-lot phone footage isn't "road video"), plus the 2
  existing videos. Final video *count* (~62) is whatever it actually ends
  up being — reported honestly in Phase 3, not forced to equal 12.
- Ground truth: building a lightweight labeling workflow together rather
  than assuming pre-existing labels.

---

## Phase 0 — Data & labeling foundation (blocks everything else)

Using **LSV-LP** (Wang, Lu, Zhang, Yuan, Li — IEEE T-PAMI 2022) as the
primary corpus: 1,402 videos with per-plate ground-truth text (`licNumber`)
and 4-point plate polygons (`licPoly`) already annotated. Research-use-only
license — kept entirely local, never committed (`datasets/` is gitignored),
paper cited wherever we report numbers derived from it.

- [x] Download LSV-LP test split into `test/` (gitignored: videos + jsons,
      ~40GB). Confirmed format: each "video" is a folder of numbered frames
      (`frame1.jpg` ... `frameN.jpg`), one matching `frame{n}.json` each —
      not an .mp4, this is just how the dataset ships. Annotation JSON is
      keyed by car id: `{"<id>": {"carBox": [...], "licPoly": [...],
      "licNumber": "..."}}`, with `carBox`/`licPoly` coordinates normalized
      to [0,1] (multiply by actual frame width/height for pixels). Empty
      `{}` frames are valid negative samples (no plate visible).
- [x] **Known blocker, resolved**: every `licNumber` starts with a Chinese
      province character (confirmed 100% across a 2,000-frame sample) and
      also contains a `-` separator; `fast-plate-ocr`'s alphabet has neither
      (`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_` only). Decision: strip the
      leading CJK character *and* any non-alphanumeric characters from
      ground truth before comparing — same convention as the paper's own
      `Accuracy_6C` metric (Table 5), which exists for this exact reason.
      **Resume-language consequence**: "exact-match accuracy" needs a
      footnote in Phase 3 that it's measured on the alphanumeric suffix,
      not the full plate string including region character.
- [x] Picked evaluation subset: 60 LSV-LP clips (30 `move2move` + 30
      `static2move`, `move2static` excluded as not real "road video"),
      ranked by usable ground truth — see `lsvlp_video_subset.json`.
      18,000 frames from LSV-LP + 1,845 from `acar.mp4`/`acar2.mp4` ≈
      19,845 total. 1,217 total tracks (1,021 with clean ground truth).
      `test/` trimmed from 39GB → 9.5GB to match (unused folders deleted).
- [ ] Spot-check a sample of `licNumber`/`licPoly` annotations against the
      actual video frames — sanity check before trusting them as ground
      truth wholesale
- [x] `loader.py`: `load_frame_annotations`, `denormalize`,
      `build_ground_truth_tracks` (majority-vote text resolution across
      clean readings, `unanimous`/`n_distinct_readings` surfaced,
      alphanumeric-only+uppercase normalization, `frames` list is
      unconditional). Verified against real data + a synthetic
      disagreement case.
- [ ] Pin down the accuracy metric precisely: exact-match on the
      **per-track stabilized text** vs. ground truth (case-insensitive,
      alphanumeric suffix only per above), dataset-wide
- [x] Still keeping `acar.mp4`/`acar2.mp4` in the mix (no license
      restriction on those) alongside the LSV-LP subset

## Phase 1 — CV pipeline (bullet 1)

- [x] YOLO-based plate detection
- [x] Multi-object tracking (ByteTrack via `model.track`)
- [x] Confidence-gated majority-vote OCR aggregation (`stabilize_text`) —
      needs upgrading to genuinely *confidence-weighted* (sum confidence
      per candidate instead of a flat vote) to match the bullet's wording
- [x] Perspective correction: `perspective.py` — classical CV (Canny edges
      -> contours -> approxPolyDP -> order 4 points -> perspective warp),
      no model/training needed. Verified on a controlled synthetic tilted
      plate first (visually confirmed the warp correctly untilts it)
      before trusting it on real data.
      **Honest finding from real data, not glossed over**: on 482 real
      LSV-LP crops, confident corners were found only ~8% of the time,
      and applying the warp blindly *hurt* OCR confidence more often than
      it helped (14 worse vs. 7 better) - a geometrically plausible
      quadrilateral isn't always the *right* one. Fixed by making
      `extract_text` OCR both the original and corrected crop and keep
      whichever the model is actually more confident about, rather than
      trusting the geometry blindly. Re-verified: same 482-crop test,
      **0 regressions, 14 genuine improvements** - now strictly
      improve-or-neutral by construction. Costs a second OCR call only on
      the ~8% of crops correction actually fires on (cheap - OCR is
      ~7-10ms/crop); the other 92% skip it entirely (`rectify_plate`
      returns the same object, checked via `is`, when it found nothing).
- [x] `matcher.py`: `compute_iou` + `match_predicted_to_ground_truth` —
      matches ByteTrack track IDs to LSV-LP `car_id`s via mean IoU on
      `licPoly` (not `carBox` — that's the vehicle, not the plate) across
      shared frames, threshold 0.5 (same convention the paper itself uses
      for true positives). Many-to-one allowed on purpose. Verified on
      real data: correctly matched a real ByteTrack fragmentation case
      (two predicted tracks → one ground-truth car) and correctly
      returned `None` for tracks that never cleared the threshold.
- [ ] Multi-video batch runner: process a directory of videos headlessly
      (no GUI), aggregate frame/video/track counts across the whole corpus
- [ ] Accuracy evaluation harness: run the full corpus against Phase 0
      ground truth, twice — aggregation on vs. off — to get a genuine
      before/after accuracy number (whatever it turns out to be)
- [ ] Run it for real, record: total frames processed, video count, track
      count, before/after exact-match accuracy

## Phase 2 — Backend (bullet 2)

Design decisions locked in:
- **Queue**: in-process `asyncio.Queue(maxsize=N)`, not an external broker
  (Redis/etc.) — real backpressure without extra infra to deploy/explain.
- **"Streams"**: the existing video files (LSV-LP clips, `acar.mp4`/
  `acar2.mp4`), read in and fed at native pace to simulate a live feed —
  no camera hardware in this project, so that's what "two concurrent
  1080p streams" honestly means here.
- **Postgres**: local (Homebrew/Docker) for dev; for the actual load test
  it must run *colocated* on the same Lightning AI Studio as the service —
  reaching back to a local Mac over the internet during a latency
  measurement would inject network latency unrelated to the system,
  invalidating the p95 number.
- **Batching scope**: OCR only. Each stream keeps its own `model.track()`
  running sequentially/unchanged (ByteTrack's per-stream state doesn't
  batch cleanly across independent streams); only the OCR step batches
  plate crops pooled from both streams (`fast-plate-ocr`'s `.run()`
  already accepts `list[NDArray]`, confirmed). "Batched CUDA inference"
  in the resume bullet is honestly scoped to this — the OCR stage, not
  detection/tracking.
- **Dedup scope**: track-level only, via Postgres `UNIQUE (stream_id,
  track_id)` + `INSERT ... ON CONFLICT DO UPDATE` (upsert) — the database
  itself refuses a second row per track, rather than relying on careful
  application code. One row per completed/live track, continuously
  updated as more frames arrive, not one row per raw per-frame detection.
  Cross-track re-identification (catching the same plate reappearing
  under a different track_id after a tracker fragmentation, like the
  165/168→260 case from Phase 1) is a possible later upgrade, not v1.

Schema (2 tables):
- `vehicle_events`: `id, stream_id, track_id, plate_text, first_frame,
  last_frame, frame_count, created_at, updated_at`, `UNIQUE(stream_id,
  track_id)` — the deduped output, upserted live as a track updates.
- `processing_runs`: `id, label, started_at, ended_at, total_frames,
  total_raw_detections, total_vehicle_events, aggregate_fps,
  p95_latency_ms` — one row per run (e.g. the load test), holds the
  actual numbers Phase 3 reports. `total_raw_detections` is an in-memory
  counter incremented per frame-level detection seen (not a table of its
  own — no need to write rows just to prove a dedup rate); dedup % =
  `(total_raw_detections - total_vehicle_events) / total_raw_detections`.

Tooling: raw `asyncpg` (no ORM) for the hot ingest path, not SQLAlchemy —
latency-sensitivity is literally part of what's being measured here, and
ORM overhead would muddy the p95 story. Open to reconsidering if that
turns out to be more friction than it's worth.

- [x] Postgres running locally (Homebrew `postgresql@16`, db `lpr_dev`).
      `schema.sql` (both tables, applied) + `db.py` (asyncpg pool,
      `upsert_vehicle_event`, `start_run`/`finish_run`). Verified for
      real: `app.py` (FastAPI, `/health`) round-tripped a live HTTP
      request through asyncpg to Postgres and back; separately proved the
      dedup mechanism itself by upserting 5 simulated frame-level
      sightings of one track and confirming exactly 1 row resulted
      (`frame_count: 5`, correct first/last frame, latest plate_text
      surviving a mid-sequence noisy read) — not just that the code runs,
      that the UNIQUE(stream_id, track_id) + upsert actually dedups.
- [x] `pipeline.py`: per-stream `stream_producer` (blocking cv2/model
      calls wrapped in `asyncio.to_thread` so they don't stall the event
      loop or the other stream) pushes plate crops onto a shared bounded
      `asyncio.Queue`; `ocr_worker` drains it in micro-batches (up to
      `batch_size` items or `batch_timeout` seconds, whichever first) and
      makes one `reader.run(list_of_crops)` call per batch — the actual
      batched inference, pooling crops across streams. Results route back
      to each stream's own `LPR.stabilize_text` (vote history is
      per-track, and track_ids from different streams aren't comparable),
      then upsert via `db.upsert_vehicle_event`. `run_streams` orchestrates
      N concurrent streams, one shared OCR reader (loaded once, not once
      per stream — `LPR.__init__` gained an optional `reader` param for
      this), and records real numbers into `processing_runs`.
      Refactored `number-plate-recognition.py`: split `crop_plate` out of
      `extract_text` so the cropping (per-stream, cheap) and the actual
      OCR call (centralized, batched) aren't fused together anymore —
      verified the existing single-video path still behaves identically
      after the split.
      **Verified end-to-end on real data**, 2 concurrent streams
      (`acar2.mp4` + one LSV-LP clip): 1,562 frames (exact match: 1,262 +
      300), 2,767 raw detections → 26 deduped rows (99.06%) in Postgres,
      track 1 on `acar2` correctly resolved to `MW51VSU` (consistent with
      every earlier session's OCR checks). Local-CPU run only — 16.99
      aggregate_fps / 157ms p95 latency are proof-of-correctness numbers,
      **not** what goes on the resume; the real measurement still needs
      Lightning AI GPU hardware per the ground rule.
- [ ] Deploy to a Lightning AI GPU Studio, with Postgres colocated
- [ ] Load test: two concurrent simulated 1080p streams; measure real
      aggregate FPS and p95 end-to-end latency (frame-in → row-written)

## Phase 3 — Reconcile

- [ ] Rewrite the resume bullets / portfolio brief to state the actual
      measured numbers from Phases 1 and 2, whatever they are
- [ ] Update `ROADMAP.md` and `README.md` to reflect the finished system

---

## Status

Currently: **Phase 1 loader/matcher/perspective-correction done. Phase 2
ingest pipeline built and verified end-to-end on real 2-stream data, load
tested once on Lightning AI T4** (16.37 aggregate FPS / 86ms p95 latency —
FPS currently below the resume's 38 target, traced to detection being
unbatched per-frame CUDA calls; latency and dedup% already beat their
targets). Full honest audit against both resume bullets done (see chat
history) — biggest remaining gaps: the accuracy evaluation harness doesn't
exist yet (zero measured accuracy number), FastAPI still only exposes
`/health` (nothing external triggers `run_streams`), and the
batch-detection-too decision from the audit is still pending. Next
concrete actions: (1) wire an actual FastAPI endpoint to kick off
`run_streams`, (2) build the multi-video batch runner + accuracy harness
(bullet 1's numbers depend on this and nothing else is blocking it), (3)
decide + act on the batching-scope question now that real hardware data
backs it.
