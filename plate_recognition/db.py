"""Thin asyncpg wrapper for the LPR backend. See ROADMAP.md "Phase 2 —
Backend" for why this is raw asyncpg rather than an ORM (latency-
sensitivity is part of what's being measured), and why dedup is enforced
by the UNIQUE(stream_id, track_id) constraint + upsert rather than
application-code discipline.
"""

import os

import asyncpg

DATABASE_URL = os.environ.get("LPR_DATABASE_URL", "postgresql:///lpr_dev")

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Returns the module-level connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


UPSERT_VEHICLE_EVENT = """
    INSERT INTO vehicle_events (stream_id, track_id, plate_text, first_frame, last_frame, frame_count)
    VALUES ($1, $2, $3, $4, $4, 1)
    ON CONFLICT (stream_id, track_id) DO UPDATE SET
        plate_text  = EXCLUDED.plate_text,
        last_frame  = EXCLUDED.last_frame,
        frame_count = vehicle_events.frame_count + 1,
        updated_at  = now()
"""


async def upsert_vehicle_event(pool: asyncpg.Pool, stream_id: str, track_id: int, plate_text: str | None, frame_num: int) -> None:
    """Records one frame's sighting of a track. First call for a
    (stream_id, track_id) pair inserts the row (first_frame = last_frame =
    frame_num); every call after that updates the *same* row in place -
    that's the dedup: a track can never produce a second row, by
    construction of the UNIQUE constraint this upsert targets, not because
    the caller remembered to check first.
    """
    async with pool.acquire() as conn:
        await conn.execute(UPSERT_VEHICLE_EVENT, stream_id, track_id, plate_text, frame_num)


START_RUN = """
    INSERT INTO processing_runs (label) VALUES ($1) RETURNING id
"""

FINISH_RUN = """
    UPDATE processing_runs SET
        ended_at             = now(),
        total_frames         = $2,
        total_raw_detections = $3,
        total_vehicle_events = $4,
        aggregate_fps        = $5,
        p95_latency_ms       = $6
    WHERE id = $1
"""


async def start_run(pool: asyncpg.Pool, label: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(START_RUN, label)


async def finish_run(
    pool: asyncpg.Pool,
    run_id: int,
    total_frames: int,
    total_raw_detections: int,
    total_vehicle_events: int,
    aggregate_fps: float,
    p95_latency_ms: float,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            FINISH_RUN, run_id, total_frames, total_raw_detections,
            total_vehicle_events, aggregate_fps, p95_latency_ms,
        )
