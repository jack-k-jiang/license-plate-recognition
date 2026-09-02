-- LPR backend schema. See ROADMAP.md "Phase 2 — Backend" for the design
-- rationale (why 2 tables, why UNIQUE(stream_id, track_id) is what makes
-- dedup real rather than just application-code discipline).

CREATE TABLE IF NOT EXISTS vehicle_events (
    id              BIGSERIAL PRIMARY KEY,
    stream_id       TEXT NOT NULL,
    track_id        INT NOT NULL,
    plate_text      TEXT,
    first_frame     INT NOT NULL,
    last_frame      INT NOT NULL,
    frame_count     INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stream_id, track_id)
);

CREATE TABLE IF NOT EXISTS processing_runs (
    id                      BIGSERIAL PRIMARY KEY,
    label                   TEXT,
    started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at                TIMESTAMPTZ,
    total_frames            INT NOT NULL DEFAULT 0,
    total_raw_detections    INT NOT NULL DEFAULT 0,
    total_vehicle_events    INT NOT NULL DEFAULT 0,
    aggregate_fps           DOUBLE PRECISION,
    p95_latency_ms          DOUBLE PRECISION
);
