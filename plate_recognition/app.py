"""LPR backend. Minimal for now - proves FastAPI + asyncpg + Postgres work
together end-to-end before the ingest/queue/worker pipeline gets built on
top. See ROADMAP.md "Phase 2 — Backend".
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await db.get_pool()
    app.state.pool = pool
    yield
    await db.close_pool()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    async with app.state.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    return {"status": "ok"}
