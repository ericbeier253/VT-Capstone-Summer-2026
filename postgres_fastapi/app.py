from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import psycopg
import configparser
import os
import pathlib

os.chdir(pathlib.Path(__file__).resolve().parent)

# ----------------------------
# Load config.ini
# ----------------------------
config = configparser.ConfigParser()
config.read("config.ini")

pg = config["postgres"]

DB_CONFIG = {
    "host": "localhost",
    "port": int(pg["port"]),
    "dbname": pg["database"],
    "user": pg["username"],
    "password": pg["password"],
}

# ----------------------------
# SQL schema
# ----------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS depth_images (
    id BIGSERIAL PRIMARY KEY,
    timestamp DOUBLE PRECISION NOT NULL,
    depth DOUBLE PRECISION NOT NULL,
    img_path TEXT NOT NULL
);
"""

# ----------------------------
# Lifespan (replaces @on_event)
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ startup
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()

    print("Database initialized")

    yield

    # shutdown
    print("Server shutting down")


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(lifespan=lifespan)


# ----------------------------
# Request model
# ----------------------------
class DepthRecord(BaseModel):
    timestamp: float
    depth: float
    img_path: str


# ----------------------------
# INSERT
# ----------------------------
@app.post("/insert")
def insert_record(record: DepthRecord):
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO depth_images (timestamp, depth, img_path)
                VALUES (%s, %s, %s)
                """,
                (record.timestamp, record.depth, record.img_path),
            )
        conn.commit()

    return {"status": "inserted"}


# ----------------------------
# READ
# ----------------------------
@app.get("/records")
def get_records(limit: int = 100):
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, depth, img_path
                FROM depth_images
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )

            rows = cur.fetchall()

    return [
        {"timestamp": r[0], "depth": r[1], "img_path": r[2]}
        for r in rows
    ]