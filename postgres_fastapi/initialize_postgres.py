import configparser
import psycopg

config = configparser.ConfigParser()
config.read("config.ini")

pg = config["postgres"]

conn = psycopg.connect(
    host="localhost",
    port=int(pg["port"]),
    dbname=pg["database"],
    user=pg["username"],
    password=pg["password"],
)

with conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gaze_img_table (
                id BIGSERIAL PRIMARY KEY,
                timestamp DOUBLE PRECISION NOT NULL,
                depth DOUBLE PRECISION NOT NULL,
                img_path TEXT NOT NULL
            );
        """)

print("Database initialized.")