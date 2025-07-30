
import sqlite3
import numpy as np
import io
from video_ad_detector import config

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

def init_db():
    with sqlite3.connect(config.DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            feature_vector array
        )
        """)
        conn.commit()

def save_material_features(filename: str, features: np.ndarray):
    with sqlite3.connect(config.DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO materials (filename, feature_vector) VALUES (?, ?)", (filename, features))
        conn.commit()

def get_all_material_features():
    with sqlite3.connect(config.DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, feature_vector FROM materials")
        return cursor.fetchall()
