import sqlite3
import numpy as np
import os
from . import config

DATABASE_FILE = os.path.join(config.DATABASE_DIR, "metadata.db")

def init_db():
    """
    Initializes the SQLite database and creates tables if they don't exist.
    """
    os.makedirs(config.DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # Main table for materials
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE
        )
    """)
    # Table for individual frame features, linked to a material
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            material_id INTEGER,
            timestamp REAL,
            feature_vector BLOB,
            FOREIGN KEY(material_id) REFERENCES materials(id)
        )
    """)
    # Table for keyframe descriptions, linked to a material
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keyframe_descriptions (
            material_id INTEGER,
            timestamp REAL,
            description TEXT,
            PRIMARY KEY (material_id, timestamp),
            FOREIGN KEY(material_id) REFERENCES materials(id)
        )
    """)
    conn.commit()
    conn.close()

def add_material(filename: str):
    """
    Adds a new material filename to the database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO materials (filename) VALUES (?)", (filename,))
    conn.commit()
    conn.close()

def material_exists(filename: str) -> bool:
    """
    Checks if a material with the given filename already exists in the database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM materials WHERE filename = ?", (filename,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_material_features(filename: str, features_data: list[tuple[float, np.ndarray]]):
    """
    Saves the features for each sampled frame of a material video to the database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # First, get or create the material ID
    cursor.execute("INSERT OR IGNORE INTO materials (filename) VALUES (?)", (filename,))
    cursor.execute("SELECT id FROM materials WHERE filename = ?", (filename,))
    material_id = cursor.fetchone()[0]

    # Delete old features for this material to prevent duplicates
    cursor.execute("DELETE FROM features WHERE material_id = ?", (material_id,))

    # Prepare data for bulk insert
    features_to_insert = [
        (material_id, timestamp, features.tobytes())
        for timestamp, features in features_data
    ]

    # Insert all features in a single transaction
    cursor.executemany("INSERT INTO features (material_id, timestamp, feature_vector) VALUES (?, ?, ?)", features_to_insert)
    
    conn.commit()
    conn.close()

def get_features_by_filename(filename: str) -> list[tuple[float, bytes]]:
    """
    Retrieves all feature vectors for a specific material filename.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.timestamp, f.feature_vector
        FROM features f
        JOIN materials m ON f.material_id = m.id
        WHERE m.filename = ?
        ORDER BY f.timestamp
    """, (filename,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_material_filenames() -> list[str]:
    """
    Retrieves all material filenames from the database.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM materials ORDER BY filename")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def save_material_descriptions(filename: str, descriptions_data: list[dict]):
    """
    Saves the semantic descriptions for each keyframe of a material video.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM materials WHERE filename = ?", (filename,))
    material_id_row = cursor.fetchone()
    if not material_id_row:
        print(f"Error: Material '{filename}' not found in database.")
        conn.close()
        return

    material_id = material_id_row[0]

    # Delete old descriptions to prevent duplicates on re-processing
    cursor.execute("DELETE FROM keyframe_descriptions WHERE material_id = ?", (material_id,))

    desc_to_insert = [
        (material_id, data['time'], data['description'])
        for data in descriptions_data
    ]

    cursor.executemany(
        "INSERT INTO keyframe_descriptions (material_id, timestamp, description) VALUES (?, ?, ?)",
        desc_to_insert
    )
    
    conn.commit()
    conn.close()

def get_descriptions_by_filename(filename: str) -> list[dict]:
    """
    Retrieves all keyframe descriptions for a specific material filename.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT kd.timestamp, kd.description
        FROM keyframe_descriptions kd
        JOIN materials m ON kd.material_id = m.id
        WHERE m.filename = ?
        ORDER BY kd.timestamp
    """, (filename,))
    rows = cursor.fetchall()
    conn.close()
    return [{"time": row[0], "description": row[1]} for row in rows]