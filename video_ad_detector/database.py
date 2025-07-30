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
    conn.commit()
    conn.close()

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