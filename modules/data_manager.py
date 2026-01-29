"""Data Management Module - SQLite database operations."""

import sqlite3
import numpy as np
from datetime import datetime
import config


class DataManager:
    """Handles user data and embedding storage in SQLite."""

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        """Open database connection and create tables if not exist."""
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """Create users table if not exists."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_user(self, user_id: str, name: str, embedding: np.ndarray) -> bool:
        """
        Add new user with embedding.
        Returns True if successful, False if user_id exists.
        """
        if self.user_exists(user_id):
            return False
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, name, embedding, created_at) VALUES (?, ?, ?, ?)",
            (user_id, name, embedding.tobytes(), datetime.now().isoformat())
        )
        self.conn.commit()
        
        return True

    def get_user(self, user_id: str) -> dict | None:
        """Get user by ID. Returns dict with user info or None."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT user_id, name, embedding, created_at FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'user_id': row[0],
                'name': row[1],
                'embedding': np.frombuffer(row[2], dtype=np.float64),
                'created_at': row[3]
            }
        
        return None

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Get all (user_id, embedding) pairs for matching."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id, embedding FROM users")
        rows = cursor.fetchall()
        
        return [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in rows]

    def delete_user(self, user_id: str) -> bool:
        """Delete user by ID. Returns True if deleted."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self.conn.commit()
        
        return cursor.rowcount > 0

    def user_exists(self, user_id: str) -> bool:
        """Check if user_id already exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        
        return cursor.fetchone() is not None