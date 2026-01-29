"""Data Management Module - SQLite database operations."""

import sqlite3
import numpy as np
import config


class DataManager:
    """Handles user data and embedding storage in SQLite."""

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self) -> None:
        """Open database connection and create tables if not exist."""
        # TODO: Create connection and call create_tables()
        pass

    def close(self) -> None:
        """Close database connection."""
        # TODO: Close connection if open
        pass

    def create_tables(self) -> None:
        """Create users table if not exists."""
        # TODO: CREATE TABLE users (id, name, embedding BLOB, created_at)
        pass

    def add_user(self, user_id: str, name: str, embedding: np.ndarray) -> bool:
        """
        Add new user with embedding.
        Returns True if successful, False if user_id exists.
        """
        # TODO: Convert embedding to bytes with .tobytes()
        # INSERT into database
        pass

    def get_user(self, user_id: str) -> dict | None:
        """Get user by ID. Returns dict with user info or None."""
        # TODO: SELECT and convert embedding bytes back to numpy
        pass

    def get_all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Get all (user_id, embedding) pairs for matching."""
        # TODO: SELECT all and convert embeddings
        pass

    def delete_user(self, user_id: str) -> bool:
        """Delete user by ID. Returns True if deleted."""
        # TODO: DELETE from database
        pass

    def user_exists(self, user_id: str) -> bool:
        """Check if user_id already exists."""
        # TODO: SELECT count
        pass