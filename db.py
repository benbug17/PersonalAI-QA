"""
Database module for voice learning assistant.
Handles user authentication and conversation history using SQLite.
"""

import sqlite3
import bcrypt
from typing import Optional, List, Dict


DB_PATH = "app.db"


def init_db():
    """
    Initialize the database with required tables.
    Creates users and history tables if they don't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Conversation history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_query TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password as string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Plain text password to verify
        hashed: Stored password hash

    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_user(username: str, password: str) -> Optional[int]:
    """
    Create a new user account.

    Args:
        username: Unique username
        password: Plain text password (will be hashed)

    Returns:
        User ID if successful, None if username already exists
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )

        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        # Username already exists
        return None
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[int]:
    """
    Authenticate a user with username and password.

    Args:
        username: Username to authenticate
        password: Plain text password

    Returns:
        User ID if authentication successful, None otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username,)
    )

    result = cursor.fetchone()
    conn.close()

    if result and verify_password(password, result[1]):
        return result[0]

    return None


def save_conversation(user_id: int, user_query: str, assistant_response: str):
    """
    Save a conversation exchange to history.

    Args:
        user_id: ID of the user
        user_query: User's question/query
        assistant_response: Assistant's response
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO history (user_id, user_query, assistant_response)
           VALUES (?, ?, ?)""",
        (user_id, user_query, assistant_response)
    )

    conn.commit()
    conn.close()


def get_user_history(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Retrieve conversation history for a user.

    Args:
        user_id: ID of the user
        limit: Maximum number of records to retrieve

    Returns:
        List of conversation dictionaries with keys: id, query, response, timestamp
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """SELECT id, user_query, assistant_response, created_at
           FROM history
           WHERE user_id = ?
           ORDER BY created_at DESC, id DESC
           LIMIT ?""",
        (user_id, limit)
    )

    results = cursor.fetchall()
    conn.close()

    history = []
    for row in results:
        history.append({
            'id': row[0],
            'query': row[1],
            'response': row[2],
            'timestamp': row[3]
        })

    return history


def get_username(user_id: int) -> Optional[str]:
    """
    Get username for a given user ID.

    Args:
        user_id: ID of the user

    Returns:
        Username if found, None otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None
