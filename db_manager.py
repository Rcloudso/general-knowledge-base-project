import sqlite3
import json
import time
import os
from typing import List, Dict, Any, Optional
from loguru import logger


class DatabaseManager:
    """管理SQLite数据库，用于存储会话内容"""

    def __init__(self, db_path: str = "./db/sessions.db"):
        """初始化数据库管理器

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库，创建必要的表"""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 会话表
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            last_updated REAL NOT NULL
        )
        """
        )

        # 消息表
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
        )
        """
        )

        cursor.execute("PRAGMA foreign_keys = ON")

        conn.commit()
        conn.close()

        logger.info(f"Database initialized at {self.db_path}")

    def create_session(self, session_id: str, system_message: str = None) -> bool:
        """创建新的会话

        Args:
            session_id: 会话ID
            system_message: 系统消息内容

        Returns:
            是否成功创建
        """
        try:
            current_time = time.time()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO sessions (session_id, created_at, last_updated) VALUES (?, ?, ?)",
                (session_id, current_time, current_time),
            )

            if system_message:
                cursor.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, "system", system_message, current_time),
                )

            conn.commit()
            conn.close()
            logger.info(f"Created session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            return False

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """向会话添加消息

        Args:
            session_id: 会话ID
            role: 消息角色 ("user", "assistant", "system")
            content: 消息内容

        Returns:
            是否成功添加
        """
        try:
            current_time = time.time()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            if not cursor.fetchone():
                logger.warning(f"Session {session_id} not found when adding message")
                conn.close()
                return False

            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, current_time),
            )

            cursor.execute(
                "UPDATE sessions SET last_updated = ? WHERE session_id = ?",
                (current_time, session_id),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            return False

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的所有消息

        Args:
            session_id: 会话ID

        Returns:
            消息列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            )

            messages = []
            for role, content in cursor.fetchall():
                messages.append({"role": role, "content": content})

            conn.close()
            return messages
        except Exception as e:
            logger.error(f"Error getting messages for session {session_id}: {e}")
            return []

    def list_sessions(self) -> Dict[str, float]:
        """列出所有会话

        Returns:
            会话ID和创建时间的字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT session_id, created_at FROM sessions ORDER BY created_at DESC"
            )

            sessions = {}
            for session_id, created_at in cursor.fetchall():
                sessions[session_id] = created_at

            conn.close()
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return {}

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其所有消息

        Args:
            session_id: 会话ID

        Returns:
            是否成功删除
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("PRAGMA foreign_keys = ON")

            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Deleted session: {session_id}")
            else:
                logger.warning(f"Session {session_id} not found when deleting")

            return deleted
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在

        Args:
            session_id: 会话ID

        Returns:
            会话是否存在
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            exists = cursor.fetchone() is not None

            conn.close()
            return exists
        except Exception as e:
            logger.error(f"Error checking if session {session_id} exists: {e}")
            return False
