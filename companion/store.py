import sqlite3
import time
from contextlib import contextmanager


class Store:
    def __init__(self, db_path):
        self._db_path = db_path
        # A persistent connection is needed only for ":memory:" databases,
        # which vanish when their last connection closes. File-backed stores
        # open a fresh connection per operation, which is what makes one
        # Store safe to share across ThreadingHTTPServer threads.
        self._keeper = sqlite3.connect(":memory:") if db_path == ":memory:" else None
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS companion_state (
                    companion_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_traces (
                    turn_id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_companion ON memories(companion_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reflection_log (
                    id TEXT PRIMARY KEY,
                    companion_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    rolled_back INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @contextmanager
    def _conn(self):
        if self._keeper is not None:
            yield self._keeper
            return
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def save_state(self, companion_id: str, data: str):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO companion_state (companion_id, data, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(companion_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at
                """,
                (companion_id, data, time.time()),
            )
            conn.commit()

    def load_state(self, companion_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM companion_state WHERE companion_id = ?", (companion_id,)
            ).fetchone()
            return row[0] if row else None

    def save_trace(self, turn_id: str, companion_id: str, data: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO turn_traces (turn_id, companion_id, data, created_at) VALUES (?, ?, ?, ?)",
                (turn_id, companion_id, data, time.time()),
            )
            conn.commit()

    def load_traces(self, companion_id: str, limit: int = 50) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM turn_traces WHERE companion_id = ? ORDER BY created_at DESC LIMIT ?",
                (companion_id, limit),
            ).fetchall()
            return [r[0] for r in rows]

    def save_memory(self, memory_id: str, companion_id: str, kind: str, data: str, created_at: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, companion_id, kind, data, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                """,
                (memory_id, companion_id, kind, data, created_at),
            )
            conn.commit()

    def load_memories(self, companion_id: str, kind: str | None = None) -> list[str]:
        with self._conn() as conn:
            if kind is None:
                rows = conn.execute(
                    "SELECT data FROM memories WHERE companion_id = ? ORDER BY created_at ASC",
                    (companion_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM memories WHERE companion_id = ? AND kind = ? ORDER BY created_at ASC",
                    (companion_id, kind),
                ).fetchall()
            return [r[0] for r in rows]

    def delete_memory(self, memory_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()

    def save_reflection(self, entry_id: str, companion_id: str, data: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO reflection_log (id, companion_id, data, rolled_back, created_at) VALUES (?, ?, ?, ?, ?)",
                (entry_id, companion_id, data, 0, time.time()),
            )
            conn.commit()

    def mark_rolled_back(self, entry_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE reflection_log SET rolled_back = 1 WHERE id = ?",
                (entry_id,),
            )
            conn.commit()

    def load_reflections(self, companion_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, data, rolled_back, created_at FROM reflection_log WHERE companion_id = ? ORDER BY created_at ASC",
                (companion_id,),
            ).fetchall()
        result = []
        for row_id, data, rolled_back, created_at in rows:
            entry = {"id": row_id, "created_at": created_at, "rolled_back": bool(rolled_back), "applied": data}
            result.append(__import__("json").dumps(entry))
        return result

    def list_state_meta(self) -> dict[str, float]:
        """companion_id -> updated_at, for every saved instance."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT companion_id, updated_at FROM companion_state"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def count_traces(self, companion_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM turn_traces WHERE companion_id = ?",
                (companion_id,)).fetchone()[0]

    def count_memories(self, companion_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM memories WHERE companion_id = ?",
                (companion_id,)).fetchone()[0]

    def purge_companion(self, companion_id: str) -> None:
        """Every row for this id, all four tables. Used by restart-as-new
        and by permanent purge. Irreversible."""
        with self._conn() as conn:
            for table in ("companion_state", "turn_traces", "memories",
                          "reflection_log"):
                conn.execute(f"DELETE FROM {table} WHERE companion_id = ?",
                             (companion_id,))
            conn.commit()
