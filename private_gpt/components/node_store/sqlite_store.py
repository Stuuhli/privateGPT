"""SQLite-backed node store implementations."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

DOCSTORE_DB = "docstore.sqlite"
INDEXSTORE_DB = "indexstore.sqlite"


def _ensure_docstore_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ref_doc_info (
            doc_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """
    )


def _write_docstore(db_path: Path, data: dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        _ensure_docstore_schema(connection)
        docs = data.get("documents", {})
        connection.execute("DELETE FROM documents")
        if docs:
            connection.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, payload) VALUES (?, ?)",
                ((doc_id, json.dumps(payload)) for doc_id, payload in docs.items()),
            )

        ref_docs = data.get("ref_doc_info", {})
        connection.execute("DELETE FROM ref_doc_info")
        if ref_docs:
            connection.executemany(
                "INSERT OR REPLACE INTO ref_doc_info (doc_id, payload) VALUES (?, ?)",
                ((doc_id, json.dumps(payload)) for doc_id, payload in ref_docs.items()),
            )

        extra = {k: v for k, v in data.items() if k not in {"documents", "ref_doc_info"}}
        connection.execute("DELETE FROM metadata")
        if extra:
            connection.executemany(
                "INSERT OR REPLACE INTO metadata (key, payload) VALUES (?, ?)",
                ((key, json.dumps(value)) for key, value in extra.items()),
            )
        connection.commit()
    finally:
        connection.close()


def _read_docstore(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    connection = sqlite3.connect(db_path)
    try:
        _ensure_docstore_schema(connection)
        data: dict[str, Any] = {"documents": {}, "ref_doc_info": {}}
        for doc_id, payload in connection.execute("SELECT doc_id, payload FROM documents"):
            data["documents"][doc_id] = json.loads(payload)
        for doc_id, payload in connection.execute(
            "SELECT doc_id, payload FROM ref_doc_info"
        ):
            data["ref_doc_info"][doc_id] = json.loads(payload)
        for key, payload in connection.execute("SELECT key, payload FROM metadata"):
            data[key] = json.loads(payload)
        return data
    finally:
        connection.close()


def _write_indexstore(db_path: Path, data: dict[str, Any]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM index_metadata")
        if data:
            connection.executemany(
                "INSERT OR REPLACE INTO index_metadata (key, payload) VALUES (?, ?)",
                ((key, json.dumps(value)) for key, value in data.items()),
            )
        connection.commit()
    finally:
        connection.close()


def _read_indexstore(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        data: Dict[str, Any] = {}
        for key, payload in connection.execute("SELECT key, payload FROM index_metadata"):
            data[key] = json.loads(payload)
        return data
    finally:
        connection.close()


class SqliteDocumentStore(SimpleDocumentStore):
    """SimpleDocumentStore that persists data in SQLite."""

    def __init__(self) -> None:
        super().__init__()
        self._db_path: Path | None = None

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str | Path,
        db_name: str | None = None,
    ) -> "SqliteDocumentStore":
        db_path = Path(persist_dir) / (db_name or DOCSTORE_DB)
        data = _read_docstore(db_path)
        base_store = SimpleDocumentStore.from_dict(data)
        instance = cls()
        instance.__dict__.update(base_store.__dict__)
        instance._db_path = db_path
        return instance

    def persist(
        self,
        persist_dir: str | Path,
        db_name: str | None = None,
    ) -> None:
        db_path = Path(persist_dir) / (db_name or DOCSTORE_DB)
        _write_docstore(db_path, self.to_dict())
        self._db_path = db_path


class SqliteIndexStore(SimpleIndexStore):
    """SimpleIndexStore that persists data in SQLite."""

    def __init__(self) -> None:
        super().__init__()
        self._db_path: Path | None = None

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str | Path,
        db_name: str | None = None,
    ) -> "SqliteIndexStore":
        db_path = Path(persist_dir) / (db_name or INDEXSTORE_DB)
        data = _read_indexstore(db_path)
        base_store = SimpleIndexStore.from_dict(data)
        instance = cls()
        instance.__dict__.update(base_store.__dict__)
        instance._db_path = db_path
        return instance

    def persist(
        self,
        persist_dir: str | Path,
        db_name: str | None = None,
    ) -> None:
        db_path = Path(persist_dir) / (db_name or INDEXSTORE_DB)
        _write_indexstore(db_path, self.to_dict())
        self._db_path = db_path
