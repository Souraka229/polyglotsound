"""Consent, provenance and validation registry (SQLite, local, append-only by convention).

Rules enforced in code:
- no voice model creation or voice conversion without an active 'voice_model' consent;
- every generation records the original work hash, artist, version, models and watermark id;
- every artist validation (lyrics, adaptation choice, final render) is logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("POLYGLOT_REGISTRY", Path(__file__).resolve().parents[2] / "datasets" / "registry.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS artists (id INTEGER PRIMARY KEY, name TEXT NOT NULL, contact TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS consents (id INTEGER PRIMARY KEY, artist_id INTEGER NOT NULL, scope TEXT NOT NULL,
  statement TEXT NOT NULL, statement_sha256 TEXT NOT NULL, granted_at TEXT NOT NULL, revoked_at TEXT,
  FOREIGN KEY(artist_id) REFERENCES artists(id));
CREATE TABLE IF NOT EXISTS works (id INTEGER PRIMARY KEY, artist_id INTEGER NOT NULL, title TEXT,
  language TEXT, audio_sha256 TEXT UNIQUE, created_at TEXT);
CREATE TABLE IF NOT EXISTS generations (id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL, artist_id INTEGER NOT NULL,
  version INTEGER NOT NULL, source_lang TEXT, target_lang TEXT, mode TEXT, lyrics_json TEXT, models_json TEXT,
  watermark_message INTEGER, output_sha256 TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS validations (id INTEGER PRIMARY KEY, generation_id INTEGER, work_id INTEGER,
  step TEXT NOT NULL, decision TEXT NOT NULL, by_whom TEXT, comment TEXT, created_at TEXT);
"""

CONSENT_SCOPES = {"voice_model", "voice_conversion", "adaptation", "publication"}


class ConsentError(PermissionError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    def __init__(self, path: str | Path | None = None):
        path = path or DB_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _ins(self, sql: str, args: tuple) -> int:
        cur = self.db.execute(sql, args)
        self.db.commit()
        return int(cur.lastrowid)

    def add_artist(self, name: str, contact: str = "") -> int:
        row = self.db.execute("SELECT id FROM artists WHERE name=?", (name,)).fetchone()
        return row["id"] if row else self._ins("INSERT INTO artists(name,contact,created_at) VALUES(?,?,?)",
                                               (name, contact, now()))

    def grant_consent(self, artist_id: int, scope: str, statement: str) -> int:
        if scope not in CONSENT_SCOPES:
            raise ValueError(f"Portee inconnue : {scope}")
        if len(statement.strip()) < 20:
            raise ValueError("La declaration de consentement doit etre explicite (texte complet).")
        digest = hashlib.sha256(statement.encode("utf-8")).hexdigest()
        return self._ins("INSERT INTO consents(artist_id,scope,statement,statement_sha256,granted_at) VALUES(?,?,?,?,?)",
                         (artist_id, scope, statement, digest, now()))

    def revoke_consent(self, artist_id: int, scope: str) -> None:
        self.db.execute("UPDATE consents SET revoked_at=? WHERE artist_id=? AND scope=? AND revoked_at IS NULL",
                        (now(), artist_id, scope))
        self.db.commit()

    def has_consent(self, artist_id: int, scope: str) -> bool:
        return self.db.execute("SELECT 1 FROM consents WHERE artist_id=? AND scope=? AND revoked_at IS NULL",
                               (artist_id, scope)).fetchone() is not None

    def require_consent(self, artist_id: int, scope: str) -> None:
        if not self.has_consent(artist_id, scope):
            raise ConsentError(f"Consentement '{scope}' absent ou revoque pour l'artiste {artist_id}.")

    def add_work(self, artist_id: int, audio_path: str | Path, title: str, language: str) -> int:
        digest = sha256_file(audio_path)
        row = self.db.execute("SELECT id FROM works WHERE audio_sha256=?", (digest,)).fetchone()
        if row:
            return row["id"]
        return self._ins("INSERT INTO works(artist_id,title,language,audio_sha256,created_at) VALUES(?,?,?,?,?)",
                         (artist_id, title, language, digest, now()))

    def add_generation(self, work_id: int, artist_id: int, source_lang: str, target_lang: str, mode: str,
                       lyrics: list[dict], models: dict) -> tuple[int, int, int]:
        """Return (generation_id, version, watermark_message)."""
        version = 1 + (self.db.execute("SELECT COUNT(*) c FROM generations WHERE work_id=? AND target_lang=?",
                                       (work_id, target_lang)).fetchone()["c"])
        gid = self._ins("INSERT INTO generations(work_id,artist_id,version,source_lang,target_lang,mode,lyrics_json,"
                        "models_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (work_id, artist_id, version, source_lang, target_lang, mode,
                         json.dumps(lyrics, ensure_ascii=False), json.dumps(models), now()))
        message = gid & 0xFFFF
        self.db.execute("UPDATE generations SET watermark_message=? WHERE id=?", (message, gid))
        self.db.commit()
        return gid, version, message

    def finalize_generation(self, generation_id: int, output_path: str | Path) -> None:
        self.db.execute("UPDATE generations SET output_sha256=? WHERE id=?", (sha256_file(output_path), generation_id))
        self.db.commit()

    def log_validation(self, step: str, decision: str, by_whom: str, comment: str = "",
                       generation_id: int | None = None, work_id: int | None = None) -> int:
        return self._ins("INSERT INTO validations(generation_id,work_id,step,decision,by_whom,comment,created_at) "
                         "VALUES(?,?,?,?,?,?,?)", (generation_id, work_id, step, decision, by_whom, comment, now()))

    def lookup_watermark(self, message: int) -> list[dict]:
        rows = self.db.execute("SELECT g.*, w.title, a.name artist FROM generations g JOIN works w ON w.id=g.work_id "
                               "JOIN artists a ON a.id=g.artist_id WHERE g.watermark_message=?", (message,)).fetchall()
        return [dict(r) for r in rows]

    def history(self, work_id: int) -> dict:
        q = lambda sql: [dict(r) for r in self.db.execute(sql, (work_id,)).fetchall()]  # noqa: E731
        return {"generations": q("SELECT id,version,target_lang,mode,watermark_message,created_at FROM generations WHERE work_id=?"),
                "validations": q("SELECT step,decision,by_whom,comment,created_at FROM validations WHERE work_id=?")}
