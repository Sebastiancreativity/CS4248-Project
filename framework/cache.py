"""SQLite cache of auditor and COMET scores, keyed by a hash of scorer, source and candidate."""
import hashlib
import json
import sqlite3
from pathlib import Path

from .data import OFFICIAL

DB = OFFICIAL / "cache" / "scores.sqlite"


def key(*parts):
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path=DB):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS s (ns TEXT, k TEXT, v TEXT, PRIMARY KEY (ns, k))")

    def get_many(self, ns, keys):
        out = {}
        keys = list(keys)
        for b in range(0, len(keys), 500):
            chunk = keys[b:b + 500]
            q = f"SELECT k, v FROM s WHERE ns = ? AND k IN ({','.join('?' * len(chunk))})"
            out.update({k: json.loads(v) for k, v in self.db.execute(q, [ns, *chunk])})
        return out

    def put_many(self, ns, items):
        self.db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?, ?)",
                            [(ns, k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()])
        self.db.commit()
