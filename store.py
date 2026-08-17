"""
Store - persist pipeline output to SQLite.

Tables are replaced rather than appended so a rerun is idempotent. The database
is the interface between the pipeline and the dashboard: the pipeline writes,
the dashboard only reads, and neither needs to know how the other works.
"""

import logging
import sqlite3

import pandas as pd

import config

log = logging.getLogger(__name__)


def save(tables: dict[str, pd.DataFrame]) -> dict:
    written = {}
    with sqlite3.connect(config.DB_PATH) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
            written[name] = len(df)
            log.info("Wrote %-16s %s rows", name, len(df))
        conn.commit()
    return written


def load(table: str) -> pd.DataFrame:
    with sqlite3.connect(config.DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
