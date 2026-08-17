"""
Clean - preprocess post text into something a topic model can use.

Every rule here removes a specific category of noise that measurably degrades
topic quality on Reddit text. The rules are ordered cheapest-first so expensive
regex work runs on the smallest surviving set of documents.
"""

import logging
import re

import pandas as pd

import config

log = logging.getLogger(__name__)

URL_RE = re.compile(r"http\S+|www\.\S+")
MARKDOWN_RE = re.compile(r"[*_~`>#\[\]()]|&amp;|&gt;|&lt;")
WHITESPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^a-z\s]")


def clean_text(raw: str) -> str:
    """Lowercase, strip URLs, markdown and non-alphabetic characters."""
    text = str(raw).lower()
    text = URL_RE.sub(" ", text)
    text = MARKDOWN_RE.sub(" ", text)
    text = NON_TEXT_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule and report what each one removed.

    Reporting per-rule counts matters: if one rule suddenly removes most of the
    input, that is a signal the source changed, not that the rule is working.
    """
    start = len(df)

    df = df.drop_duplicates(subset=["post_id"]).copy()
    log.info("  dropped %s duplicate post ids", start - len(df))

    before = len(df)
    df = df[~df["author"].isin(config.BOT_AUTHORS)]
    log.info("  dropped %s bot/removed posts", before - len(df))

    # Title and body are concatenated because Reddit titles carry as much topic
    # signal as bodies, and link posts have no body at all.
    df["document"] = (df["title"].fillna("") + " " + df["body"].fillna("")).map(clean_text)

    before = len(df)
    df = df[df["document"].str.len() >= config.MIN_DOC_CHARS]
    log.info("  dropped %s posts under %s chars", before - len(df),
             config.MIN_DOC_CHARS)

    df["post_date"] = pd.to_datetime(df["created_utc"]).dt.date
    log.info("Cleaning kept %s of %s posts (%.1f%%)",
             len(df), start, len(df) / start * 100)
    return df.reset_index(drop=True)
