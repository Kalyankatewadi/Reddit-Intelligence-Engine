"""
Sentiment - VADER scoring.

VADER is a rule-based model tuned for social media. It handles negation,
intensifiers, emphasis and emoji without a training step, runs at thousands of
documents per second on CPU, and needs no model download.

A fine-tuned transformer would score better on nuance, but it would add a GPU-
scale dependency to a pipeline whose value is being runnable on a schedule.
VADER's known weakness is sarcasm, which Reddit has in quantity - the README
says so plainly rather than presenting these scores as precise.
"""

import logging

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05


def score_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Return compound score and a categorical label per document.

    Scoring runs on the ORIGINAL text, not the cleaned text, because VADER's
    rules depend on punctuation, capitalisation and emoji - exactly the signals
    the topic-modelling cleaner strips out.
    """
    analyzer = SentimentIntensityAnalyzer()
    compounds = [analyzer.polarity_scores(str(t))["compound"] for t in texts]

    labels = pd.cut(
        pd.Series(compounds),
        bins=[-1.01, NEGATIVE_CUTOFF, POSITIVE_CUTOFF, 1.01],
        labels=["negative", "neutral", "positive"],
    )
    log.info("Sentiment: %s", dict(labels.value_counts()))
    return pd.DataFrame({"sentiment_score": compounds,
                         "sentiment_label": labels.astype(str)})
