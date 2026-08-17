"""
Trends - detect topic volume spikes with a rolling z-score.

A topic being large is not news. A topic being unusually large relative to its
own recent baseline is. The z-score is computed per topic against that topic's
own rolling mean and standard deviation, so a consistently busy topic does not
flag every day while a normally quiet one flags when it wakes up.
"""

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def daily_topic_volume(posts: pd.DataFrame) -> pd.DataFrame:
    """One row per topic per day, with zero-filled gaps.

    Zero-filling matters: without it, a topic that goes silent for three days
    has those days missing rather than recorded as zero, and the rolling mean
    reads a quiet period as if it never happened.
    """
    volume = (posts.groupby(["topic_id", "post_date"])
              .size().rename("post_count").reset_index())

    all_dates = pd.date_range(posts["post_date"].min(),
                              posts["post_date"].max(), freq="D").date
    grid = pd.MultiIndex.from_product(
        [sorted(posts["topic_id"].unique()), all_dates],
        names=["topic_id", "post_date"],
    ).to_frame(index=False)

    return grid.merge(volume, on=["topic_id", "post_date"], how="left") \
               .fillna({"post_count": 0})


def detect_spikes(volume: pd.DataFrame, window: int = None,
                  threshold: float = None) -> pd.DataFrame:
    """Flag topic-days whose volume exceeds the rolling baseline by n sigma."""
    window = window or config.ROLLING_WINDOW
    threshold = threshold or config.ZSCORE_THRESHOLD

    out = volume.sort_values(["topic_id", "post_date"]).copy()
    grouped = out.groupby("topic_id")["post_count"]

    # shift(1) excludes the current day from its own baseline. Without it a
    # large day inflates the mean it is being compared against and the spike
    # partially hides itself.
    out["rolling_mean"] = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    out["rolling_std"] = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).std())

    out["zscore"] = (out["post_count"] - out["rolling_mean"]) / \
                    out["rolling_std"].replace(0, np.nan)
    out["is_spike"] = out["zscore"] > threshold

    spikes = int(out["is_spike"].sum())
    log.info("Trend detection: %s spike days above %.1f sigma", spikes, threshold)
    return out
