"""
Reddit Intelligence Engine - full pipeline.

    python run_pipeline.py --sample     # fixtures, no credentials needed
    python run_pipeline.py              # live Reddit via PRAW

Stages: collect -> clean -> topics -> sentiment -> trends -> store
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

import config
import collect
import clean
import topics
import sentiment
import trends
import store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="use synthetic fixtures instead of live Reddit")
    parser.add_argument("--topics", type=int, default=config.N_TOPICS)
    args = parser.parse_args()

    started = time.time()
    log.info("=" * 62)
    log.info("REDDIT INTELLIGENCE ENGINE")
    log.info("=" * 62)

    try:
        log.info("[1/6] Collect")
        raw = collect.sample_posts() if args.sample else collect.collect_reddit()

        log.info("[2/6] Clean")
        posts = clean.clean_posts(raw)
        if len(posts) < 50:
            log.error("Only %s posts survived cleaning - too few to model", len(posts))
            return 1

        log.info("[3/6] Topic modelling")
        topic_table, assignments = topics.fit_topics(posts["document"], args.topics)
        posts = pd.concat([posts.reset_index(drop=True), assignments], axis=1)

        log.info("[4/6] Sentiment")
        scores = sentiment.score_sentiment(posts["title"] + " " + posts["body"])
        posts = pd.concat([posts, scores], axis=1)

        log.info("[5/6] Trend detection")
        volume = trends.daily_topic_volume(posts)
        volume = trends.detect_spikes(volume)

        log.info("[6/6] Store")
        sizes = topics.topic_sizes(assignments, topic_table)
        topic_sentiment = (posts.groupby("topic_id")["sentiment_score"]
                           .mean().round(3).rename("avg_sentiment").reset_index())
        summary = sizes.merge(topic_sentiment, on="topic_id")

        posts["post_date"] = posts["post_date"].astype(str)
        volume["post_date"] = volume["post_date"].astype(str)

        store.save({"posts": posts.drop(columns=["document"]),
                    "topics": summary,
                    "topic_volume": volume})

    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    log.info("=" * 62)
    log.info("PIPELINE COMPLETE in %.1fs", time.time() - started)
    log.info("Posts analysed: %s", len(posts))
    log.info("")
    log.info("Topics found:")
    for _, row in summary.iterrows():
        log.info("  [%2s] n=%-5s sentiment=%+.2f  %s",
                 row.topic_id, row.post_count, row.avg_sentiment, row.top_terms)
    log.info("")
    log.info("Spike days detected: %s", int(volume["is_spike"].sum()))
    log.info("Database: %s", config.DB_PATH)
    log.info("=" * 62)
    log.info("These are the numbers for your README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
