"""
Collect - pull posts from Reddit via PRAW, or generate fixtures for testing.

PRAW needs a free Reddit app registration. The README documents the three
environment variables. Fixtures exist so the pipeline can be run and tested
before credentials are set up, and so the analysis code has something
deterministic to run against.
"""

import logging
import os

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


def collect_reddit(subreddits=None, limit=None, sort=None) -> pd.DataFrame:
    """Pull posts from each subreddit into one dataframe.

    One row per post. Comments are not collected - they roughly triple the
    volume and the topic signal in this pipeline comes from titles and post
    bodies, so the extra data would cost time without changing the output.
    """
    import praw                                  # imported here so fixtures
                                                 # work without praw installed
    subreddits = subreddits or config.SUBREDDITS
    limit = limit or config.POSTS_PER_SUBREDDIT
    sort = sort or config.SORT

    missing = [v for v in (config.ENV_CLIENT_ID, config.ENV_CLIENT_SECRET,
                           config.ENV_USER_AGENT) if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Register a script app at reddit.com/prefs/apps and set them, "
            "or run with --sample to use fixtures."
        )

    reddit = praw.Reddit(
        client_id=os.getenv(config.ENV_CLIENT_ID),
        client_secret=os.getenv(config.ENV_CLIENT_SECRET),
        user_agent=os.getenv(config.ENV_USER_AGENT),
    )

    rows = []
    for name in subreddits:
        log.info("Collecting r/%s (%s, limit %s)", name, sort, limit)
        listing = getattr(reddit.subreddit(name), sort)(limit=limit)
        for post in listing:
            rows.append({
                "post_id": post.id,
                "subreddit": name,
                "title": post.title,
                "body": post.selftext or "",
                "author": str(post.author),
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": pd.to_datetime(post.created_utc, unit="s"),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["post_id"])
    log.info("Collected %s unique posts across %s subreddits",
             len(df), len(subreddits))
    return df


def sample_posts(n_per_sub: int = 300) -> pd.DataFrame:
    """Deterministic fixtures with realistic structure and realistic problems.

    Includes bot posts, near-empty bodies and duplicate ids so the cleaning
    stage has something to remove. Themes differ per subreddit so topic
    modelling has genuine structure to find rather than noise.
    """
    rng = np.random.default_rng(42)

    themes = {
        "stocks": [
            "earnings report beat expectations revenue growth guidance raised",
            "fed interest rate decision inflation cpi print market reaction",
            "portfolio allocation index funds long term holding strategy",
            "semiconductor demand supply chain chip manufacturing outlook",
        ],
        "CryptoCurrency": [
            "bitcoin etf inflows institutional custody spot approval",
            "ethereum staking yield validator node network upgrade",
            "exchange security hack cold storage private key custody",
            "regulation sec enforcement compliance stablecoin framework",
        ],
        "artificial": [
            "language model context window benchmark reasoning evaluation",
            "gpu compute training cluster inference cost scaling",
            "open source weights license fine tuning community release",
            "agents tool use automation workflow deployment production",
        ],
    }
    openers = ["Thoughts on", "Discussion:", "Why is", "Anyone else seeing",
               "Update on", "Question about", "Analysis of"]

    rows = []
    base = pd.Timestamp("2024-06-01")
    for sub, topics in themes.items():
        for i in range(n_per_sub):
            theme = topics[rng.integers(len(topics))]
            words = theme.split()
            title = f"{openers[rng.integers(len(openers))]} {' '.join(rng.choice(words, 4))}"
            body = " ".join(rng.choice(words, rng.integers(25, 70)))
            # Volume rises over the window so trend detection has a signal.
            day = int(rng.triangular(0, 55, 60))
            rows.append({
                "post_id": f"{sub[:3]}{i:05d}",
                "subreddit": sub,
                "title": title,
                "body": body,
                "author": rng.choice(["user_a", "user_b", "AutoModerator", "user_c"],
                                     p=[0.35, 0.3, 0.05, 0.3]),
                "score": int(rng.gamma(2, 40)),
                "num_comments": int(rng.gamma(1.6, 12)),
                "created_utc": base + pd.Timedelta(days=day,
                                                   hours=int(rng.integers(0, 24))),
            })

    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.03, random_state=1).index, "body"] = ""   # empty bodies
    df = pd.concat([df, df.head(20)], ignore_index=True)              # duplicate ids
    log.warning("Using SYNTHETIC fixtures, not live Reddit data")
    return df
