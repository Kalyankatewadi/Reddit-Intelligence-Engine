"""
Configuration for the Reddit Intelligence Engine.

Every tunable lives here so the pipeline modules stay free of magic numbers.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "reddit.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
SUBREDDITS = ["stocks", "CryptoCurrency", "artificial"]
POSTS_PER_SUBREDDIT = 300
SORT = "hot"                    # hot | new | top

# PRAW reads these from environment variables. Never hardcode credentials -
# a committed API key is the most common way a public repo leaks something.
ENV_CLIENT_ID = "REDDIT_CLIENT_ID"
ENV_CLIENT_SECRET = "REDDIT_CLIENT_SECRET"
ENV_USER_AGENT = "REDDIT_USER_AGENT"

# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
MIN_DOC_CHARS = 60              # below this a post carries no topic signal
BOT_AUTHORS = {"AutoModerator", "[deleted]", "VisualMod"}

# --------------------------------------------------------------------------
# Topic modelling
# --------------------------------------------------------------------------
# TF-IDF + NMF rather than BERTopic. NMF fits in seconds on CPU, has no
# transformer dependency, and its components are directly readable as term
# weights. BERTopic clusters better on short informal text, but it needs
# sentence-transformers, UMAP and HDBSCAN - roughly 2GB of dependencies and
# minutes per run. For a pipeline meant to run on a schedule and be understood
# by whoever inherits it, the cheaper model is the right trade.
N_TOPICS = 10
N_TOP_WORDS = 8
MAX_FEATURES = 5_000
MIN_DF = 3                      # ignore terms in fewer than 3 documents
MAX_DF = 0.85                   # ignore terms in >85% of docs (boilerplate)

# --------------------------------------------------------------------------
# Trend detection
# --------------------------------------------------------------------------
ROLLING_WINDOW = 7              # days
ZSCORE_THRESHOLD = 2.0          # flag a topic-day above this many sigma
