"""
Topics - TF-IDF vectorisation plus NMF factorisation.

NMF decomposes the document-term matrix into topic-term and document-topic
matrices, both constrained non-negative. Non-negativity is what makes the output
readable: every topic is a positive combination of terms, so the top-weighted
terms genuinely describe it. LDA gives similar output but is slower to fit and
more sensitive to its priors on short text.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import config

log = logging.getLogger(__name__)

# Reddit-specific stopwords. The sklearn English list does not cover forum
# filler, and leaving these in produces topics whose top terms are "just",
# "think", "know" - technically frequent, analytically useless.
EXTRA_STOPWORDS = [
    "just", "like", "know", "think", "really", "people", "want", "going",
    "make", "time", "way", "thing", "things", "good", "lot", "sure", "does",
    "did", "doing", "got", "get", "getting", "say", "said", "new", "use",
    "using", "used", "post", "edit", "thanks", "https", "www", "com",
]


def fit_topics(documents: pd.Series, n_topics: int = None):
    """Fit TF-IDF + NMF. Returns the model pieces and per-document assignments."""
    n_topics = n_topics or config.N_TOPICS

    stopwords = list(TfidfVectorizer(stop_words="english")
                     .get_stop_words()) + EXTRA_STOPWORDS

    # sublinear_tf dampens repeated terms - a post saying "bitcoin" 30 times is
    # not 30 times more about bitcoin than one saying it once.
    vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF,
        stop_words=stopwords,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)
    log.info("TF-IDF matrix: %s documents x %s terms", *matrix.shape)

    model = NMF(n_components=n_topics, init="nndsvd", random_state=42,
                max_iter=400)
    doc_topic = model.fit_transform(matrix)
    log.info("NMF fitted: %s topics, reconstruction error %.3f",
             n_topics, model.reconstruction_err_)

    terms = np.array(vectorizer.get_feature_names_out())
    labels = []
    for idx, component in enumerate(model.components_):
        top = terms[component.argsort()[::-1][:config.N_TOP_WORDS]]
        labels.append({"topic_id": idx, "top_terms": ", ".join(top)})
    topic_table = pd.DataFrame(labels)

    # Assign each document its highest-weight topic, and keep that weight as a
    # confidence score. A document whose top weight is near zero matched nothing
    # well, and downstream aggregates can filter on that.
    assignments = pd.DataFrame({
        "topic_id": doc_topic.argmax(axis=1),
        "topic_weight": doc_topic.max(axis=1),
    })
    return topic_table, assignments


def topic_sizes(assignments: pd.DataFrame, topic_table: pd.DataFrame) -> pd.DataFrame:
    sizes = (assignments.groupby("topic_id").size()
             .rename("post_count").reset_index())
    return topic_table.merge(sizes, on="topic_id").sort_values(
        "post_count", ascending=False)
