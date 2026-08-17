"""
Streamlit dashboard over the pipeline output.

Read-only. It queries the SQLite database the pipeline writes and never
recomputes anything, so opening the dashboard is instant regardless of how long
the pipeline took. Run the pipeline first, then this.

    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

import config
import store

st.set_page_config(page_title="Reddit Intelligence Engine", layout="wide")


@st.cache_data
def load_all():
    posts = store.load("posts")
    topics_df = store.load("topics")
    volume = store.load("topic_volume")

    # SQLite has no boolean type, so is_spike round-trips as integer 0/1.
    # Casting it back is required before it can be used as a pandas mask -
    # an integer Series is read as column labels, not as a filter.
    volume["is_spike"] = volume["is_spike"].astype(bool)
    return posts, topics_df, volume


st.title("Reddit Intelligence Engine")

if not config.DB_PATH.exists():
    st.error("No database found. Run `python run_pipeline.py --sample` first.")
    st.stop()

posts, topics_df, volume = load_all()
volume["post_date"] = pd.to_datetime(volume["post_date"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts analysed", f"{len(posts):,}")
c2.metric("Topics", len(topics_df))
c3.metric("Subreddits", posts["subreddit"].nunique())
c4.metric("Spike days", int(volume["is_spike"].sum()))

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Topics by volume")
    chart = topics_df.set_index("top_terms")["post_count"].sort_values()
    st.bar_chart(chart, horizontal=True)

with right:
    st.subheader("Sentiment distribution")
    st.bar_chart(posts["sentiment_label"].value_counts())

st.divider()

st.subheader("Topic volume over time")
selected = st.multiselect(
    "Topics", options=topics_df["topic_id"].tolist(),
    default=topics_df["topic_id"].head(3).tolist(),
    format_func=lambda i: f"[{i}] " + topics_df.loc[
        topics_df.topic_id == i, "top_terms"].iloc[0][:45],
)
if selected:
    subset = volume[volume["topic_id"].isin(selected)]
    pivot = subset.pivot(index="post_date", columns="topic_id", values="post_count")
    st.line_chart(pivot)

    spikes = subset[subset["is_spike"]]
    if len(spikes):
        st.caption(f"{len(spikes)} spike day(s) in the selected topics")
        spike_table = spikes[["topic_id", "post_date", "post_count",
                              "rolling_mean", "zscore"]].copy()
        # round() warns on datetime columns, so only the numeric ones are rounded
        for col in ["rolling_mean", "zscore"]:
            spike_table[col] = spike_table[col].round(2)
        spike_table["post_date"] = spike_table["post_date"].dt.date
        st.dataframe(
            spike_table.sort_values("zscore", ascending=False).head(10),
            width="stretch", hide_index=True,
        )

st.divider()

st.subheader("Topic detail")
topic_id = st.selectbox(
    "Select a topic", topics_df["topic_id"],
    format_func=lambda i: f"[{i}] " + topics_df.loc[
        topics_df.topic_id == i, "top_terms"].iloc[0][:60],
)
detail = posts[posts["topic_id"] == topic_id].sort_values("score", ascending=False)
st.caption(f"{len(detail)} posts | average sentiment "
           f"{detail['sentiment_score'].mean():+.2f}")
st.dataframe(
    detail[["subreddit", "title", "score", "num_comments",
            "sentiment_label", "topic_weight"]].head(25).round(3),
    width="stretch", hide_index=True,
)
