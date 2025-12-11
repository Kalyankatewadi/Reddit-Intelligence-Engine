import streamlit as st
import sqlite3
import pandas as pd
from bertopic import BERTopic
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

# ---------------------------------------
# Load Data
# ---------------------------------------
st.title("Reddit Intelligence Engine — Advanced Dashboard")

conn = sqlite3.connect("database/reddit.db")
df = pd.read_sql_query("SELECT * FROM posts", conn)

df["full_text"] = df["title"].fillna("") + " " + df["body"].fillna("")

st.sidebar.header("Dashboard Controls")
subreddit_filter = st.sidebar.multiselect("Select Subreddits:", df["subreddit"].unique(), df["subreddit"].unique())
df = df[df["subreddit"].isin(subreddit_filter)]

# ---------------------------------------
# Load BERTopic Model
# ---------------------------------------
topic_model = BERTopic.load("data/models/bertopic_model")

topics, probs = topic_model.transform(df["full_text"].tolist())
df["topic"] = topics

# ---------------------------------------
# Topic Frequency Bar Chart
# ---------------------------------------
st.subheader("Topic Frequency Distribution")
freq = df["topic"].value_counts().reset_index()
freq.columns = ["topic", "count"]

fig = px.bar(freq, x="topic", y="count", title="Topics by Frequency", color="count")
st.plotly_chart(fig)

# ---------------------------------------
# Sentiment Distribution
# ---------------------------------------
st.subheader("Sentiment Distribution")

from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")

df["sentiment"] = df["full_text"].apply(lambda x: sentiment_model(x[:512])[0]["label"])

sent_freq = df["sentiment"].value_counts()

fig = px.pie(names=sent_freq.index, values=sent_freq.values, title="Sentiment Breakdown")
st.plotly_chart(fig)

# ---------------------------------------
# UMAP Topic Embedding Visualization
# ---------------------------------------
st.subheader("UMAP Topic Embeddings")

embeddings = topic_model.reduce_dimensionality(topic_model.embedding_model.transform(df["full_text"].tolist()))

fig = px.scatter(
    x=embeddings[:, 0],
    y=embeddings[:, 1],
    color=df["topic"],
    title="UMAP Embedding of Reddit Topics",
    labels={"x": "UMAP-1", "y": "UMAP-2"}
)

st.plotly_chart(fig)

# ---------------------------------------
# Intertopic Distance Map
# ---------------------------------------
st.subheader("Intertopic Distance Map")

fig = topic_model.visualize_topics()
st.components.v1.html(fig.to_html(), height=600)

# ---------------------------------------
# Topic Timeline
# ---------------------------------------
st.subheader("Topic Timeline Heatmap")

df["time"] = pd.to_datetime(df["created_utc"], unit="s")
df["date"] = df["time"].dt.date

timeline = df.groupby(["date", "topic"]).size().reset_index(name="count")

fig = px.density_heatmap(
    timeline,
    x="date",
    y="topic",
    z="count",
    title="Topic Activity Over Time",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig)

# ---------------------------------------
# Topic Summary Viewer
# ---------------------------------------
st.subheader("Topic Summaries")

topic_info = topic_model.get_topic_info()

st.write(topic_info)

topic_id = st.selectbox("Select Topic ID to View Details:", topic_info["Topic"].unique())

if topic_id != -1:
    st.write(topic_model.get_topic(topic_id))
