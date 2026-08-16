Reddit Intelligence Engine
Real-time NLP pipeline that collects Reddit posts, models the topics being discussed, scores sentiment, detects trend spikes, and serves the results through a Streamlit dashboard.

Monitored subreddits: r/stocks, r/crypto, r/artificial


What it does
Reddit produces more discussion than anyone can read, and keyword search only finds what you already know to look for. This pipeline structures the text so it can be aggregated and trended: every post gets a topic, a sentiment label, and a timestamp, which makes it possible to ask what a subreddit is talking about this week and whether that is changing.
Pipeline
1. Collection (PRAW)

Scrapes live posts and writes them to SQLite. Stored fields: title, body, score, comment count, timestamp.

2. Cleaning (spaCy)

URL removal, tokenization, lemmatization, stopword removal.

3. Topic modeling (BERTopic)

Transformer embeddings, UMAP for dimensionality reduction, HDBSCAN for clustering. Outputs topic IDs, keywords per topic, and topic probabilities.

Embedding-based clustering was chosen over keyword methods because the same subject gets discussed in very different vocabulary across threads. Bag-of-words approaches split those into several weak topics instead of one coherent topic.

4. Sentiment analysis (Transformers)

Classifies each post as positive, negative, or neutral.

5. Trend detection

Rolling mean and rolling standard deviation over topic volume, with z-score thresholds to flag spikes. A topic accelerating is a stronger signal than a topic that is simply large.

6. Dashboard (Streamlit)

Topic frequency bar chart
Sentiment distribution
UMAP 2D embedding plot
Intertopic distance map
Topic timeline heatmap
Topic details viewer
Tech stack
Python, PRAW, spaCy, BERTopic, UMAP, HDBSCAN, Transformers, SQLite, Streamlit, pandas
Project structure
TODO: paste your actual folder structure here
Setup
git clone https://github.com/Kalyankatewadi/Reddit-Intelligence-Engine.git

cd Reddit-Intelligence-Engine

pip install -r requirements.txt

python -m spacy download en_core_web_sm

PRAW requires Reddit API credentials. Create an app at reddit.com/prefs/apps and set:

export REDDIT_CLIENT_ID=your_id

export REDDIT_CLIENT_SECRET=your_secret

export REDDIT_USER_AGENT=your_agent_string
Running
python scrape.py          # collect posts into SQLite

python pipeline.py        # clean, model topics, score sentiment

streamlit run app.py      # launch the dashboard

TODO: correct these filenames to match your repo.
Scale
TODO: add posts collected, date range, and number of topics identified.
Notes and limitations
Topic labels come from cluster keywords and are worth reviewing before presenting as findings. Sentiment models trained on general text handle sarcasm and community slang poorly, and both are common on Reddit, so treat sentiment as directional rather than precise. HDBSCAN assigns some posts to an outlier cluster by design; these are excluded from topic-level aggregates.



Kalyan Katewadi · Portfolio · LinkedIn

