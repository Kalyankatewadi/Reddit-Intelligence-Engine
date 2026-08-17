# Reddit Intelligence Engine

An NLP pipeline that collects Reddit posts, models the topics being discussed, scores sentiment, detects volume spikes, and serves the results through a Streamlit dashboard.

**Monitored subreddits:** r/stocks, r/CryptoCurrency, r/artificial

\---

## The problem

A busy subreddit produces more discussion than anyone can read, and keyword search only finds what you already know to look for. The useful questions — what is being discussed this week, how do people feel about it, what is accelerating — need the text structured before they can be asked at all.

This pipeline gives every post a topic, a sentiment score and a date, which turns unstructured discussion into something that can be aggregated and trended.

## Design decisions

**TF-IDF + NMF instead of BERTopic.** NMF fits in under a second on CPU and its output is directly readable, since non-negativity means each topic is a positive combination of terms. BERTopic clusters short informal text better, but it pulls in sentence-transformers, UMAP and HDBSCAN — roughly 2GB of dependencies and minutes per run. For a pipeline meant to run on a schedule and be maintained by whoever inherits it, the cheaper model is the better trade. If topic quality became the binding constraint, this is the first thing I would change.

**VADER for sentiment.** Rule-based, tuned for social media, handles negation and intensifiers, no training step and no model download. It scores thousands of documents per second.

**Sentiment runs on raw text, topics run on cleaned text.** VADER's rules depend on punctuation, capitalisation and emoji — exactly what the topic cleaner strips. Running both on the same preprocessed text would break one of them.

**Title and body concatenated.** Reddit titles carry as much topic signal as bodies, and link posts have no body at all.

**Rolling z-score per topic, not global.** A topic being large is not news; a topic being large relative to its own recent baseline is. The current day is excluded from its own baseline via `shift(1)`, otherwise a spike inflates the mean it is measured against and partly hides itself.

**Zero-filled daily volume.** Without it, a topic that goes quiet has missing days rather than zeroes, and the rolling mean reads the silence as if it never happened.

## Pipeline

```
collect → clean → topics → sentiment → trends → store
```

**Collect.** PRAW pulls posts per subreddit. Comments are not collected — they roughly triple the volume without changing the topic signal.

**Clean.** Duplicate ids, bot and removed authors, URLs, markdown and non-alphabetic characters are stripped, then documents under 60 characters are dropped as carrying no topic signal. Each rule reports what it removed, so a rule suddenly removing most of the input is visible as a source change rather than silent.

**Topics.** TF-IDF with bigrams and sublinear term frequency, then NMF. A custom stopword list extends sklearn's English list with forum filler — without it the top terms come out as "just", "think", "know".

**Sentiment.** VADER compound score per post, bucketed into positive, neutral and negative.

**Trends.** Daily volume per topic, zero-filled, with a rolling mean and standard deviation and a z-score threshold to flag spikes.

**Store.** Everything lands in SQLite. The dashboard only reads, so it opens instantly regardless of pipeline runtime.

## Tech stack

Python, PRAW, scikit-learn, VADER, pandas, SQLite, Streamlit

## Project structure

```
├── run\_pipeline.py      orchestrator
├── app.py               Streamlit dashboard
├── config.py            subreddits, thresholds, model parameters
├── requirements.txt
├── src/
│   ├── collect.py       PRAW collection and test fixtures
│   ├── clean.py         text preprocessing
│   ├── topics.py        TF-IDF and NMF
│   ├── sentiment.py     VADER scoring
│   ├── trends.py        rolling z-score spike detection
│   └── store.py         SQLite persistence
└── data/                the generated database
```

## Running it

```bash
git clone https://github.com/Kalyankatewadi/Reddit-Intelligence-Engine.git
cd Reddit-Intelligence-Engine
pip install -r requirements.txt
```

Run against synthetic fixtures, no credentials needed:

```bash
python run\_pipeline.py --sample
```

Run against live Reddit. Register a script app at reddit.com/prefs/apps, then set:

```bash
set REDDIT\_CLIENT\_ID=your\_id
set REDDIT\_CLIENT\_SECRET=your\_secret
set REDDIT\_USER\_AGENT=script:reddit-intel:v1 (by /u/yourname)

python run\_pipeline.py
```

Credentials are read from environment variables and never stored in the repo.

Then launch the dashboard:

```bash
streamlit run app.py
```

## Test fixtures

`--sample` generates posts across twelve planted themes, with bot authors, empty bodies and duplicate ids included so the cleaning rules have something to catch. The fixtures are for exercising the pipeline, never presented as findings.

## Results

Run against synthetic fixtures (920 posts across twelve planted themes):



| Stage | Result |

|---|---|

| Posts collected | 920 |

| Removed as duplicates | 20 |

| Removed as bot/deleted authors | 39 |

| Removed as too short | 25 |

| Posts analysed | 836 (90.9% retained) |

| TF-IDF matrix | 836 documents x 811 terms |

| Topics fitted | 10 |

| Spike days detected | 58 |

| Runtime | 0.4s |



The model recovered the planted themes cleanly, separating bitcoin ETF flows,

Fed rate decisions, GPU compute costs and open-weight model releases into

distinct topics without supervision.



Live collection requires Reddit API credentials, set through environment

variables as documented above. These figures are from the fixture run and

are not presented as findings about real Reddit discussion.

## Limitations

Topic labels are the top-weighted terms, not human labels, and are worth reviewing before being presented as findings.

VADER handles sarcasm and community-specific idiom poorly, and Reddit has a great deal of both. Sentiment here is directional, not precise.

NMF requires the topic count to be chosen in advance. Ten is a reasonable default for three subreddits; a different corpus would need it retuned, and there is no automatic selection in this pipeline.

Posts are assigned only their single highest-weight topic. A post genuinely spanning two subjects is counted once, which understates overlap between topics.

\---

**Kalyan Katewadi** · [Portfolio](https://kalyankatewadi.github.io) · [LinkedIn](https://www.linkedin.com/in/kalyan-katewadi/)

