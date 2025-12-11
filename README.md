# Reddit Intelligence Engine  
### Real-Time NLP • BERTopic • Sentiment • Trend Detection • Streamlit Dashboard

This project is an end-to-end Reddit Insight Mining System that collects posts from:

- r/stocks  
- r/crypto  
- r/artificial  

It performs NLP preprocessing, topic modeling, sentiment analysis, trend detection, and visualizes all insights through an advanced interactive dashboard.

---

## 🚀 Features

### 1. Automated Reddit Scraping (PRAW)
Scrapes live posts and stores them in SQLite with fields:
- Title  
- Body  
- Score  
- Comments count  
- Timestamp  

### 2. NLP Cleaning Pipeline (spaCy)
- URL removal  
- Tokenization  
- Lemmatization  
- Stopword removal  

### 3. Topic Modeling (BERTopic)
Uses:
- Transformer embeddings  
- UMAP dimensionality reduction  
- HDBSCAN clustering  
Produces:
- Topic IDs  
- Keywords per topic  
- Topic probabilities  

### 4. Sentiment Analysis (Transformers)
Classifies posts into:
- Positive  
- Negative  
- Neutral  

### 5. Trend Detection Engine
Detects spikes using:
- Rolling mean  
- Rolling standard deviation  
- Z-score anomalies  

### 6. Advanced Streamlit Dashboard
Interactive visuals:
- Topic frequency bar chart  
- Sentiment distribution  
- UMAP 2D embedding  
- Intertopic distance map  
- Topic timeline heatmap  
- Topic details viewer  

---

## 🧠 Architecture

