from flask import Flask, request, jsonify
import sqlite3
import praw
import nltk
import pandas as pd
import networkx as nx
from collections import Counter
import datetime
import threading
import time
import base64
from io import BytesIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords

# -------------------------
# Download required NLTK data
# -------------------------
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# -------------------------
# Reddit API Credentials (replace with your actual values)
# -------------------------
CLIENT_ID = "je3zlc6OY0kwZ1QBv1saUQ"
CLIENT_SECRET = "WW8cw1B4ghOL6IKRqahr5qFJQcN87w"
USER_AGENT = "RedditGNN/1.0"

reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)
reddit.read_only = True

DATABASE = "reddit_data.db"
app = Flask(__name__)

# Global variables for progress tracking
PROGRESS = 0         # percentage (0-100) based on posts processed
PROCESSING_DONE = False
TOTAL_POSTS = 0      # total posts fetched

# -------------------------
# Initialize and clear database
# -------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT,
            author TEXT,
            title TEXT,
            content TEXT,
            upvotes INTEGER,
            comments INTEGER,
            timestamp INTEGER,
            sentiment REAL,
            search_phrase TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reddit_comments (
            id TEXT PRIMARY KEY,
            post_id TEXT,
            subreddit TEXT,
            author TEXT,
            content TEXT,
            score INTEGER,
            timestamp INTEGER,
            parent_id TEXT,
            search_phrase TEXT,
            FOREIGN KEY (post_id) REFERENCES reddit_posts(id)
        )
    ''')
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reddit_posts")
    cursor.execute("DELETE FROM reddit_comments")
    conn.commit()
    conn.close()

# -------------------------
# Fetch and process data (posts and comments)
# -------------------------
def fetch_and_process_data(phrase, limit=49):
    global PROGRESS, PROCESSING_DONE, TOTAL_POSTS
    PROGRESS = 0
    PROCESSING_DONE = False
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    posts = list(reddit.subreddit("all").search(phrase, limit=limit, sort="relevance"))
    TOTAL_POSTS = len(posts)
    print(f"Found {TOTAL_POSTS} posts for phrase '{phrase}'")
    processed = 0
    for post in posts:
        if post and post.id:
            text = post.selftext if post.selftext.strip() != "" else post.title
            compound = sia.polarity_scores(text)['compound']
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO reddit_posts (id, subreddit, author, title, content, upvotes, comments, timestamp, sentiment, search_phrase)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post.id,
                    post.subreddit.display_name,
                    post.author.name if post.author else "N/A",
                    post.title,
                    post.selftext,
                    post.score,
                    post.num_comments,
                    int(post.created_utc),
                    compound,
                    phrase
                ))
                print(f"Inserted post: {post.id} - {post.title[:50]}...")
            except Exception as e:
                print(f"Error inserting post {post.id}: {e}")
            # Fetch all comments for the post
            try:
                post.comments.replace_more(limit=None)
                comments = post.comments.list()
                for comment in comments:
                    if comment and comment.id:
                        try:
                            cursor.execute('''
                                INSERT OR IGNORE INTO reddit_comments (id, post_id, subreddit, author, content, score, timestamp, parent_id, search_phrase)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                comment.id,
                                post.id,
                                post.subreddit.display_name,
                                comment.author.name if comment.author else "N/A",
                                comment.body,
                                comment.score,
                                int(comment.created_utc),
                                comment.parent_id,
                                phrase
                            ))
                            print(f"   Inserted comment: {comment.id} (score: {comment.score})")
                        except Exception as ce:
                            print(f"Error inserting comment {comment.id}: {ce}")
            except Exception as ce:
                print(f"Error processing comments for post {post.id}: {ce}")
            processed += 1
            PROGRESS = int((processed / TOTAL_POSTS) * 100)
            print(f"Progress: {PROGRESS}%")
    conn.commit()
    conn.close()
    PROCESSING_DONE = True

# -------------------------
# Analysis: compute sentiment distribution, top subreddits/redditors, trends, and word cloud
# -------------------------
def sentiment_category(compound):
    if compound > 0.8:
        return "Very Positive"
    elif compound > 0.4:
        return "Positive"
    elif compound >= -0.4:
        return "Neutral"
    elif compound >= -0.8:
        return "Negative"
    else:
        return "Very Negative"

def analyze_data(phrase):
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM reddit_posts WHERE search_phrase=?", conn, params=(phrase,))
    conn.close()
    if df.empty:
        return {}
    # Classify sentiment into five categories
    df['sentiment_category'] = df['sentiment'].apply(sentiment_category)
    sentiment_counts = df['sentiment_category'].value_counts().to_dict()
    # Top 5 subreddits and top 5 redditors (by post count)
    top_subreddits = df['subreddit'].value_counts().head(5).to_dict()
    top_redditors = df['author'].value_counts().head(5).to_dict()
    # Positive vs Negative trends over time (using posts with sentiment >0.4 and < -0.4)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    pos = df[df['sentiment'] > 0.4].groupby('date').size()
    neg = df[df['sentiment'] < -0.4].groupby('date').size()
    trend = pd.concat([pos, neg], axis=1).fillna(0)
    trend.columns = ["Positive", "Negative"]
    trend_dict = {str(k): {"Positive": int(v["Positive"]), "Negative": int(v["Negative"])} for k, v in trend.to_dict(orient='index').items()}
    # Generate a beautiful word cloud from titles and content (max 25 words)
    text_all = " ".join(df['title'].tolist() + df['content'].tolist()).lower()
    wc = WordCloud(width=800, height=400, background_color='white', max_words=25, 
                   stopwords=stop_words, prefer_horizontal=1.0, collocations=False).generate(text_all)
    img_buffer = BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return {
        "sentiment_distribution": sentiment_counts,
        "top_subreddits": top_subreddits,
        "top_redditors": top_redditors,
        "trend": trend_dict,
        "word_cloud": img_str
    }

# -------------------------
# Flask Endpoints
# -------------------------
@app.route('/start_process', methods=['POST'])
def start_process():
    global PROGRESS, PROCESSING_DONE
    data = request.get_json()
    phrase = data.get("search_phrase", "").strip()
    if not phrase:
        return jsonify({"error": "search_phrase is required"}), 400
    init_db()
    clear_db()
    PROGRESS = 0
    PROCESSING_DONE = False
    thread = threading.Thread(target=fetch_and_process_data, args=(phrase, 49))
    thread.start()
    return jsonify({"message": "Processing started", "search_phrase": phrase})

@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify({"progress": PROGRESS, "done": PROCESSING_DONE})

@app.route('/results', methods=['GET'])
def get_results():
    phrase = request.args.get("search_phrase", "").strip()
    if not phrase:
        return jsonify({"error": "search_phrase parameter is required"}), 400
    if not PROCESSING_DONE:
        return jsonify({"error": "Processing not complete"}), 400
    analysis = analyze_data(phrase)
    return jsonify(analysis)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from Render, default to 5000 if not set
    app.run(host="0.0.0.0", port=port, debug=True)
    