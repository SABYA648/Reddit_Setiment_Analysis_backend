import os
import threading
import time
import base64
from io import BytesIO
import datetime
import logging

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import scoped_session, sessionmaker

import praw
import nltk
import pandas as pd
import networkx as nx
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords

# -------------------------
# Setup logging for debugging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Download required NLTK data
# -------------------------
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# -------------------------
# Flask App and PostgreSQL Database Setup
# -------------------------
app = Flask(__name__)
# DATABASE_URL must be set in environment; default to sqlite if not provided.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///reddit_data.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Optionally set engine options (e.g. pool size) here if needed.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30
}
db = SQLAlchemy(app)

# -------------------------
# Define Models
# -------------------------
class RedditPost(db.Model):
    __tablename__ = "reddit_posts"
    id = db.Column(db.String, primary_key=True)
    subreddit = db.Column(db.String, nullable=False)
    author = db.Column(db.String, nullable=False)
    title = db.Column(db.String, nullable=False)
    content = db.Column(db.Text, nullable=False)
    upvotes = db.Column(db.Integer)
    comments = db.Column(db.Integer)
    timestamp = db.Column(db.Integer)
    sentiment = db.Column(db.Float)
    search_phrase = db.Column(db.String, nullable=False)

class RedditComment(db.Model):
    __tablename__ = "reddit_comments"
    id = db.Column(db.String, primary_key=True)
    post_id = db.Column(db.String, db.ForeignKey('reddit_posts.id'), nullable=False)
    subreddit = db.Column(db.String, nullable=False)
    author = db.Column(db.String, nullable=False)
    content = db.Column(db.Text, nullable=False)
    score = db.Column(db.Integer)
    timestamp = db.Column(db.Integer)
    parent_id = db.Column(db.String)
    search_phrase = db.Column(db.String, nullable=False)

# -------------------------
# Create tables and insert a dummy row (with zeros) for initial testing.
# -------------------------
with app.app_context():
    db.create_all()
    # Check if dummy row exists; if not, insert it.
    dummy = RedditPost.query.filter_by(id="dummy").first()
    if not dummy:
        dummy = RedditPost(
            id="dummy",
            subreddit="dummy",
            author="dummy",
            title="dummy",
            content="dummy",
            upvotes=0,
            comments=0,
            timestamp=int(time.time()),
            sentiment=0.0,
            search_phrase="dummy"
        )
        db.session.add(dummy)
        db.session.commit()
        logger.info("Inserted dummy row into reddit_posts.")

# -------------------------
# Initialize PRAW
# -------------------------
try:
    CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "RedditGNN/1.0")
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=USER_AGENT)
    reddit.read_only = True
    logger.info("PRAW initialized successfully.")
except Exception as e:
    logger.error("Error initializing PRAW: %s", e)
    raise

# -------------------------
# Global Variables for Progress Tracking
# -------------------------
PROGRESS = 0         # 0-100 percent
PROCESSING_DONE = False
TOTAL_POSTS = 0      # total posts fetched

# -------------------------
# Helper: sentiment category function
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

# -------------------------
# Function to Process Comments for a Given Post
# This function runs in a separate thread. It wraps all operations in an app context.
# -------------------------
def process_comments(post, phrase):
    with app.app_context():
        SessionFactory = scoped_session(sessionmaker(bind=db.engine))
        session = SessionFactory()
        # Confirm the parent post exists in the DB.
        parent = session.query(RedditPost).get(post.id)
        if not parent:
            logger.warning("Parent post %s not found in DB. Skipping comment processing.", post.id)
            SessionFactory.remove()
            return

        try:
            post.comments.replace_more(limit=None)
            for comment in post.comments.list():
                if comment and comment.id:
                    try:
                        # Double-check parent's existence
                        if not session.query(RedditPost).get(post.id):
                            logger.warning("Post %s missing during comment processing for comment %s. Skipping.", post.id, comment.id)
                            continue
                        rc = RedditComment(
                            id=comment.id,
                            post_id=post.id,
                            subreddit=post.subreddit.display_name,
                            author=comment.author.name if comment.author else "N/A",
                            content=comment.body,
                            score=comment.score,
                            timestamp=int(comment.created_utc),
                            parent_id=comment.parent_id,
                            search_phrase=phrase
                        )
                        session.merge(rc)
                        logger.info("Inserted comment: %s (score: %s)", comment.id, comment.score)
                    except Exception as ce:
                        logger.error("Error inserting comment %s: %s", comment.id, ce)
            session.commit()
        except Exception as ce:
            session.rollback()
            logger.error("Error processing comments for post %s: %s", post.id, ce)
        finally:
            SessionFactory.remove()

# -------------------------
# Function to Fetch and Process Data (Posts and Comments)
# -------------------------
def fetch_and_process_data(phrase, limit=49):
    global PROGRESS, PROCESSING_DONE, TOTAL_POSTS
    with app.app_context():
        PROGRESS = 0
        PROCESSING_DONE = False
        try:
            posts = list(reddit.subreddit("all").search(phrase, limit=limit, sort="relevance"))
        except Exception as e:
            logger.error("Error fetching posts: %s", e)
            return
        TOTAL_POSTS = len(posts)
        logger.info("Found %s posts for phrase '%s'.", TOTAL_POSTS, phrase)
        processed = 0
        comment_threads = []
        # Process each post one-by-one
        for post in posts:
            if post and post.id:
                text = post.selftext if post.selftext.strip() != "" else post.title
                compound = sia.polarity_scores(text)['compound']
                try:
                    rp = RedditPost(
                        id=post.id,
                        subreddit=post.subreddit.display_name,
                        author=post.author.name if post.author else "N/A",
                        title=post.title,
                        content=post.selftext,
                        upvotes=post.score,
                        comments=post.num_comments,
                        timestamp=int(post.created_utc),
                        sentiment=compound,
                        search_phrase=phrase
                    )
                    db.session.merge(rp)
                    # Commit each post immediately so that the foreign key exists for comments.
                    db.session.commit()
                    logger.info("Inserted post: %s - %s", post.id, post.title[:50])
                except Exception as e:
                    db.session.rollback()
                    logger.error("Error inserting post %s: %s", post.id, e)
                # Spawn a thread for comments processing.
                t = threading.Thread(target=process_comments, args=(post, phrase))
                t.start()
                comment_threads.append(t)
                processed += 1
                PROGRESS = int((processed / TOTAL_POSTS) * 100)
                logger.info("Progress (posts processed): %s%%", PROGRESS)
        # Wait for all comment threads to finish.
        for t in comment_threads:
            t.join()
        PROGRESS = 100
        PROCESSING_DONE = True
        logger.info("Processing complete.")

# -------------------------
# Analysis Function
# -------------------------
def analyze_data(phrase):
    try:
        # Use a separate connection from SQLAlchemy’s engine.
        conn = db.engine.connect()
        # Use a text query with SQLAlchemy text() if needed; here pd.read_sql_query works with a raw string.
        df = pd.read_sql_query("SELECT * FROM reddit_posts WHERE search_phrase = :phrase", conn, params={"phrase": phrase})
        conn.close()
    except Exception as e:
        logger.error("Error querying data: %s", e)
        return {}
    if df.empty:
        logger.info("No data found for phrase: %s", phrase)
        return {}
    df['sentiment_category'] = df['sentiment'].apply(sentiment_category)
    sentiment_counts = df['sentiment_category'].value_counts().to_dict()
    top_subreddits = df['subreddit'].value_counts().head(5).to_dict()
    top_redditors = df['author'].value_counts().head(5).to_dict()
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    pos = df[df['sentiment'] > 0.4].groupby('date').size()
    neg = df[df['sentiment'] < -0.4].groupby('date').size()
    trend = pd.concat([pos, neg], axis=1).fillna(0)
    trend.columns = ["Positive", "Negative"]
    trend_dict = {str(k): {"Positive": int(v["Positive"]), "Negative": int(v["Negative"])} for k, v in trend.to_dict(orient='index').items()}
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
# Flask Routes
# -------------------------
@app.route("/")
def home():
    return "Reddit Sentiment Analysis Backend is running!"

@app.route('/start_process', methods=['POST'])
def start_process_route():
    global PROGRESS, PROCESSING_DONE
    data = request.get_json()
    phrase = data.get("search_phrase", "").strip()
    if not phrase:
        return jsonify({"error": "search_phrase is required"}), 400
    # Clear database tables (wrapped in app context)
    with app.app_context():
        try:
            db.session.query(RedditComment).delete()
            db.session.query(RedditPost).delete()
            db.session.commit()
            logger.info("Cleared database tables.")
        except Exception as e:
            db.session.rollback()
            logger.error("Error clearing database: %s", e)
            return jsonify({"error": "Database clear failed"}), 500
    PROGRESS = 0
    PROCESSING_DONE = False
    # Start data processing in a new thread (wrap function call in app context)
    t = threading.Thread(target=fetch_and_process_data, args=(phrase, 49))
    t.start()
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

# -------------------------
# Run the Flask App
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Run on all interfaces so the server is externally accessible.
    app.run(host="0.0.0.0", port=port, debug=True)