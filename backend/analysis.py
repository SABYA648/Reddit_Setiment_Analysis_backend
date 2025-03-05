import os
import threading
import time
import base64
import logging
from io import BytesIO
import datetime

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import praw
import nltk
import pandas as pd
import networkx as nx
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
# Use Render's DATABASE_URL environment variable; fallback to local SQLite for testing.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///reddit_data.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
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
# Initialize database and insert dummy row
# -------------------------
with app.app_context():
    db.create_all()
    # Insert dummy row if table is empty
    if not RedditPost.query.first():
        dummy = RedditPost(
            id="dummy",
            subreddit="dummy",
            author="dummy",
            title="dummy",
            content="dummy",
            upvotes=0,
            comments=0,
            timestamp=0,
            sentiment=0.0,
            search_phrase="dummy"
        )
        db.session.add(dummy)
        db.session.commit()
        logging.info("Inserted dummy row into reddit_posts.")

# -------------------------
# Initialize PRAW
# -------------------------
CLIENT_ID = "je3zlc6OY0kwZ1QBv1saUQ"
CLIENT_SECRET = "WW8cw1B4ghOL6IKRqahr5qFJQcN87w"
USER_AGENT = "RedditGNN/1.0"

try:
    reddit = praw.Reddit(client_id=CLIENT_ID,
                         client_secret=CLIENT_SECRET,
                         user_agent=USER_AGENT)
    reddit.read_only = True
    logging.info("PRAW initialized successfully.")
except Exception as e:
    logging.error(f"PRAW initialization error: {e}")

# -------------------------
# Global Variables for Progress Tracking
# -------------------------
PROGRESS = 0         # Percentage (0-100) based on posts processed
PROCESSING_DONE = False
TOTAL_POSTS = 0      # Total posts fetched

# -------------------------
# Home Route
# -------------------------
@app.route("/")
def home():
    return "Reddit Sentiment Analysis Backend is running!"

# -------------------------
# Function to Process Comments for a Given Post
# -------------------------
def process_comments(post, phrase):
    with app.app_context():
        from sqlalchemy.orm import scoped_session, sessionmaker
        Session = scoped_session(sessionmaker(bind=db.engine))
        session = Session()
        # Verify that the parent post exists
        parent = session.query(RedditPost).get(post.id)
        if not parent:
            logging.warning(f"Parent post {post.id} not found in DB. Skipping comment processing.")
            session.remove()
            return
        try:
            post.comments.replace_more(limit=None)
            for comment in post.comments.list():
                if comment and comment.id:
                    try:
                        # Re-check parent existence (optional)
                        if not session.query(RedditPost).get(post.id):
                            logging.warning(f"Post {post.id} missing when processing comment {comment.id}. Skipping.")
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
                        logging.info(f"Inserted comment: {comment.id} (score: {comment.score})")
                    except Exception as ce:
                        logging.error(f"Error inserting comment {comment.id}: {ce}")
            session.commit()
        except Exception as ce:
            logging.error(f"Error processing comments for post {post.id}: {ce}")
            session.rollback()
        finally:
            session.remove()

# -------------------------
# Fetch and Process Data (Posts and Comments)
# -------------------------
def fetch_and_process_data(phrase, limit=49):
    global PROGRESS, PROCESSING_DONE, TOTAL_POSTS
    with app.app_context():
        PROGRESS = 0
        PROCESSING_DONE = False
        try:
            posts = list(reddit.subreddit("all").search(phrase, limit=limit, sort="relevance"))
        except Exception as e:
            logging.error(f"Error fetching posts: {e}")
            posts = []
        TOTAL_POSTS = len(posts)
        logging.info(f"Found {TOTAL_POSTS} posts for phrase '{phrase}'")
        processed = 0
        comment_threads = []
        for post in posts:
            try:
                if post and post.id:
                    text = post.selftext if post.selftext.strip() != "" else post.title
                    compound = sia.polarity_scores(text)['compound']
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
                    db.session.commit()
                    logging.info(f"Inserted post: {post.id} - {post.title[:50]}...")
                    # Start a separate thread for processing comments
                    t = threading.Thread(target=process_comments, args=(post, phrase))
                    t.start()
                    comment_threads.append(t)
                    processed += 1
                    PROGRESS = int((processed / TOTAL_POSTS) * 100)
                    logging.info(f"Progress (posts processed): {PROGRESS}%")
            except Exception as e:
                db.session.rollback()
                logging.error(f"Error inserting post {post.id if post and post.id else 'unknown'}: {e}")
        for t in comment_threads:
            t.join()
        PROGRESS = 100
        PROCESSING_DONE = True
        logging.info("Processing complete.")

# -------------------------
# Analysis Functions
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
    try:
        conn = db.engine.connect()
        df = pd.read_sql_query("SELECT * FROM reddit_posts WHERE search_phrase = :phrase", conn, params={"phrase": phrase})
        conn.close()
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return {}
    if df.empty:
        logging.info("No data found for analysis.")
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
    try:
        wc = WordCloud(width=800, height=400, background_color='white', max_words=25,
                       stopwords=stop_words, prefer_horizontal=1.0, collocations=False).generate(text_all)
        img_buffer = BytesIO()
        wc.to_image().save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error generating word cloud: {e}")
        img_str = ""
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
def start_process_endpoint():
    global PROGRESS, PROCESSING_DONE
    data = request.get_json()
    phrase = data.get("search_phrase", "").strip()
    if not phrase:
        return jsonify({"error": "search_phrase is required"}), 400
    with app.app_context():
        # Clear the database before starting
        try:
            db.session.query(RedditComment).delete()
            db.session.query(RedditPost).delete()
            db.session.commit()
            logging.info("Cleared database tables.")
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error clearing database: {e}")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)