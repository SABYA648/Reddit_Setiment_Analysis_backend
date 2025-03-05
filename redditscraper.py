import praw
import requests
import sqlite3
import time

# Reddit API Credentials
CLIENT_ID = "je3zlc6OY0kwZ1QBv1saUQ"
CLIENT_SECRET = "WW8cw1B4ghOL6IKRqahr5qFJQcN87w"
USER_AGENT = "RedditGNN/1.0"

# Initialize Reddit API
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

# SQLite Setup
conn = sqlite3.connect("reddit_data.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''CREATE TABLE IF NOT EXISTS reddit_posts (
    id TEXT PRIMARY KEY, 
    subreddit TEXT, 
    user TEXT, 
    title TEXT, 
    content TEXT, 
    upvotes INTEGER, 
    comments INTEGER, 
    timestamp INTEGER
)''')

conn.commit()

# Function to fetch posts from a subreddit
def fetch_reddit_posts(subreddit_name, limit=50):
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=limit):  # Get top 50 posts
        cursor.execute("INSERT OR IGNORE INTO reddit_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                       (post.id, subreddit_name, post.author.name, post.title, post.selftext, post.score, post.num_comments, int(post.created_utc)))
    conn.commit()

# Example Usage
fetch_reddit_posts("technology", limit=50)

print("✅ Reddit data fetched & stored in SQLite.")