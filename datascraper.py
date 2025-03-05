import praw
import sqlite3
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import time

# -------------------------------
# DOWNLOAD NLTK DATA (if needed)
# -------------------------------
nltk.download('stopwords')
nltk.download('punkt')

# -------------------------------
# SET UP STOPWORDS
# -------------------------------
stop_words = set(stopwords.words("english"))

# -------------------------------
# REDDIT API CREDENTIALS (Update These)
# -------------------------------
CLIENT_ID = "je3zlc6OY0kwZ1QBv1saUQ"
CLIENT_SECRET = "WW8cw1B4ghOL6IKRqahr5qFJQcN87w"
USER_AGENT = "RedditGNN/1.0"

# -------------------------------
# INITIALIZE PRAW (READ-ONLY)
# -------------------------------
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# -------------------------------
# IMPORTANT: Ensure you are starting with a fresh database.
# Delete any existing "reddit_data.db" file before running this script.
# For example, in the terminal: rm reddit_data.db
# -------------------------------
conn = sqlite3.connect("reddit_data.db")
cursor = conn.cursor()

# -------------------------------
# CREATE TABLES FOR POSTS AND COMMENTS WITH THE NEW SCHEMA
# -------------------------------
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
        awards INTEGER, 
        flair TEXT, 
        url TEXT,
        search_phrase TEXT,
        source TEXT
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS reddit_comments (
        id TEXT PRIMARY KEY,
        post_id TEXT,
        subreddit TEXT,
        author TEXT,
        content TEXT,
        upvotes INTEGER,
        timestamp INTEGER,
        parent_id TEXT,
        search_phrase TEXT,
        FOREIGN KEY (post_id) REFERENCES reddit_posts(id)
    )
''')
conn.commit()

# -------------------------------
# FUNCTION: FETCH REDDIT DATA FROM MULTIPLE ENDPOINTS
# -------------------------------
def fetch_reddit_data(phrase, limit=10):
    """
    Fetch posts and comments for the given phrase from multiple PRAW endpoints.
    The default limit is set to 10 for debugging purposes.
    """
    print(f"\n🔍 Starting Reddit search for phrase: '{phrase}' with limit={limit}...")
    post_count = 0
    comment_count = 0

    # Dictionary mapping source names to PRAW calls
    sources = {
        "SEARCH": reddit.subreddit("all").search(phrase, limit=limit, sort="relevance"),
        "HOT": reddit.subreddit("all").hot(limit=limit),
        "NEW": reddit.subreddit("all").new(limit=limit),
        "RISING": reddit.subreddit("all").rising(limit=limit),
        "TOP": reddit.subreddit("all").top(limit=limit)
    }

    for source_name, source in sources.items():
        print(f"\n--> Processing source: {source_name}")
        try:
            for post in source:
                # Debug: print post id and title if available
                try:
                    print(f"   Processing post: {post.id} - {post.title[:60]}...")
                except Exception as e:
                    print("   Unable to retrieve post id/title.")

                if post and post.id:
                    try:
                        cursor.execute(
                            "INSERT OR IGNORE INTO reddit_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (post.id,
                             post.subreddit.display_name,
                             post.author.name if post.author else "N/A",
                             post.title,
                             post.selftext,
                             post.score,
                             post.num_comments,
                             int(post.created_utc),
                             len(post.all_awardings),
                             post.link_flair_text if post.link_flair_text else "None",
                             post.url,
                             phrase,
                             source_name)
                        )
                        post_count += 1

                        # Process comments for this post
                        # Debug: indicate that we're about to process comments
                        print(f"      Fetching comments for post: {post.id}...")
                        post.comments.replace_more(limit=3)
                        for comment in post.comments.list():
                            if comment and comment.id:
                                cursor.execute(
                                    "INSERT OR IGNORE INTO reddit_comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (comment.id,
                                     post.id,
                                     post.subreddit.display_name,
                                     comment.author.name if comment.author else "N/A",
                                     comment.body,
                                     comment.score,
                                     int(comment.created_utc),
                                     comment.parent_id,
                                     phrase)
                                )
                                comment_count += 1
                    except Exception as e:
                        print(f"❌ Error inserting post {post.id}: {e}")
        except Exception as e:
            print(f"⚠️ Error retrieving data from {source_name}: {e}")
        print(f"--> Finished processing source: {source_name}")

    conn.commit()
    print(f"\n✅ Retrieved {post_count} posts and {comment_count} comments for '{phrase}'.")
    return post_count, comment_count

# -------------------------------
# FUNCTION: GENERATE WORD CO-OCCURRENCE MATRIX
# -------------------------------
def generate_word_matrix(search_phrase):
    print(f"\n📊 Generating word co-occurrence matrix for '{search_phrase}'...")

    # Fetch text data from posts and comments
    cursor.execute("SELECT title, content FROM reddit_posts WHERE search_phrase=?", (search_phrase,))
    post_rows = cursor.fetchall()

    cursor.execute("SELECT content FROM reddit_comments WHERE search_phrase=?", (search_phrase,))
    comment_rows = cursor.fetchall()

    # Combine text from posts and comments
    text_data = " ".join(
        [row[0] + " " + row[1] for row in post_rows if row[0] or row[1]] +
        [row[0] for row in comment_rows if row[0]]
    ).lower()

    if not text_data.strip():
        print("⚠️ No text data available for matrix generation.")
        return

    # Tokenize and filter words
    words = [word for word in nltk.word_tokenize(text_data) if word.isalpha() and word not in stop_words]

    # Count word frequency
    word_freq = Counter(words)
    top_words = word_freq.most_common(20)

    print(f"\n🔹 **Top 20 Co-occurring Words with '{search_phrase}'**:")
    for word, count in top_words:
        print(f"{word}: {count}")

    # Build vocabulary list and ensure the search phrase is included
    vocab = [w[0] for w in top_words]
    if search_phrase not in vocab:
        vocab.append(search_phrase)

    # Create the co-occurrence matrix using CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)
    word_matrix = vectorizer.fit_transform([text_data])
    df = pd.DataFrame(word_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    print("\n📊 Word Co-occurrence Matrix (first 10 rows):")
    print(df.head(10))
    print("\n✅ Word matrix created successfully!")

# -------------------------------
# MAIN PIPELINE EXECUTION
# -------------------------------
if __name__ == "__main__":
    user_input = input("Enter a phrase to search on Reddit: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
    else:
        # For debugging purposes, we start with a lower limit.
        # You can later increase the limit once debugging is complete.
        fetch_reddit_data(user_input, limit=10)
        generate_word_matrix(user_input)

    # Close the database connection
    conn.close()
    print("\n🔚 Database connection closed. Pipeline complete.")