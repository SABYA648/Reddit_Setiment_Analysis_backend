import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import base64

st.set_page_config(page_title="Reddit Analysis Dashboard", layout="wide")
st.title("Reddit Analysis Dashboard")
st.write("This dashboard displays sentiment analysis, top subreddits/redditors, trends, and a word cloud for the given topic.")

# Sidebar: Input form
with st.sidebar.form("search_form"):
    search_phrase = st.text_input("Enter a search phrase", "electric")
    submit_button = st.form_submit_button("Analyze")

if submit_button and search_phrase:
    st.info("Processing... Please wait.")
    backend_url = "https://reddit-setiment-analysis.onrender.com"  # Replace with your actual backend URL

    try:
        r = requests.post(f"{backend_url}/start_process", json={"search_phrase": search_phrase})
        if r.status_code != 200:
            st.error(f"Error starting process: {r.text}")
        else:
            st.success("Processing started.")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

    # Poll progress endpoint
    progress_bar = st.progress(0)
    progress_text = st.empty()
    while True:
        time.sleep(5)
        try:
            prog = requests.get(f"{backend_url}/progress").json()
            progress_bar.progress(prog.get("progress", 0))
            progress_text.text(f"Processing: {prog.get('progress', 0)}% complete")
            if prog.get("done"):
                progress_bar.progress(100)
                progress_text.text("Processing complete!")
                break
        except Exception as e:
            st.error(f"Error retrieving progress: {e}")
            break

    # Fetch results once processing is complete
    try:
        res = requests.get(f"{backend_url}/results", params={"search_phrase": search_phrase})
        if res.status_code != 200:
            st.error(f"Error fetching results: {res.text}")
        else:
            results = res.json()

            st.subheader("Sentiment Distribution")
            sentiment = results.get("sentiment_distribution", {})
            if sentiment:
                df_sent = pd.DataFrame(list(sentiment.items()), columns=["Sentiment Category", "Count"])
                fig = px.bar(df_sent, x="Sentiment Category", y="Count", title="Sentiment Distribution")
                st.plotly_chart(fig)
            else:
                st.write("No sentiment data available.")

            st.subheader("Top 5 Subreddits")
            top_subs = results.get("top_subreddits", {})
            if top_subs:
                st.table(pd.DataFrame(list(top_subs.items()), columns=["Subreddit", "Count"]).sort_values("Count", ascending=False))
            else:
                st.write("No subreddit data available.")

            st.subheader("Top 5 Redditors")
            top_redditors = results.get("top_redditors", {})
            if top_redditors:
                st.table(pd.DataFrame(list(top_redditors.items()), columns=["Redditor", "Count"]).sort_values("Count", ascending=False))
            else:
                st.write("No redditor data available.")

            st.subheader("Positive vs Negative Trends Over Time")
            trend = results.get("trend", {})
            if trend:
                df_trend = pd.DataFrame(trend).T.reset_index().rename(columns={"index": "Date"})
                try:
                    df_trend["Date"] = pd.to_datetime(df_trend["Date"])
                except Exception as e:
                    st.write("Date conversion error:", e)
                fig2 = px.line(df_trend, x="Date", y=["Positive", "Negative"], title="Trends Over Time")
                st.plotly_chart(fig2)
            else:
                st.write("No trend data available.")

            st.subheader("Word Cloud")
            wc_base64 = results.get("word_cloud", "")
            if wc_base64:
                img_html = f'<img src="data:image/png;base64,{wc_base64}" alt="Word Cloud" style="max-width:100%; height:auto;">'
                st.markdown(img_html, unsafe_allow_html=True)
            else:
                st.write("No word cloud available.")
    except Exception as e:
        st.error(f"Error fetching results: {e}")