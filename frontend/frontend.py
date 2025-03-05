import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
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
    backend_url = "https://reddit-setiment-analysis.onrender.com"  # Ensure your Flask backend is running at this URL

    # Start processing
    start_response = requests.post(f"{backend_url}/start_process", json={"search_phrase": search_phrase})
    if start_response.status_code != 200:
        st.error(f"Error starting process: {start_response.text}")
    else:
        st.success("Processing started.")
    
    # Poll progress endpoint
    progress_bar = st.progress(0)
    progress_text = st.empty()
    while True:
        time.sleep(5)
        prog_response = requests.get(f"{backend_url}/progress")
        if prog_response.status_code == 200:
            prog_data = prog_response.json()
            prog = prog_data.get("progress", 0)
            progress_bar.progress(prog)
            progress_text.text(f"Processing: {prog}% complete")
            if prog_data.get("done"):
                progress_bar.progress(100)
                progress_text.text("Processing complete!")
                break
        else:
            st.error("Error retrieving progress")
            break

    # Once processing is complete, fetch results
    results_response = requests.get(f"{backend_url}/results", params={"search_phrase": search_phrase})
    if results_response.status_code != 200:
        st.error(f"Error fetching results: {results_response.text}")
    else:
        results = results_response.json()
        
        st.subheader("Sentiment Distribution")
        sentiment = results.get("sentiment_distribution", {})
        if sentiment:
            sentiment_df = pd.DataFrame(list(sentiment.items()), columns=["Sentiment Category", "Count"])
            fig_sent = px.bar(sentiment_df, x="Sentiment Category", y="Count", title="Sentiment Distribution")
            st.plotly_chart(fig_sent)
        else:
            st.write("No sentiment distribution data available.")
        
        st.subheader("Top 5 Subreddits")
        top_subs = results.get("top_subreddits", {})
        if top_subs:
            subs_df = pd.DataFrame(list(top_subs.items()), columns=["Subreddit", "Count"]).sort_values(by="Count", ascending=False)
            st.table(subs_df)
        else:
            st.write("No subreddit data available.")
        
        st.subheader("Top 5 Redditors")
        top_redditors = results.get("top_redditors", {})
        if top_redditors:
            redditors_df = pd.DataFrame(list(top_redditors.items()), columns=["Redditor", "Count"]).sort_values(by="Count", ascending=False)
            st.table(redditors_df)
        else:
            st.write("No redditor data available.")
        
        st.subheader("Positive vs Negative Trends Over Time")
        trend = results.get("trend", {})
        if trend:
            trend_df = pd.DataFrame(trend).T.reset_index().rename(columns={"index": "Date"})
            try:
                trend_df["Date"] = pd.to_datetime(trend_df["Date"])
            except Exception as e:
                st.write("Error converting dates:", e)
            fig_trend = px.line(trend_df, x="Date", y=["Positive", "Negative"], title="Positive vs Negative Trends Over Time")
            st.plotly_chart(fig_trend)
        else:
            st.write("No trend data available.")
        
        st.subheader("Word Cloud")
        wc_base64 = results.get("word_cloud", "")
        if wc_base64:
            img_html = f'<img src="data:image/png;base64,{wc_base64}" alt="Word Cloud" style="max-width:100%; height:auto;">'
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            st.write("No word cloud available.")