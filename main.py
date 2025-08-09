import streamlit as st
import yfinance as yf
import requests
from datetime import date, timedelta
from huggingface_hub import InferenceClient
from typing import List, Optional

# --- Config ---
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY") or ""  # Or set in .streamlit/secrets.toml
HF_TOKEN = st.secrets.get("HF_TOKEN") or ""

HF_PROVIDER = "hyperbolic"
HF_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MAX_NEWS = 5
MAX_TOKENS = 400

# --- Hugging Face client (lazy init) ---
@st.cache_resource(show_spinner=False)
def get_hf_client() -> InferenceClient:
    return InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)

# --- News Fetcher ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(ticker: str, max_articles=MAX_NEWS) -> List[dict]:
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])

# --- Stock Data Fetcher ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, start: date, end: date):
    df = yf.download(ticker, start=start, end=end)
    return df

# --- Build AI Prompt ---
def build_prompt(news: List[dict], stock_df, ticker: str) -> str:
    news_lines = []
    for article in news:
        title = article.get("title", "No Title")
        source = article.get("source", {}).get("name", "")
        news_lines.append(f"- {title} ({source})")
    news_text = "\n".join(news_lines) if news_lines else "No relevant news available."

    closes = stock_df['Close'] if not stock_df.empty else []
    if not closes.empty:
        latest_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else latest_close
        pct_change_day = (latest_close - prev_close) / prev_close * 100 if prev_close else 0.0
        pct_change_period = (latest_close - closes[0]) / closes[0] * 100 if len(closes) > 1 else 0.0
        trend = "up" if pct_change_period > 0 else ("down" if pct_change_period < 0 else "flat")
        closes_str = "\n".join([f"{d.strftime('%Y-%m-%d')}: ${c:.2f}" for d, c in zip(stock_df.index.to_pydatetime(), closes)])
        stock_summary = (
            f"TICKER: {ticker}\n"
            f"Latest close: ${latest_close:.2f}\n"
            f"Change vs prior day: {pct_change_day:.2f}%\n"
            f"Change over period: {pct_change_period:.2f}% ({trend})\n"
            f"Recent closes:\n{closes_str}"
        )
    else:
        stock_summary = f"No stock data available for {ticker} in the selected date range."

    prompt = f"""
You are a financial assistant. Summarize the current market sentiment based on the following news and stock data for {ticker}. Provide a concise summary (3-5 sentences) and 3 action-oriented investor takeaways.

NEWS:
{news_text}

STOCK DATA:
{stock_summary}

Summary:
"""
    return prompt.strip()

# --- Call Hugging Face Chat Completion ---
def query_hf(prompt: str) -> Optional[str]:
    if not HF_TOKEN:
        return "Hugging Face API token not set."
    client = get_hf_client()
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial assistant that summarizes market trends."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        content = completion.choices[0].message.get("content")
        return content
    except Exception as e:
        return f"Error calling Hugging Face API: {e}"

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="📈 Finance AI Dashboard", layout="wide")
    st.title("📈 AI-Powered Finance Dashboard")
    st.write(
        "Enter a stock ticker and date range to get the latest news, stock data, and an AI-generated market summary."
    )

    with st.sidebar:
        ticker = st.text_input("Stock Ticker", "AAPL").upper().strip()
        today = date.today()
        default_start = today - timedelta(days=30)
        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)
        st.markdown("---")
        st.caption("Data sources: NewsAPI, Yahoo Finance, Hugging Face")

    if ticker == "":
        st.warning("Please enter a stock ticker symbol.")
        return

    # Fetch data
    with st.spinner("Fetching news..."):
        try:
            news = fetch_news(ticker)
        except Exception as e:
            st.error(f"Failed to fetch news: {e}")
            news = []

    with st.spinner("Fetching stock data..."):
        try:
            stock_df = fetch_stock_data(ticker, start_date, end_date)
        except Exception as e:
            st.error(f"Failed to fetch stock data: {e}")
            stock_df = None

    # Show news headlines
    st.subheader("📰 Latest News Headlines")
    if news:
        for article in news:
            title = article.get("title", "No Title")
            source = article.get("source", {}).get("name", "")
            url = article.get("url", "")
            st.markdown(f"**[{title}]({url})** - *{source}*")
    else:
        st.info("No news found for this ticker.")

    # Show stock chart
    st.subheader(f"📈 Stock Price Chart: {ticker}")
    if stock_df is not None and not stock_df.empty:
        st.line_chart(stock_df["Close"])
    else:
        st.info("No stock price data available for the selected dates.")

    # Generate AI summary button
    if st.button("Generate AI Market Summary"):
        prompt = build_prompt(news, stock_df, ticker)
        with st.spinner("Generating AI summary..."):
            summary = query_hf(prompt)
            if summary:
                st.subheader("🤖 AI Market Summary")
                st.write(summary)
            else:
                st.error("Failed to get AI summary.")

if __name__ == "__main__":
    main()
