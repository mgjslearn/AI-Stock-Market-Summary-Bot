#!/usr/bin/env python3
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
import yfinance as yf
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

HF_PROVIDER = "hyperbolic"
HF_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct" 


MAX_HEADLINES = 5     
MAX_PROMPT_CHARS = 25000  

# my helper functions
def get_finance_news(query="stock market", max_headlines=MAX_HEADLINES):
    """this method fetches the most recent finance headlines via NewsAPI; return list of trimmed headlines."""
    if not NEWS_API_KEY:
        return ["No NEWS_API_KEY in environment; skipping news fetch."]
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_headlines,
        "apiKey": NEWS_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = []
        for a in data.get("articles", []):
            published = a.get("publishedAt", "")
            title = a.get("title", "").strip()
            src = a.get("source", {}).get("name", "")
            # produce a short single-line headline
            items.append(f"{published} — {title} ({src})")
        if not items:
            return ["No recent headlines found for query."]
        return items
    except Exception as e:
        return [f"Error fetching news: {e}"]

def get_stock_summary(ticker="AAPL", days=5):
    """this method gets the recent history and compute a compact numeric summary for a stock."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{days}d", interval="1d", auto_adjust=False)
        if hist.empty:
            return f"No history found for {ticker}."
        # compute metrics
        closes = hist["Close"].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in hist.index.to_pydatetime()]
        latest_close = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
        pct_change = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0.0
        five_day_change = (latest_close - closes[0]) / closes[0] * 100 if len(closes) >= 2 and closes[0] != 0 else 0.0
        # small textual trend
        trend = "up" if five_day_change > 0 else ("down" if five_day_change < 0 else "flat")
        # format a compact table-like string for prompt
        rows = [f"{d}: {c:.2f}" for d, c in zip(dates, closes)]
        summary = (
            f"TICKER: {ticker}\n"
            f"Latest close: ${latest_close:.2f}\n"
            f"Change vs prior day: {pct_change:.2f}%\n"
            f"Change over {days}d: {five_day_change:.2f}% ({trend})\n"
            f"Recent closes:\n" + "\n".join(rows)
        )
        return summary
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"

def build_prompt(headlines, stock_summary, ticker="AAPL"):
    """creates a compact prompt for the chat model, trimming if necessary."""
    # prepare news block
    news_block = "\n".join(f"- {h}" for h in headlines)
    # assemble prompt
    prompt = (
        "You are a concise financial assistant. Given the news and stock data, provide a short market summary "
        "(3-6 sentences) covering: overall market tone, notable headlines' likely impact, and the stock-specific "
        f"implications for {ticker}. Give one bullet list of 3 action-oriented takeaways for an investor (high level).\n\n"
        "NEWS:\n"
        f"{news_block}\n\n"
        "STOCK_DATA:\n"
        f"{stock_summary}\n\n"
        "Answer:"
    )
    # crude safety trimming
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
    return prompt

# ---- asking the huggingface chat completion model I chose----
def ask_llm(prompt, system_message="You are a financial assistant that summarizes market trends."):
    if not HF_TOKEN:
        return "No HF_TOKEN found in environment. Put your Hugging Face API key in .env as HF_TOKEN."
    client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
    try:
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.6,
        )
        # extracting content
        choice = completion.choices[0]
        if hasattr(choice, "message"):
            content = choice.message.get("content") or choice.message.get("content") if isinstance(choice.message, dict) else None
        else:
            # fallback (some clients return "text" or dict)
            content = choice.get("message", {}).get("content") if isinstance(choice, dict) else None
        # final fallback: stringify the whole completion
        if not content:
            content = str(completion)
        return content
    except Exception as e:
        return f"Error calling Hugging Face chat API: {e}"

# -main func to run
def main(ticker="AAPL", news_query="stock market"):
    print(f"Fetching news for: {news_query}")
    headlines = get_finance_news(query=news_query)
    print(f"Got {len(headlines)} headlines (showing first {MAX_HEADLINES}):")
    for h in headlines:
        print("  ", h)

    print(f"\nFetching stock data for: {ticker}")
    stock_summary = get_stock_summary(ticker=ticker)
    print(stock_summary)

    prompt = build_prompt(headlines, stock_summary, ticker=ticker)
    print("\nPrompt length (chars):", len(prompt))

    print("\nCalling LLM...")
    llm_answer = ask_llm(prompt)
    print("\nLLM Response:\n", llm_answer)

if __name__ == "__main__":
    # doing an example test run case
    main(ticker="AAPL", news_query="Apple OR AAPL OR stock market")