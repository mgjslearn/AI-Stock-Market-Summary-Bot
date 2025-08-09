# AI Stock Marker Summarizer
## Overview
This project retrieves the latest financial news and real-time stock market data, then uses a Hugging Face large language model (LLM) to summarize trends, highlight market movers, and provide insights for investors.

It is an AI-powered market intelligence tool with two versions:

Initial prototype: A simple Python script (basic_stock_summarizer.py) that fetches news and stock data, sends it to the Hugging Face LLM, and prints a summary in the console. This version is minimal and has no user interface.

Full app with UI: A Streamlit-based interactive web application (main.py) that provides a clean interface for users to input stock tickers, select date ranges, view news headlines and stock charts, and generate AI summaries on demand.

Initial Prototype (basic_stock_summarizer.py)
Fetches financial news from News API using a fixed query.

Retrieves recent stock data for a fixed ticker using Yahoo Finance.

Combines both inputs into a prompt and sends it to a Hugging Face LLM.

Prints the AI-generated market summary in the console.

No graphical or web interface, designed as a simple end-to-end proof of concept.

Full Application with Interface (main.py)
Built using Streamlit for a user-friendly, interactive dashboard.

Sidebar inputs allow users to enter any stock ticker and date range.

## Displays:

Latest finance news headlines (clickable links).

Interactive stock price charts.

On-demand AI-generated market summaries using Hugging Face LLM.

Includes loading spinners, caching for faster repeated queries, and error handling.

Provides a polished, modern user experience suitable for local use or deployment.

## Features (Both Versions)
📊 Real-time stock data from Yahoo Finance

📰 Breaking financial news from News API

🤖 AI-generated summaries via Hugging Face LLM

🔄 Automated pipeline combining data retrieval and AI summarization

## How It Works
Data Collection

get_finance_news() pulls top finance headlines from News API.

get_stock_data() retrieves recent stock prices and trends from Yahoo Finance.

## LLM Integration

Combines news and stock data into a prompt.

Sends the prompt to the Meta LLaMA 3.1 8B Instruct model via Hugging Face Inference API.

## Output

Returns a concise, human-readable market summary.

## Requirements
Python 3.9+

Hugging Face account with API token

News API key

## Dependencies (install with pip install -r requirements.txt):

nginx
Copy
Edit
huggingface_hub
requests
yfinance
python-dotenv
tqdm
pyyaml
streamlit  # For the full app version
Setup
Clone the Repository

bash
Copy
Edit
git clone <repo_url>
cd ai-stock-news-summarizer
Create & Activate a Virtual Environment

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Create .env File with Your API Keys

ini
Copy
Edit
NEWS_API_KEY=your_news_api_key
HF_TOKEN=your_huggingface_api_token
Usage
Run the initial prototype script (no UI):

bash
Copy
Edit
python3 basic_stock_summarizer.py
This fetches data for a fixed ticker/news query and prints the market summary in the console.

Run the full Streamlit app with interface:

bash
Copy
Edit
streamlit run main.py
This launches a web app where you can enter tickers, select dates, view charts/news, and generate AI summaries interactively.

Example Output (from the prototype)
kotlin
Copy
Edit
Fetching latest finance news...
Fetching stock market data...
Generating AI summary...
LLM Response:
The U.S. stock market closed higher today, led by gains in technology...
