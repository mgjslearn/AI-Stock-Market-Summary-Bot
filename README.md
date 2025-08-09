## Overview
This project retrieves the latest financial news and real-time stock market data, then uses a Hugging Face large language model (LLM) to summarize trends, highlight market movers, and provide insights for investors.

It’s an end-to-end AI-powered market intelligence tool:

Fetch financial news from the News API

Retrieve stock data from Yahoo Finance

Send both to a Hugging Face LLM

Get a concise market summary in natural language

### Features
📊 Real-time stock data from Yahoo Finance

📰 Breaking financial news from News API

🤖 AI-generated summaries via Hugging Face LLM

🔄 Fully automated — one command runs everything end-to-end

### How It Works
Data Collection

get_finance_news() → Pulls top finance headlines using News API

get_stock_data() → Retrieves real-time prices, percentage changes, and trends for selected tickers via Yahoo Finance

LLM Integration

Combines news + stock data into a single text prompt

Sends prompt to Meta LLaMA 3.1 8B Instruct model via Hugging Face Inference API

Output

The model returns a concise, human-readable market summary

### Requirements
Python 3.9+

Hugging Face account with API token

News API key

Dependencies (install with pip install -r requirements.txt):

txt
Copy
Edit
huggingface_hub
requests
yfinance
python-dotenv
tqdm
pyyaml
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
Create .env File

env
Copy
Edit
NEWS_API_KEY=your_news_api_key
HF_TOKEN=your_huggingface_api_token
Usage
Run the script:

bash
Copy
Edit
python3 main.py
Expected output:

kotlin
Copy
Edit
Fetching latest finance news...
Fetching stock market data...
Generating AI summary...
LLM Response:
The U.S. stock market closed higher today, led by gains in technology...
Example Output
Input:

News: "Dow gains as investors await inflation report"

Stock data:

bash
Copy
Edit
AAPL: $192.22 (+0.55%)
TSLA: $239.18 (-0.90%)
MSFT: $423.35 (+1.12%)
AI Summary:

The market saw broad gains today, particularly in the tech sector, with Microsoft leading major indices higher. Tesla declined amid concerns over slowing demand, while Apple saw modest gains ahead of its product launch. Investors remain focused on tomorrow’s inflation report, which could influence Federal Reserve policy.

