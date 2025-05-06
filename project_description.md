# Project Description: Time-LLM + RAG for Financial Forecasting

This project is a modified and extended version of the [Time-LLM](https://arxiv.org/abs/2306.00927) open-source repository.  
It was developed as part of a final project for the course **CS554 - Data Intensive Computing Systems** at **Illinois Institute of Technology** (Spring 2025).

## Purpose

The goal of this project is to explore the feasibility of integrating **Retrieval-Augmented Generation (RAG)** techniques into the **Time-LLM** forecasting architecture. The core motivation is to investigate whether time-aligned financial news, when used as prompt-based context, can improve cryptocurrency (Bitcoin) price forecasting.

## Key Modifications from Original Repository

- 🔍 **News Retrieval Module**:
  - Introduced a semantic retrieval pipeline using Sentence-BERT (`all-MiniLM-L6-v2`) and FAISS for efficient similarity search.
  - Retrieves top-k relevant news articles based on statistical prompts and time window filtering.

- 🧠 **Prompt Generation**:
  - Prompts are dynamically generated from input time-series statistics (min, max, median, trend, lags) to guide both retrieval and forecasting.

- 🔄 **Forecast Integration**:
  - Retrieved news is formatted as natural language and inserted into the LLM input, alongside time-series patch embeddings.

- 🧪 **Evaluation**:
  - Performed empirical tests comparing `Time-LLM (baseline)` vs `Time-LLM + RAG (proposed)` using MAE and MSE metrics on Bitcoin price data.

## Dataset

- `BTCPrice_H.csv` — hourly Bitcoin price data (open, high, low, close, volume) from 2021–2023
- `CryptoNews.json` — news headlines, content, sentiment, and timestamps from CryptoNews (2021–2023)

## License

The base code is under the original license provided by the Time-LLM authors.  
All modifications and additions in this repository are for academic and research purposes only.

## Acknowledgements

- Original Time-LLM Authors: [arXiv:2306.00927](https://arxiv.org/abs/2306.00927)
- Sentence-BERT and FAISS: used for semantic retrieval
- ChatGPT-4o: used to assist with polishing and revision
