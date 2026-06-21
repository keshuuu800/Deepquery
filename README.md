# DeepQuery - Wikipedia RAG Chatbot

## Overview

DeepQuery is an AI-powered Retrieval-Augmented Generation (RAG) chatbot that retrieves live information from Wikipedia, generates semantic embeddings, and answers user queries using a Large Language Model (LLM).

The system combines web scraping, vector embeddings, semantic search, and generative AI to provide accurate and context-aware responses from Wikipedia articles.

---

## Features

* Live Wikipedia article scraping
* Semantic search using embeddings
* Retrieval-Augmented Generation (RAG)
* Intelligent question answering
* Ocean-themed immersive user interface
* Response caching for faster retrieval
* OpenRouter/OpenAI compatible LLM integration
* Dynamic article indexing

---

## Technology Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### AI Components

* Sentence Transformers
* Vector Similarity Search
* OpenRouter LLM

### Data Source

* Wikipedia
* BeautifulSoup Web Scraping

---

## Project Structure

```text
wiki-rag-chatbot/
│
├── static/
│   └── index.html
│
├── cache/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── other project modules
```

---

## System Workflow

1. User enters a topic.
2. Wikipedia article is fetched using the Wikipedia API.
3. Article content is scraped and cleaned.
4. Text is divided into semantic chunks.
5. Embeddings are generated using Sentence Transformers.
6. Relevant chunks are retrieved through vector similarity search.
7. Retrieved context is sent to the LLM.
8. The LLM generates the final answer.
9. The response is displayed to the user.

---

## API Endpoints

### Search Article

**POST**

```http
/api/search
```

Request:

```json
{
  "query": "Virat Kohli"
}
```

---

### Ask Question

**POST**

```http
/api/ask
```

Request:

```json
{
  "query": "Virat Kohli",
  "question": "When was Virat Kohli born?"
}
```

---

## Flowchart

```text
+----------------+
| User Query     |
+--------+-------+
         |
         v
+----------------+
| Wikipedia API  |
+--------+-------+
         |
         v
+----------------+
| Web Scraping   |
+--------+-------+
         |
         v
+----------------+
| Text Cleaning  |
+--------+-------+
         |
         v
+----------------+
| Chunking       |
+--------+-------+
         |
         v
+----------------+
| Embeddings     |
+--------+-------+
         |
         v
+----------------+
| Vector Search  |
+--------+-------+
         |
         v
+----------------+
| Top Chunks     |
+--------+-------+
         |
         v
+----------------+
| OpenRouter LLM |
+--------+-------+
         |
         v
+----------------+
| Final Answer   |
+----------------+
```

---

## Future Improvements

* Multi-document RAG
* PDF Upload Support
* Chat History Memory
* ChromaDB Integration
* Hybrid Search
* Source Citation System
* User Authentication

---

## Author

**Keshav Gupta**
B.Tech Information Technology
Maharaja Agrasen Institute of Technology (MAIT), Delhi

---

## License

This project is intended for educational and learning purposes.
## Screenshots

### Homepage

The landing page where users can search any topic and initiate the RAG pipeline.

![Homepage](screenshots/Homepage.png)

---

### Article Retrieval

After selecting a topic, the system fetches and processes the corresponding Wikipedia article.

![Article](screenshots/Article.png)

---

### Question Answering

Users can ask questions related to the loaded article, and the RAG pipeline retrieves relevant context before generating answers.

![Question Answering](screenshots/Question%20answering.png)

---