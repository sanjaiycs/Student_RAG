# RAG System 

A fully containerized Retrieval-Augmented Generation (RAG) system built
using:

-   Ollama (Mistral LLM)
-   Qdrant (Vector Database)
-   LangChain
-   Docker & Docker Compose

This version runs entirely in the terminal with no frontend.

------------------------------------------------------------------------

## Architecture

User Question\
→ CLI Program\
→ LangChain Retriever\
→ Qdrant Vector Database\
→ Relevant Context Chunks\
→ Ollama (Mistral)\
→ Structured Answer

------------------------------------------------------------------------

## Tech Stack

LLM Serving: - Ollama - Mistral Model

Vector Database: - Qdrant

Framework: - LangChain

Containerization: - Docker - Docker Compose

------------------------------------------------------------------------

## Project Structure

. ├── app.py ├── Dockerfile ├── docker-compose.yml ├── requirements.txt
└── doc/ └── Place your PDF files here

------------------------------------------------------------------------

## Setup Instructions

### 1. Clone the Repository

git clone `<your-repo-url>`{=html} cd `<repo-folder>`{=html}

### 2. Add Your PDF

Place your textbook or documents inside the `doc/` folder.

### 3. Build and Start Services

docker compose up --build

### 4. Pull Required Models

docker exec -it ollama ollama pull mistral docker exec -it ollama ollama
pull nomic-embed-text

### 5. Attach to App Container

docker attach rag_app

You can now type questions directly in the terminal. Type `exit` to
quit.

------------------------------------------------------------------------

## Features

-   Fully local RAG system
-   No external APIs
-   Strict context-based answering
-   Vector similarity retrieval
-   Clean Dockerized architecture
-   Terminal-based interaction

------------------------------------------------------------------------

## Performance Notes

-   Optimized for CPU environments
-   Uses Mistral model via Ollama
-   Adjustable chunk size and retrieval parameters
-   Suitable for academic RAG systems

------------------------------------------------------------------------
