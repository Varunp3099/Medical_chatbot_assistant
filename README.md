# 🩺 Medical Chatbot Assistant

An AI-powered medical document assistant that lets users upload medical PDFs and ask natural-language questions — receiving clear, context-based answers grounded strictly in the uploaded documents.

---

## 1. Project Overview

**Medical Chatbot Assistant** is a full-stack AI application that helps users understand medical documents without needing technical or medical expertise.

### What it does
- Users upload medical PDFs (reports, research papers, guidelines)
- The system processes and indexes the documents
- Users can ask questions in plain English
- The assistant returns accurate answers **only from the uploaded content**

### The problem it solves
Medical documents are often:
- Dense and hard to understand
- Time-consuming to read
- Scattered across multiple files

This project turns static PDFs into a searchable, conversational knowledge base.

### Who it’s for
- Patients trying to understand medical reports
- Students or researchers reading medical papers
- Clinicians or analysts reviewing large document sets
- Recruiters evaluating real-world AI + backend skills 😉

---

## 2. Why I Built This

I built this project to demonstrate how modern AI systems are designed **end-to-end**, not just model calls.

### Skills & concepts showcased
- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- API design and backend architecture
- Safe, controlled LLM responses (no hallucinations)
- Clean separation of frontend and backend

### Real-world relevance
This mirrors how AI assistants are actually built in production — especially in **healthcare, legal, and enterprise search** where accuracy and traceability matter.

---

## 3. Key Features

- **📄 PDF Upload & Processing**  
  Upload multiple PDFs and automatically convert them into searchable knowledge.

- **🔍 Semantic Search (Not Keyword Search)**  
  Uses embeddings to understand *meaning*, not just matching words.

- **💬 Conversational Q&A Interface**  
  Ask natural-language questions through a chat UI.

- **🧠 Context-Grounded AI Responses**  
  The assistant answers **only from uploaded documents** — no guessing.

- **🛡️ Safety-Aware Prompting**  
  Explicitly avoids giving medical advice or diagnoses.

- **🧩 Modular Architecture**  
  Easy to extend, test, or swap components.

---

## 4. Tech Stack

### Backend
- **FastAPI** – High-performance API framework with clean routing
- **LangChain** – Orchestrates retrieval + LLM reasoning
- **Groq ** – Fast, high-quality language model inference
- **Pinecone** – Vector database for semantic document search
- **Google Generative AI Embeddings** – Reliable medical-grade embeddings

### Frontend
- **Streamlit** – Rapid, clean UI for demos and reviewers

### Other Tools
- **Python** – Core language
- **Pydantic** – Data validation
- **Logging & Middleware** – Production-style observability

Each tool was chosen to reflect **real production trade-offs**, not just convenience.

---

## 5. Architecture & Design Decisions

### High-level flow
1. User uploads PDFs
2. PDFs are split into chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in Pinecone
5. User asks a question
6. Relevant chunks are retrieved
7. LLM generates an answer using only retrieved context

### Key decisions
- **RAG over fine-tuning**  
  Ensures answers stay grounded and documents can be updated instantly.

- **Chunking strategy (500 tokens + overlap)**  
  Balances context quality and retrieval accuracy.

- **Custom retriever wrapper**  
  Keeps retrieval logic explicit and testable.

- **Strict prompt constraints**  
  Prevents hallucinations and unsafe medical advice.

---

## 6. What This Project Demonstrates

- System design & architecture
- API development
- Retrieval-Augmented Generation (RAG)
- Vector databases & embeddings
- LLM prompt engineering
- Error handling & logging
- Frontend–backend integration
- AI safety and constraint design
- Production-minded coding practices

---

## 7. Getting Started (Quick Setup)

> Designed so reviewers can skim without running — but easy to run if desired.

### Prerequisites
- Python 3.10+
- API keys for:
  - Groq
  - Pinecone
  - Google Generative AI

### Steps
```bash
# Clone repository
git clone <repo-url>
cd Medical_chatbot_assistant

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=...
export PINECONE_API_KEY=...
export GOOGLE_API_KEY=...

# Start backend
uvicorn server.main:app --reload

# Start frontend
streamlit run client/app.py
