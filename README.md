# JobCraft AI

A local Python web app that stores your professional profile and uses a multi-agent RAG architecture to generate tailored job application content — powered by Ollama.

## Features

- **Profile Manager** — single source of truth stored in `data/profile.json`
- **RAG pipeline** — profile chunked, embedded locally (`all-MiniLM-L6-v2`), and stored in ChromaDB
- **4 specialized agents:**
  - Google Form Fill
  - LinkedIn Cold DM
  - Referral Email
  - Cold Job Outreach Email
- **Generate → Refine loop** — iterate on any output until it's right
- **100% local option** — point `OLLAMA_BASE_URL` at `http://localhost:11434`

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally **or** an Ollama Cloud API key

Pull the chat model (local Ollama only — embeddings run via sentence-transformers, no API key):

```bash
ollama pull gemma3:4b
```

### 2. Install dependencies

```bash
python -m venv job_filler_env
job_filler_env\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configure

Copy `.env` and set your values:

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` (local) or `https://api.ollama.com` (cloud) |
| `OLLAMA_API_KEY` | Required for cloud; leave as-is for local |
| `OLLAMA_MODEL` | Chat model — on Ollama Cloud use `gemma3:4b`, `ministral-3:3b`, etc. (run `client.list()` to see available) |

### 4. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

## Usage

1. **Fill your profile** at `/profile` and click **Save & Re-index Profile**
2. Navigate to any agent from the sidebar
3. Fill in the job-specific inputs and click **Generate**
4. Use **Refine** to iterate, then **Copy to Clipboard**

## Project Structure

```
app.py                  Flask entry point
agents/                 Agent implementations
rag/                    Embedding, ChromaDB, retrieval
user_profile/           Profile load/save
data/profile.json       Your profile (gitignored)
data/vectorstore/       ChromaDB persistence (gitignored)
templates/              Jinja2 HTML templates
static/style.css        Minimal styling
```

## Architecture

```
Browser → Flask routes → Agent → RAG Retriever → ChromaDB
                              ↓
                         Ollama LLM
```

Profile is chunked into semantic sections (personal info, each job, each project, skills, etc.), embedded locally via sentence-transformers (`all-MiniLM-L6-v2`), and stored in a persistent ChromaDB collection. Each agent builds a retrieval query from its inputs, fetches the top-k relevant chunks, and passes them as context to the Ollama LLM.
