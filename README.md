# ⚖️ Legal Risk Analysis Agent

An autonomous, fault-tolerant AI microservice designed to ingest third-party contracts, cross-reference them against internal company policies, and output a structured legal risk assessment.

---

# 📖 The Business Problem

Manual contract review is a massive bottleneck in legal and procurement departments. While basic **"AI wrappers"** exist to summarize text, they fail in enterprise environments because they:

- Lack deterministic routing
- Hallucinate responses
- Cannot strictly adhere to complex JSON schemas

This project was built to solve that by engineering a **LangGraph state machine** that acts as a strict, reliable compliance calculator rather than a creative chatbot.

---

# 🏗 System Architecture

This system relies on a **decoupled microservice architecture**. By separating the client interface from the heavy AI orchestration, the system ensures:

- Scalability
- Timeout resilience
- Deployment flexibility

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| **Frontend Client** | `Streamlit` (Enterprise Dashboard Layout) |
| **Backend API** | `FastAPI` (Asynchronous Python REST API) |
| **Orchestration Engine** | `LangGraph` (Cyclical State Machine) |
| **Vector Database** | `ChromaDB` (Local persistent storage) |
| **Embeddings** | `HuggingFace` (`all-MiniLM-L6-v2`) |
| **LLM Inference** | `Groq API` (`Llama 3.3 70B Versatile`) |

---

# 🚧 Engineering Challenges & Solutions

Building a reliable AI agent requires overcoming strict infrastructure and model constraints. Here are the core technical hurdles faced during development and how they were resolved:

---

## 1️⃣ The N+1 Network Bottleneck

### ❌ The Problem

Initially, the LangGraph node initialized the HuggingFace embedding model for every single chunk of the parsed PDF.

A **6-chunk document** resulted in:

- 6 separate network calls
- 6 model cold-boots
- ~10-minute processing delays

### ✅ The Solution

Wrapped the retriever function in Python's singleton cache pattern:

```python
@lru_cache(maxsize=1)
```

This forced the system to load the model into memory **exactly once per session**, dropping execution time by **over 90%**.

---

## 2️⃣ Output Hallucinations & Schema Failures

### ❌ The Problem

Attempted to use a smaller, faster model (`Llama 3.1 8B`) to speed up inference.

The smaller model failed to respect the strict **Pydantic JSON schema** required by LangChain's structured output parser, resulting in:

- HTTP 400 errors
- Duplicate JSON keys
- Invalid structured outputs

### ✅ The Solution

Reverted to the `Llama 3.3 70B Versatile` model.

To eliminate probabilistic hallucinations, the LLM `temperature` was strictly forced to:

```python
temperature = 0.0
```

This ensured the model behaved as a **deterministic compliance calculator**, yielding identical violation reports for identical contracts.

---

## 3️⃣ API Rate Limiting (HTTP 429)

### ❌ The Problem

Attempting to process all contract chunks concurrently (**Map-Reduce architecture**) instantly triggered:

```http
429 Too Many Requests
```

errors due to strict **Tokens-Per-Minute (TPM)** API limits.

### ✅ The Solution

Designed the LangGraph edges to process document chunks **sequentially**.

While this increased total wait time (handled asynchronously by FastAPI to prevent UI freezing), it acted as a **natural rate limiter**, ensuring:

- Fault-tolerant execution
- Stable inference
- Budget-aware API usage

---

# 🚀 Quick Start (Docker)

The fastest way to test this application is via **Docker Compose**.

This builds both the backend and frontend containers and links them via a virtual network.

---

## 1️⃣ Configure Environment

Clone the repository and set up your environment variables:

```bash
git clone https://github.com/your-username/legal-risk-agent.git

cd legal-risk-agent

cp .env.example .env
```

Edit the `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_actual_key_here
```

---

## 2️⃣ Build and Run

```bash
docker-compose up --build
```

### 🌐 Access Points

| Service | URL |
|---|---|
| **UI Dashboard** | `http://localhost:8501` |
| **API Swagger Docs** | `http://localhost:8000/docs` |

---

# 💻 Manual Developer Setup

If you prefer to run the application natively without Docker:

## 1️⃣ Create and Activate Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```

> On Windows:

```bash
venv\Scripts\activate
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Start the FastAPI Backend (Terminal 1)

```bash
python -m uvicorn src.api.main:api --reload
```

---

## 4️⃣ Start the Streamlit Frontend (Terminal 2)

```bash
python -m streamlit run src.ui.app.py
```

---

# 📂 Repository Structure

```plaintext
legal-risk-agent/
├── data/                  # Persistent ChromaDB storage & local test files
├── src/
│   ├── api/               # FastAPI microservice wrapper
│   ├── agents/            # LangGraph state machine & LLM tools
│   ├── ingestion/         # PDF parsing and vectorization logic
│   └── ui/                # Streamlit dashboard client
├── Dockerfile.api         # Backend container blueprint
├── Dockerfile.ui          # Frontend container blueprint
├── docker-compose.yml     # Multi-container orchestration
└── requirements.txt       # Frozen Python dependencies
```