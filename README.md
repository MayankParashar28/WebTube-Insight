# 🪶 Vellum: Technical Architecture & System Design

**Vellum 2.1** is a high-performance orchestration layer designed for context-aware summarization and interactive document analysis. It leverages the **Groq LPU™ Inference Engine** for sub-second LLM latency and **LangChain** for robust data extraction.

---

## 🏗️ Technical Architecture (Vellum v2.1 SOA)

Vellum has evolved into a modular **Service-Oriented Architecture (SOA)**, separating core intelligence from the user interface.

-   **🧠 Core Engine (`core/engine.py`)**: Stateless business logic, content extraction (YouTube, Web, OCR), and RAG orchestration.
-   **⚙️ Backend API (`api/main.py`)**: A high-performance **FastAPI** REST layer that exposes Vellum's capabilities to any client.
-   **🪶 Frontend Client (`app.py`)**: A premium **Streamlit** interface that communicates with the API via asynchronous streams.

### 🔌 REST Infrastructure
-   `POST /summarize/url`: Decoupled URL ingestion and summarization.
-   `POST /summarize/file`: Multi-modal file processing (PDF, Audio, Video, Image).
-   `POST /chat`: Stateful 'Interview Mode' over REST.

### 🛠️ New Execution Model
To run Vellum in its modular state:
```bash
# Start both Backend and Frontend
./start.sh
```
Or run them independently:
```bash
# Terminal 1: API
uvicorn api.main:app --port 8000
# Terminal 2: UI
streamlit run app.py
```

### 3. Data Persistence
-   **Relational Storage**: SQLite3 backend with a normalized `history` schema for auditability and session persistence.
-   **Schema**:
    ```sql
    CREATE TABLE history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT, source TEXT, summary TEXT, context TEXT
    );
    ```

### 4. Interface & Design System
-   **Runtime**: [Streamlit](https://streamlit.io/) for Pythonic reactivity.
-   **UI/UX (Vellum 2.0)**: Custom CSS injection for:
    -   **Glassmorphism**: `backdrop-filter: blur(12px)` and subtle alpha-transparent overlays.
    -   **Typography**: Outfit (Headers) and Inter (Body) font stacks.
    -   **Reactivity**: Real-time token streaming using `llm.stream()` generator patterns.

---

##  Deployment & Environment

###  Containerized (DevContainer)
Pre-configured logic for VS Code Remote Containers:
-   **Base Image**: `python:3.11-slim`
-   **System Binaries**: `ffmpeg` (audio/video processing), `libsqlite3-dev`, `build-essential`.
-   **Port Mapping**: `8501:8501` (TCP)

###  Local Manual Setup
1.  **Dependencies**: `pip install -r requirements.txt`
2.  **Auth**: Vellum looks for `GROQ_API_KEY` in the environment or a `.env` file (TOML format for Streamlit Cloud).
3.  **Bootstrap**: `streamlit run app.py`

---

##  Technical Specifications

| Feature | Implementation |
| :--- | :--- |
| **Summarization** | LangChain `PromptTemplate` + Groq LPU |
| **Streaming** | Python Generator-based chunking |
| **Mind Maps** | DOT Language + Graphviz Rendering |
| **Audio** | gTTS + BytesIO streaming |
| **Search** | DuckDuckGo Search API Wrapper |

---

## 📄 License
MIT License. High-performance inference courtesy of Groq.
