# 🪶 Vellum: Technical Architecture & System Design

**Vellum 2.1** is a high-performance orchestration layer designed for context-aware summarization and interactive document analysis. It leverages the **Groq LPU™ Inference Engine** for sub-second LLM latency and **LangChain** for robust data extraction.

---

##  System Architecture

Vellum is built on a modular four-layer architecture:

### 1. Ingestion Layer
-   **Structured Extraction**: Utilizes `yt-dlp` and `YoutubeLoader` for high-fidelity transcript retrieval.
-   **Unstructured Scraping**: Employs `BeautifulSoup4` and `WebBaseLoader` for semantic content extraction from arbitrary URLs.
-   **Multi-modal Support**: Integrated `pypdf` for document parsing and `yt-dlp` for video metadata sampling.

### 2. Orchestration & RAG-Lite Layer
-   **Prompt Engineering**: Dynamic `PromptTemplate` injection based on user-selected "Focus Lenses".
-   **Context Tracking**: Implements a persistent `st.session_state` context store (sliding window up to 20k tokens) for multi-turn follow-ups.
-   **Inference Engine**: Standardized on `llama-3.3-70b-versatile` for deep reasoning and `llama-3.1-8b-instant` for low-latency summarization tasks.

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
