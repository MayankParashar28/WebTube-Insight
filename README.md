# 🪶 Vellum: Technical Architecture & System Design

**Vellum 2.1** is a high-performance orchestration layer designed for context-aware summarization and interactive document analysis. It leverages the **Groq LPU™ Inference Engine** for sub-second LLM latency and **LangChain** for robust data extraction. It features a fully decoupled architecture with a React frontend and a Flask backend API.

---

##  System Architecture

Vellum is built on a decoupled full-stack architecture:

### 1. Ingestion Layer (Backend)
-   **Structured Extraction**: Utilizes `yt-dlp` and `YoutubeLoader` for high-fidelity transcript retrieval.
-   **Unstructured Scraping**: Employs `BeautifulSoup4` and `WebBaseLoader` for semantic content extraction from arbitrary URLs.
-   **Multi-modal Support**: Integrated `pypdf` for document parsing, vision models for images, and whisper for audio/video transcription.

### 2. Orchestration & RAG-Lite Layer (Backend)
-   **Prompt Engineering**: Dynamic `PromptTemplate` injection based on user-selected "Focus Lenses".
-   **Context Tracking**: Maintains context through React component state and SQLite history for multi-turn chat interactions.
-   **Inference Engine**: Standardized on `llama-3.3-70b-versatile` for deep reasoning and `llama-3.1-8b-instant` for low-latency summarization tasks.

### 3. Data Persistence (Backend)
-   **Relational Storage**: SQLite3 backend with a normalized `history` schema for auditability and session persistence.
-   **Schema**:
    ```sql
    CREATE TABLE history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT, source TEXT, summary TEXT, context TEXT
    );
    ```

### 4. Interface & Design System (Frontend)
-   **Runtime**: React 19 / Vite with Node.js Express server.
-   **UI/UX (Vellum 2.1)**: Custom CSS injection for:
    -   **Glassmorphism**: `backdrop-filter: blur(12px)` and subtle alpha-transparent overlays.
    -   **Typography**: Outfit (Headers) and Inter (Body) font stacks.
    -   **Reactivity**: Real-time token streaming using `ReadableStream` decoder for real-time chunk rendering.

---

##  Deployment & Environment

###  Containerized (DevContainer)
Pre-configured logic for VS Code Remote Containers:
-   **Base Image**: `python:3.11-slim`
-   **System Binaries**: `ffmpeg` (audio/video processing), `libsqlite3-dev`, `build-essential`.

###  Local Manual Setup
1.  **Backend Dependencies**: `cd backend && pip install -r requirements.txt`
2.  **Frontend Dependencies**: `cd frontend && npm install`
3.  **Auth**: Vellum looks for `GROQ_API_KEY` in the environment or a `.env` file.
4.  **Bootstrap Backend**: `cd backend && python app.py` (Runs on 5001)
5.  **Bootstrap Frontend**: `cd frontend && npm run dev` (Runs on 5173/3000)

---

##  Technical Specifications

| Feature | Implementation |
| :--- | :--- |
| **Summarization** | LangChain `PromptTemplate` + Groq LPU |
| **Streaming** | Flask generators + React `ReadableStream` |
| **Mind Maps** | DOT Language + Graphviz Rendering |
| **Audio** | gTTS + Blob streaming |
| **Search** | DuckDuckGo Search API Wrapper |

---

## 📄 License
MIT License. High-performance inference courtesy of Groq.
