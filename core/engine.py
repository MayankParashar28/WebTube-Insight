import os
import time
import sqlite3
import re
import validators
import requests
import tempfile
import base64
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL

class VellumEngine:
    def __init__(self, db_path='summarizer_history.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      timestamp TEXT, 
                      source TEXT, 
                      summary TEXT, 
                      context TEXT)''')
        conn.commit()
        conn.close()

    def save_to_history(self, source, summary, context):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute("INSERT INTO history (timestamp, source, summary, context) VALUES (?, ?, ?, ?)",
                  (timestamp, source, summary, context))
        conn.commit()
        conn.close()

    def get_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT timestamp, source, summary, context FROM history ORDER BY id DESC LIMIT ?", (limit,))
        history_items = [{"timestamp": row[0], "source": row[1], "summary": row[2], "context": row[3]} for row in c.fetchall()]
        conn.close()
        return history_items

    def get_all_history_raw(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT timestamp, source, summary FROM history ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        return rows

    def clear_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()

    def extract_from_url(self, url):
        if "youtube.com" in url or "youtu.be" in url:
            try:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False) 
                docs = loader.load()
                return docs[0].page_content
            except Exception:
                with YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return f"**VIDEO TITLE**: {info.get('title')}\n\n**VIDEO DESCRIPTION**: {info.get('description')}\n\n(Transcript unavailable)"
        else:
            # Web scraping logic
            try:
                loader = WebBaseLoader(
                    url,
                    header_template={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    },
                    verify_ssl=False
                )
                docs = loader.load()
                if docs and docs[0].page_content.strip() and len(docs[0].page_content) > 200:
                    return docs[0].page_content
            except:
                pass
            
            # Fallback
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, verify=False, timeout=12)
            soup = BeautifulSoup(response.text, 'html.parser')
            for s in soup(["script", "style"]): s.decompose()
            main_text = soup.get_text(separator=' ', strip=True)
            return main_text[:5000]

    def extract_from_pdf(self, file_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def extract_from_audio_visual(self, file_bytes, file_ext, api_key):
        from groq import Groq
        client = Groq(api_key=api_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_path, file.read()),
                    model="distil-whisper-large-v3-en"
                )
            return transcription.text
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def extract_from_image(self, file_bytes, file_ext, api_key):
        from groq import Groq
        client = Groq(api_key=api_key)
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail. Explain any charts or text."},
                {"type": "image_url", "image_url": {"url": f"data:image/{file_ext};base64,{encoded_image}"}}
            ]}]
        )
        return completion.choices[0].message.content

    def conduct_research(self, topic):
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
        results = wrapper.results(topic, max_results=10)
        if not results:
            return None, "No results found."
        
        search_text = "\n\n".join([f"--- SOURCE: {r['title']} ---\n{r['snippet']}" for r in results])
        citations = "\n\n### 🔗 Sources Found:\n" + "\n".join([f"- [{r['title']}]({r['link']})" for r in results])
        return citations, f"**RESEARCH TOPIC**: {topic}\n\n**SEARCH RESULTS**:\n{search_text}"

    def get_summary_stream(self, text, api_key, model, length, format, language, focus_area=""):
        llm = ChatGroq(api_key=api_key, model=model)
        focus_instr = f"\nCRITICAL FOCUS: {focus_area}" if focus_area else ""
        
        prompt_template = f"""
        You are a helpful assistant providing a **{length}** summary in **{format}** format.
        IMPORTANT: Write in **{language}**.
        {focus_instr}
        
        Content:
        {{text}}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        formatted_prompt = prompt.format(text=text)
        
        for chunk in llm.stream(formatted_prompt):
            yield chunk.content

    def get_chat_stream(self, query, context, api_key, model):
        llm = ChatGroq(api_key=api_key, model=model)
        chat_prompt = f"""
        You are Vellum AI, a research assistant. Use the following context to answer the question.
        Context:
        {context[:20000]}
        
        Question: {query}
        Answer:
        """
        for chunk in llm.stream(chat_prompt):
            yield chunk.content

    def generate_mind_map(self, summary, api_key):
        from groq import Groq
        client = Groq(api_key=api_key)
        mm_prompt = f"Create professional Graphviz DOT code for a mind map of this summary. RETURN ONLY DOT CODE.\n\nSummary:\n{summary}"
        completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": mm_prompt}])
        return completion.choices[0].message.content.strip().replace("```dot", "").replace("```", "")
