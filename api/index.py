import os
import time
import ssl
import sqlite3
import pg8000
import base64
import tempfile
import validators
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from groq import Groq

from dotenv import load_dotenv

# SSL setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load .env from parent directory since app is in backend/
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

app = Flask(__name__, static_folder='../dist', static_url_path='/')
CORS(app)

USE_POSTGRES = bool(os.getenv("POSTGRES_URL"))
IS_VERCEL = bool(os.getenv("VERCEL"))

def get_db_path():
    if IS_VERCEL:
        return '/tmp/summarizer_history.db'
    return 'summarizer_history.db'

def get_db_connection():
    if USE_POSTGRES:
        conn = pg8000.connect(dsn=os.getenv("POSTGRES_URL"))
    else:
        conn = sqlite3.connect(get_db_path())
    return conn

def get_param():
    return "%s" if USE_POSTGRES else "?"

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        if USE_POSTGRES:
            c.execute('''CREATE TABLE IF NOT EXISTS history
                         (id SERIAL PRIMARY KEY, 
                          timestamp TEXT, 
                          source TEXT, 
                          summary TEXT, 
                          context TEXT)''')
            c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='history' and column_name='chat_history';")
            if not c.fetchone():
                c.execute("ALTER TABLE history ADD COLUMN chat_history TEXT")
        else:
            c.execute('''CREATE TABLE IF NOT EXISTS history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          timestamp TEXT, 
                          source TEXT, 
                          summary TEXT, 
                          context TEXT)''')
            c.execute("PRAGMA table_info(history)")
            columns = [col[1] for col in c.fetchall()]
            if "chat_history" not in columns:
                c.execute("ALTER TABLE history ADD COLUMN chat_history TEXT")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB init warning: {e}")

init_db()

def save_to_history(source, summary, context):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        p = get_param()
        c.execute(f"INSERT INTO history (timestamp, source, summary, context) VALUES ({p}, {p}, {p}, {p})",
                  (timestamp, source, summary, context))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to save history: {e}")

@app.route('/api/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', 50, type=int)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        p = get_param()
        c.execute(f"SELECT id, timestamp, source, summary, context, chat_history FROM history ORDER BY id DESC LIMIT {p}", (limit,))
        history_items = [{"id": row[0], "timestamp": row[1], "source": row[2], "summary": row[3], "context": row[4], "chat_history": row[5]} for row in c.fetchall()]
        conn.close()
        return jsonify(history_items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_documents_from_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False) 
            return loader.load()
        except Exception:
            try:
                from yt_dlp import YoutubeDL
                with YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    content = f"**VIDEO TITLE**: {info.get('title')}\n\n**VIDEO DESCRIPTION**: {info.get('description')}\n\n(Transcript unavailable)"
                    return [Document(page_content=content)]
            except Exception as dl_err:
                raise Exception(f"Failed to fetch video info: {dl_err}")
    else:
        scrape_success = False
        docs = []
        try:
            loader = WebBaseLoader(
                url,
                header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5'
                },
                verify_ssl=False
            )
            docs = loader.load()
            if docs and docs[0].page_content.strip() and len(docs[0].page_content) > 200:
                scrape_success = True
        except Exception as web_err:
            print(f"Primary scrape failed: {web_err}")
        
        if not scrape_success:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                }
                response = requests.get(url, headers=headers, verify=False, timeout=12)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.string if soup.title else "No Title"
                meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                description = meta_desc['content'] if meta_desc else "No Description"
                
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                
                main_text = soup.get_text(separator=' ', strip=True)
                if len(main_text) > 5000: main_text = main_text[:5000] + "..."
                
                fallback_content = f"**WEBPAGE TITLE**: {title}\n\n**DESCRIPTION**: {description}\n\n**CONTENT EXCERPT**:\n{main_text}"
                docs = [Document(page_content=fallback_content)]
            except Exception as fallback_err:
                raise Exception(f"All scraping attempts failed: {fallback_err}")
        return docs

def get_documents_from_research(topic):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=10, backend="html") 
    results = wrapper.results(topic, max_results=10)
    
    if not results:
        search_run = DuckDuckGoSearchRun(api_wrapper=wrapper)
        search_results_text = search_run.run(topic)
        if not search_results_text or "No results found" in search_results_text:
            raise Exception("No search results found.")
        content = f"**RESEARCH TOPIC**: {topic}\n\n**SEARCH RESULTS**:\n{search_results_text}"
        return [Document(page_content=content)]
    else:
        search_results_text = "\n\n".join([f"--- SOURCE: {r['title']} ---\n{r['snippet']}" for r in results])
        content = f"**RESEARCH TOPIC**: {topic}\n\n**SEARCH RESULTS**:\n{search_results_text}"
        return [Document(page_content=content)]

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.form
    api_key = data.get('api_key') or os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing API Key"}), 400

    input_method = data.get('input_method')
    model = data.get('model', 'llama-3.3-70b-versatile')
    summary_length = data.get('summary_length', 'Medium')
    summary_format = data.get('summary_format', 'Paragraph')
    output_language = data.get('output_language', 'English')
    focus_area = data.get('focus_area', '')
    
    docs = []
    source_name = "Unknown"
    
    try:
        if input_method == "URL":
            url = data.get('url')
            if not url or not validators.url(url):
                return jsonify({"error": "Invalid URL"}), 400
            docs = get_documents_from_url(url)
            source_name = url
            
        elif input_method == "Upload File":
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files['file']
            file_ext = file.filename.split('.')[-1].lower()
            source_name = file.filename
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name
                
            try:
                if file_ext == 'pdf':
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                elif file_ext in ["mp3", "wav", "m4a", "mp4", "mov"]:
                    client = Groq(api_key=api_key)
                    with open(tmp_path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=(tmp_path, f.read()),
                            model="distil-whisper-large-v3-en",
                            response_format="json",
                            language="en",
                            temperature=0.0
                        )
                    docs = [Document(page_content=transcription.text)]
                elif file_ext in ["jpg", "jpeg", "png"]:
                    client = Groq(api_key=api_key)
                    with open(tmp_path, "rb") as f:
                        encoded_image = base64.b64encode(f.read()).decode('utf-8')
                    completion = client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": "Describe this image in detail. If it contains text, charts, or diagrams, explain them clearly."},
                            {"type": "image_url", "image_url": {"url": f"data:image/{file_ext};base64,{encoded_image}"}}
                        ]}],
                        temperature=0.1,
                        max_tokens=1024
                    )
                    description = completion.choices[0].message.content
                    docs = [Document(page_content=f"**IMAGE DESCRIPTION**:\n{description}")]
                else:
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        docs = [Document(page_content=f.read())]
            finally:
                os.remove(tmp_path)
                
        elif input_method == "Topic Research":
            topic = data.get('topic')
            if not topic:
                return jsonify({"error": "No topic provided"}), 400
            docs = get_documents_from_research(topic)
            source_name = f"Research: {topic}"
            
        else:
            return jsonify({"error": "Invalid input method"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not docs or not docs[0].page_content.strip():
        return jsonify({"error": "No content could be extracted."}), 400

    content_to_summarize = docs[0].page_content
    focus_instruction = f"\nCRITICAL FOCUS: The user is specifically interested in: {focus_area}. Prioritize and emphasize information related to this in your summary." if focus_area and focus_area != "General" else ""
    
    prompt_template = f"""
    You are a helpful assistant providing a **{summary_length}** summary in **{summary_format}** format.
    IMPORTANT: Write in **{output_language}**.
    {focus_instruction}
    
    Content:
    {{text}}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    formatted_prompt = prompt.format(text=content_to_summarize)
    
    llm = ChatGroq(api_key=api_key, model=model, max_tokens=None)
    
    def generate():
        full_response = ""
        try:
            for chunk in llm.stream(formatted_prompt):
                full_response += chunk.content
                yield chunk.content
            # After stream completes, save to history
            save_to_history(source_name, full_response, content_to_summarize)
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
            
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/api/mindmap', methods=['POST'])
def mindmap():
    data = request.json
    api_key = data.get('api_key') or os.getenv("GROQ_API_KEY")
    summary = data.get('summary')
    
    if not api_key or not summary:
        return jsonify({"error": "Missing API Key or Summary"}), 400
        
    client = Groq(api_key=api_key)
    mm_prompt = f"""
    Analyze the following summary and generate a structured JSON object representing a mind map.
    The JSON MUST have two keys: "nodes" and "edges".
    Each node in the "nodes" array MUST be an object with:
      - "id": a unique string ID (e.g., "1", "2")
      - "data": an object containing a "label" key with the concept name.
    Each edge in the "edges" array MUST be an object with:
      - "id": a unique string ID (e.g., "e1-2")
      - "source": the ID of the parent node
      - "target": the ID of the child node
    Keep the mind map concise but informative. Focus on the core concepts and their relationships.
    RETURN ONLY VALID JSON.
    
    Summary:
    {summary}
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": mm_prompt}], 
            response_format={"type": "json_object"},
            temperature=0.1
        )
        import json
        mindmap_data = json.loads(completion.choices[0].message.content.strip())
        return jsonify(mindmap_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or os.getenv("GROQ_API_KEY")
    context = data.get('context')
    user_query = data.get('query')
    
    if not api_key or not context or not user_query:
        return jsonify({"error": "Missing required fields"}), 400
        
    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", max_tokens=None)
    chat_prompt = f"""
    You are Vellum AI, a research assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know based on the provided material.
    Keep answers concise and professional.
    
    Context:
    {context[:20000]}
    
    Question: {user_query}
    
    Answer:
    """
    
    def generate():
        try:
            for chunk in llm.stream(chat_prompt):
                yield chunk.content
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
            
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/api/chat/save', methods=['POST'])
def save_chat():
    data = request.json
    history_id = data.get('history_id')
    chat_messages = data.get('chat_messages')
    
    if not history_id or chat_messages is None:
        return jsonify({"error": "Missing history_id or chat_messages"}), 400
        
    import json
    chat_history_str = json.dumps(chat_messages)
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        p = get_param()
        c.execute(f"UPDATE history SET chat_history = {p} WHERE id = {p}", (chat_history_str, history_id))
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en')
        
        # We'll save it to a temporary file, then read it into memory and return it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tts.save(tmp_audio.name)
            tmp_path = tmp_audio.name
            
        def generate():
            with open(tmp_path, "rb") as f:
                yield from f
            os.remove(tmp_path)
            
        return Response(generate(), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
