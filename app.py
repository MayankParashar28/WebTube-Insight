import validators
import streamlit as st
import ssl
import re
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Streamlit UI
# Streamlit UI
st.set_page_config(page_title="Vellum", page_icon="🪶", layout="wide")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }
    
    h1, h2, h3, h4, .stHeader {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(18, 18, 18, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Premium Cards & Containers */
    div.stButton > button {
        border-radius: 8px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        font-weight: 500 !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Segmented Toggle Styling for Radio */
    div[role="radiogroup"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 5px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    div[role="radiogroup"] label {
        padding: 8px 16px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    div[role="radiogroup"] label[data-baseweb="radio"] {
        background: transparent !important;
    }

    /* Soft Glows & Borders */
    .stTextInput > div > div, .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* Summary Result Box */
    .result-card {
        padding: 24px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom Header Section
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("<h1 style='font-size: 3.5rem; margin:0;'>🪶</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0; padding-top: 5px;'>Vellum</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-top: -10px;'>Premium Universal Summarizer</p>", unsafe_allow_html=True)

# Sidebar for API key and Customization
groq_api = os.getenv("GROQ_API_KEY")

# Database
import sqlite3
def init_db():
    conn = sqlite3.connect('summarizer_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  source TEXT, 
                  summary TEXT, 
                  context TEXT)''')
    conn.commit()
    conn.close()

def save_to_history(source, summary, context):
    try:
        conn = sqlite3.connect('summarizer_history.db')
        c = conn.cursor()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute("INSERT INTO history (timestamp, source, summary, context) VALUES (?, ?, ?, ?)",
                  (timestamp, source, summary, context))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def get_history(limit=10):
    history_items = []
    try:
        conn = sqlite3.connect('summarizer_history.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, source, summary, context FROM history ORDER BY id DESC LIMIT ?", (limit,))
        history_items = [{"timestamp": row[0], "source": row[1], "summary": row[2], "context": row[3]} for row in c.fetchall()]
        conn.close()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
    return history_items

def get_all_history():
    try:
        conn = sqlite3.connect('summarizer_history.db')
        import pandas as pd
        df = pd.read_sql_query("SELECT timestamp, source, summary FROM history ORDER BY id DESC", conn)
        conn.close()
        return df
    except:
        return None

# Sidebar
with st.sidebar:
    st.header(" Settings")
    
    # API Key
    env_api_key = os.environ.get("GROQ_API_KEY")
    if env_api_key:
        groq_api_key = env_api_key
        pass
    else:
        groq_api_key = st.text_input("Groq API Key", value=groq_api if groq_api else "", type="password")
        st.caption("Tip: specificy `GROQ_API_KEY` in `.env` to avoid manual entry.")

    st.divider()
    
    with st.expander("⚙️ Advanced Settings"):
      
        st.subheader("Model & Performance")
        model_options = {
            "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B (Recommended)", "desc": "High Intelligence, Balanced Speed", "cost": "~$0.79/M tokens"},
            "llama-3.1-8b-instant": {"name": "Llama 3.1 8B (Fastest)", "desc": "Super Fast, Low Cost, Good for simple tasks", "cost": "~$0.05/M tokens"},
            "mixtral-8x7b-32768": {"name": "Mixtral 8x7b (Context)", "desc": "Long Context, Good Reasoning", "cost": "~$0.24/M tokens"}
        }
        
        selected_model_key = st.selectbox(
            "Choose AI Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x]["name"],
            index=0
        )
        st.caption(f"**{model_options[selected_model_key]['desc']}** | Est. Cost: {model_options[selected_model_key]['cost']}")
        selected_model = selected_model_key 
        
        st.divider()
        st.subheader("Summary Config")
        summary_length = st.select_slider("Length", options=["Short", "Medium", "Long"], value="Medium")
        summary_format = st.radio("Format", options=["Bullet Points", "Paragraph", "ELI5", "Actionable Insights"])
        output_language = st.selectbox("Output Language", options=["English", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese"], index=0)
        
        st.divider()
        st.subheader(" Focus Lens")
        focus_options = [
            "General", 
            "Technical Specs", 
            "Pricing & Plans", 
            "Pros & Cons", 
            "Key Takeaways", 
            "Action Items", 
            "Custom..."
        ]
        focus_choice = st.selectbox("Focus Area", options=focus_options, index=0)
        
        focus_area = ""
        if focus_choice == "Custom...":
            focus_area = st.text_input("Enter custom focus...", placeholder="e.g. key technical specs", key="focus_lens_custom")
        elif focus_choice != "General":
            focus_area = focus_choice
            
        if focus_area:
            st.caption(f"Active: Focusing on *{focus_area}*")
        # Session Management (Moved here for Zen)
        st.divider()
        st.subheader(" Session Quota")
        if "api_calls_count" not in st.session_state:
            st.session_state.api_calls_count = 0
        if "session_start" not in st.session_state:
            import time
            st.session_state.session_start = time.time()
        
        usage_pct = st.session_state.api_calls_count / 100
        st.progress(min(usage_pct, 1.0))
        
        # Simplified Status
        status_msg = "✅ Healthy" if usage_pct <= 0.70 else ("🟡 Warning" if usage_pct <= 0.85 else "🔴 Critical")
        elapsed = time.time() - st.session_state.session_start
        remaining_sec = max(0, (24 * 3600) - elapsed)
        st.caption(f"Status: {status_msg} | Resets in {int(remaining_sec//3600)}h {int((remaining_sec%3600)//60)}m")
        
        st.divider()
        st.subheader("👤 Account Info")
        
        # Initialize Account Metrics
        if "session_id" not in st.session_state:
            import random, string
            st.session_state.session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
        if "start_time_display" not in st.session_state:
            from datetime import datetime
            st.session_state.start_time_display = datetime.now().strftime("%I:%M %p")
            
        # Using columns for cleaner account metrics
        acol1, acol2 = st.columns(2)
        with acol1:
            st.caption("👤 **User**")
            st.caption("🆔 **Session**")
            st.caption("🕒 **Active**")
        with acol2:
            st.caption("Anonymous")
            st.caption(st.session_state.session_id)
            st.caption(st.session_state.start_time_display)
        
        import time
        elapsed_min = int((time.time() - st.session_state.session_start) // 60)
        st.metric("Summaries Made", st.session_state.api_calls_count, delta=None)
        st.caption(f"⏱️ **Time Spent**: {elapsed_min} minutes")

    st.caption("You are using the free tier. You can upgrade to a paid plan to remove the quota limit.")
    
    # History
    st.divider()
    st.header(" History")
    
    search_history = st.text_input("🔍 Search History", placeholder="Filter by topic...", key="hist_search")
    
    # We fetch more than 10 to allow for grouping and filtering, but only display TOP 3 in sidebar
    history_all = get_history(limit=50)
    
    if history_all:
        import pandas as pd
        from datetime import datetime, date

        if search_history:
            history_all = [h for h in history_all if search_history.lower() in h['source'].lower() or search_history.lower() in h['summary'].lower()]

        # Slice for ZEN MODE
        display_history = history_all[:3]
        
        # Grouping Logic
        today = date.today()
        groups = {"Today": [], "Earlier": []}
        
        for item in display_history:
            try:
                item_date = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                if item_date == today:
                    groups["Today"].append(item)
                else:
                    groups["Earlier"].append(item)
            except:
                groups["Earlier"].append(item)

        for group_name, items in groups.items():
            if items:
                with st.expander(f"{group_name} ({len(items)})", expanded=(group_name == "Today")):
                    for i, item in enumerate(items):
                        # Determine Icon
                        source = item['source'].lower()
                        icon = "🌐" if "research" in source else ("📂" if "." in source and len(source.split(".")[-1]) < 5 else "🔗")
                        
                        display_name = item['source'][:25] + "..." if len(item['source']) > 25 else item['source']
                        if st.button(f"{icon} {display_name}", key=f"hist_{group_name}_{i}", use_container_width=True):
                            st.session_state.last_summary = item['summary']
                            st.session_state.context = item['context']
                            st.session_state.messages = []
                            st.session_state.search_results = None
                            st.session_state.last_mind_map = None # Reset mind map on history load
                            st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
             if st.button("👁️ View All", use_container_width=True):
                 st.session_state.show_history_modal = True
        with col2:
             # CSV Export
             export_df = pd.DataFrame(history_all)
             st.download_button("📥 Export", data=export_df.to_csv(index=False), file_name="vellum_history.csv", mime="text/csv", use_container_width=True)

    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        try:
            conn = sqlite3.connect('summarizer_history.db')
            c = conn.cursor()
            c.execute("DELETE FROM history")
            conn.commit()
            conn.close()
            st.toast("History Cleared!")
            st.rerun()
        except: pass

# --- GLOBAL MODALS ---
if st.session_state.get('show_history_modal'):
    with st.expander("📂 FULL HISTORY LOG", expanded=True):
        df = get_all_history()
        if df is not None:
            st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Close Log"):
                st.session_state.show_history_modal = False
                st.rerun()
        else:
            st.write("No history found.")
            

# Input Options: URL, File, or Research
input_method = st.radio("Choose Input Method", ["URL", "Upload File", "Topic Research"], horizontal=True)

generic_url = ""
uploaded_file = None
research_topic = ""

if input_method == "URL":
    generic_url = st.text_input(
        "URL",
        placeholder="https://www.anywebsite.com/anyarticle",
        label_visibility="collapsed",
    )
elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload a document or media", type=["pdf", "txt", "mp3", "wav", "m4a", "mp4", "mov", "jpg", "jpeg", "png"])
elif input_method == "Topic Research":
    research_topic = st.text_input(
        "Enter a Topic",
        placeholder="e.g. Latest advancements in AI Agents",
        label_visibility="collapsed",
    )

# Initialize LLM
if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key, model=selected_model, max_tokens=None)
else:
    llm = None

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "last_mind_map" not in st.session_state:
    st.session_state.last_mind_map = None

# Summarization logic
if st.button("Summarize", type="primary"):
    # 1. Input Validation
    valid_input = True
    if not groq_api_key:
        st.error("⚠️ Please provide a Groq API Key.")
        valid_input = False
    elif input_method == "URL":
        if not generic_url.strip():
             st.error("⚠️ Please enter a URL.")
             valid_input = False
        elif not validators.url(generic_url) or not re.match(r'^https?://', generic_url):
             st.error("⚠️ Invalid URL. Please enter a valid http/https URL.")
             valid_input = False
    elif input_method == "Upload File" and not uploaded_file:
         st.error("⚠️ Please upload a media file or document.")
         valid_input = False
    elif input_method == "Topic Research" and not research_topic.strip():
         st.error("⚠️ Please enter a topic to research.")
         valid_input = False
    
    if valid_input:
        # Rate Limit Check
        if st.session_state.api_calls_count >= 100:
             st.error("🛑 Daily Quota Limit Reached (100/100). Please restart session.")
        else:
            try:
                start_time = time.time()
                # Track Usage
                st.session_state.api_calls_count += 1
                
                # UX: Use st.status for better progress tracking
                with st.status("🚀 Processing...", expanded=True) as status:
                    st.session_state.processing_active = True
                    # Reset state
                    st.session_state.messages = []
                    st.session_state.context = None
                    st.session_state.last_summary = None
                    st.session_state.search_results = None
 
                    
                    docs = []
                    
                    if input_method == "Upload File" and uploaded_file:
                        status.write("📂 Processing file upload...")
                        file_ext = uploaded_file.name.split(".")[-1].lower()
                        
                        if file_ext == "pdf":
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            try:
                                from langchain_community.document_loaders import PyPDFLoader
                                loader = PyPDFLoader(tmp_path)
                                docs = loader.load()
                            finally:
                                os.remove(tmp_path)
                                
                        elif file_ext in ["mp3", "wav", "m4a", "mp4", "mov"]:
                            status.write("🎧 Transcribing audio/video...")
                            import tempfile
                            from groq import Groq
                            client = Groq(api_key=groq_api_key)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            try:
                                with open(tmp_path, "rb") as file:
                                    transcription = client.audio.transcriptions.create(
                                        file=(tmp_path, file.read()),
                                        model="distil-whisper-large-v3-en",
                                        response_format="json",
                                        language="en",
                                        temperature=0.0
                                    )
                                docs = [Document(page_content=transcription.text)]
                            except Exception as audio_err:
                                raise Exception(f"Transcription failed: {audio_err}")
                            finally:
                                os.remove(tmp_path)
                                
                        elif file_ext in ["jpg", "jpeg", "png"]:
                            status.write("👁️ Analyzing image...")
                            import base64
                            from groq import Groq
                            client = Groq(api_key=groq_api_key)
                            image_bytes = uploaded_file.read()
                            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                            try:
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
                            except Exception as vision_err:
                                 raise Exception(f"Vision analysis failed: {vision_err}")
                        else:
                            text = uploaded_file.read().decode("utf-8")
                            docs = [Document(page_content=text)]
                    
                    elif input_method == "Topic Research":
                        status.write(f"🌐 Searching the web for '{research_topic}'...")
                        try:
                            from langchain_community.tools import DuckDuckGoSearchRun
                            from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
                            # Use wrapper to get results with snippets and potentially metadata
                            # Broaden search to get better results for niche topics
                            wrapper = DuckDuckGoSearchAPIWrapper(max_results=10) 
                            results = wrapper.results(research_topic, max_results=10)
                            
                            if not results:
                                 # Fallback to standard run if structured results fail
                                 search_run = DuckDuckGoSearchRun(api_wrapper=wrapper)
                                 search_results_text = search_run.run(research_topic)
                                 if not search_results_text or "No results found" in search_results_text:
                                      st.warning("No search results found.")
                                 else:
                                      content = f"**RESEARCH TOPIC**: {research_topic}\n\n**SEARCH RESULTS**:\n{search_results_text}"
                                      docs = [Document(page_content=content)]
                            else:
                                search_results_text = "\n\n".join([f"--- SOURCE: {r['title']} ---\n{r['snippet']}" for r in results])
                                citations = "\n\n### 🔗 Sources Found:\n" + "\n".join([f"- [{r['title']}]({r['link']})" for r in results])
                                
                                st.session_state.search_results = citations
                                content = f"**RESEARCH TOPIC**: {research_topic}\n\n**SEARCH RESULTS**:\n{search_results_text}"
                                docs = [Document(page_content=content)]
                        except Exception as search_err:
                            raise Exception(f"Research search failed: {search_err}")
    
                    elif input_method == "URL":
                        status.write(f"🔗 Analyzing URL: {generic_url}")
                        
                        if "youtube.com" in generic_url or "youtu.be" in generic_url:
                            try:
                                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False) 
                                docs = loader.load()
                            except Exception:
                                status.write("⚠️ Transcript unavailable. Trying metadata...")
                                try:
                                    from yt_dlp import YoutubeDL
                                    with YoutubeDL({'quiet': True}) as ydl:
                                        info = ydl.extract_info(generic_url, download=False)
                                        content = f"**VIDEO TITLE**: {info.get('title')}\n\n**VIDEO DESCRIPTION**: {info.get('description')}\n\n(Transcript unavailable)"
                                        docs = [Document(page_content=content)]
                                except Exception as dl_err:
                                     raise Exception(f"Failed to fetch video info: {dl_err}")
                        else:
                            scrape_success = False
                            
                            # Attempt 1: WebBaseLoader (Primary)
                            try:
                                status.write("🔍 Attempting standard scrape...")
                                loader = WebBaseLoader(
                                    generic_url,
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
                                status.write(f"⚠️ Primary scrape failed: {str(web_err)[:50]}...")
                            
                            # Attempt 2: SPA/Metadata Fallback (Always if Primary fails/result is poor)
                            if not scrape_success:
                                try:
                                    status.write("🌐 Attempting SPA/Metadata fallback...")
                                    from bs4 import BeautifulSoup
                                    headers = {
                                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                                    }
                                    response = requests.get(generic_url, headers=headers, verify=False, timeout=12)
                                    response.raise_for_status()
                                    soup = BeautifulSoup(response.text, 'html.parser')
                                    
                                    # Extract core metadata
                                    title = soup.title.string if soup.title else "No Title"
                                    meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
                                    description = meta_desc['content'] if meta_desc else "No Description"
                                    
                                    # Attempt to grab main text content if primary failed
                                    # Clean up script and style elements
                                    for script_or_style in soup(["script", "style"]):
                                        script_or_style.decompose()
                                    
                                    # Get text and clean up whitespace
                                    main_text = soup.get_text(separator=' ', strip=True)
                                    if len(main_text) > 5000: main_text = main_text[:5000] + "..."
                                    
                                    fallback_content = f"**WEBPAGE TITLE**: {title}\n\n**DESCRIPTION**: {description}\n\n**CONTENT EXCERPT**:\n{main_text}"
                                    docs = [Document(page_content=fallback_content)]
                                except Exception as fallback_err:
                                    st.warning(f"❌ All scraping attempts failed: {fallback_err}")
    
                    if not docs or not docs[0].page_content.strip():
                        status.update(label="❌ Extraction Failed", state="error")
                        st.error("No content could be extracted. Please try a different URL or refine your topic.")
                    else:
                        status.write("🧠 Generating Summary...")
                        st.session_state.context = docs[0].page_content
                        
                        
                        focus_instruction = f"\nCRITICAL FOCUS: The user is specifically interested in: {focus_area}. Prioritize and emphasize information related to this in your summary." if focus_area else ""
                        
                        prompt_template = f"""
                        You are a helpful assistant providing a **{summary_length}** summary in **{summary_format}** format.
                        IMPORTANT: Write in **{output_language}**.
                        {focus_instruction}
                        
                        Content:
                        {{text}}
                        """
                        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                        
                        try:
                            # Streaming Execution
                            summary_container = st.empty()
                            
                            # Prepare logic for streaming
                            content_to_summarize = docs[0].page_content
                            formatted_prompt = prompt.format(text=content_to_summarize)
                            
                            full_response = ""
                            for chunk in llm.stream(formatted_prompt):
                                full_response += chunk.content
                                # Wrap the stream in the result-card div
                                summary_container.markdown(f"""
                                <div class="result-card">
                                    <h3>📝 Summary</h3>
                                    {full_response}▌
                                </div>
                                """, unsafe_allow_html=True)
                            
                            summary_container.markdown(f"""
                            <div class="result-card">
                                <h3>📝 Summary</h3>
                                {full_response}
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.last_summary = full_response
                            
                            source_name = f"Research: {research_topic}" if input_method == "Topic Research" else (uploaded_file.name if uploaded_file else generic_url)
                            save_to_history(source_name, full_response, docs[0].page_content)
                        except Exception as chain_err:
                            st.error(f"Summarization Failed: {chain_err}")
                            raise chain_err
                        
                        st.session_state.processing_active = False
                        st.session_state.processing_time = time.time() - start_time
                        status.update(label="✅ Summary Ready!", state="complete", expanded=False)
                        
            except requests.exceptions.Timeout:
                st.error("⚠️ Request Timed Out.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Display Logic
if st.session_state.last_summary:
    st.toast("✅ Summary Generated!", icon="✨")
    
    # Metadata Display
    word_count = len(st.session_state.last_summary.split())
    processing_time = st.session_state.get('processing_time', 0)
    st.caption(f"📝 {word_count} words | ⏱️ {processing_time:.2f}s")
    
    # 1. Display Search Results (if available)
    if st.session_state.search_results:
         st.markdown(st.session_state.search_results)

    # 2. Display Extracted Content (Debugger)
    if st.session_state.context:
        with st.expander("View Extracted Content"):
            st.write(st.session_state.context[:1500] + "..." if len(st.session_state.context) > 1500 else st.session_state.context)

    # 3. Display Main Summary
    # The summary is already displayed via streaming during processing, 
    # but we need it here for page reruns (like after mind map click)
    if not st.session_state.get('processing_active', False):
        st.markdown(f"""
        <div class="result-card">
            <h3>📝 Summary</h3>
            {st.session_state.last_summary}
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧠 Mind Map", help="Generate Visual Map"):
            st.session_state.trigger_mm = True

    with col2:
        if st.button("🔊 Listen", help="Text-to-Speech"):
            st.session_state.trigger_audio = True

    with col3:
        st.download_button("💾 Download", data=st.session_state.last_summary, file_name="summary.md", mime="text/markdown", help="Save as Markdown")

    with col4:
        if st.button("📋", help="Copy summary to clipboard"):
            import streamlit.components.v1 as components
            # Avoid backslash in f-string to prevent SyntaxError
            safe_text = st.session_state.last_summary.replace('`', '\\`').replace('$', '\\$')
            copy_html = f"""
            <script>
            const textToCopy = `{safe_text}`;
            navigator.clipboard.writeText(textToCopy).then(() => {{
                console.log('Copied!');
            }});
            </script>
            """
            components.html(copy_html, height=0)
            st.toast("✅ Summary Copied!", icon="📋")

    # --- ACTION TRIGGERS ---
    if st.session_state.get('trigger_mm'):
        st.session_state.trigger_mm = False
        try:
            with st.spinner("Generating Mind Map..."):
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                mm_prompt = f"""
                Create a professional Graphviz DOT code to visualize this summary as an elegant mind map.
                Rules:
                1. Start with 'digraph G {{'
                2. Use 'rankdir=LR; bgcolor="transparent";'
                3. Use modern styling: 'node [shape=rect, style="rounded,filled", fillcolor="#FDF6E3", color="#859900", fontname="Helvetica", penwidth=2];'
                4. Use colored edges: 'edge [color="#93A1A1", penwidth=1.5];'
                5. RETURN ONLY DOT CODE.
                
                Summary:
                {st.session_state.last_summary}
                """
                completion = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": mm_prompt}], temperature=0.1)
                dot_code = completion.choices[0].message.content.strip().replace("```dot", "").replace("```", "")
                st.session_state.last_mind_map = dot_code
        except Exception as mm_err: st.error(f"Mind Map failed: {mm_err}")

    # 4. Show Mind Map (Persistent)
    if st.session_state.last_mind_map:
        st.divider()
        st.subheader("📊 Visual Mind Map")
        st.graphviz_chart(st.session_state.last_mind_map)

    if st.session_state.get('trigger_audio'):
        st.session_state.trigger_audio = False
        try:
            with st.spinner("Generating audio..."):
                from gtts import gTTS
                import tempfile
                tts = gTTS(text=st.session_state.last_summary, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    st.audio(tmp_audio.name, format="audio/mp3")
        except Exception as e: st.error(f"Audio failed: {e}")



# Chat Interface
if st.session_state.context:
    st.divider()
    st.header("💬 Chat with Content")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask something about the content..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            chat_prompt_template = f"""
            Answer the user's question based strictly on the content provided below.
            
            Context:
            {st.session_state.context[:25000]}  # Limit context to avoid token limits if very large
            
            Question: {prompt}
            
            Answer:
            """
            try:
                response_container = st.empty()
                response_text = llm.invoke(chat_prompt_template).content
                response_container.markdown(response_text)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as chat_err:
                st.error(f"Failed to generate answer: {chat_err}")