import validators
import streamlit as st
import ssl
import re
import requests
import os
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_URL = os.getenv("VELLUM_API_URL", "http://localhost:8000")

# Streamlit UI Config
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

    /* Soft Glows & Borders */
    .stTextInput > div > div, .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    .result-card {
        padding: 24px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper for API History
def get_history(limit=5):
    try:
        r = requests.get(f"{API_URL}/history?limit={limit}")
        return r.json()
    except: return []

def save_history(source, summary, context):
    try:
        requests.post(f"{API_URL}/history/save", params={"source": source, "summary": summary, "context": context})
    except: pass

# Header
col1, col2 = st.columns([1, 6])
with col1: st.markdown("<h1 style='font-size: 3.5rem; margin:0;'>🪶</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin:0; padding-top: 5px;'>Vellum</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-top: -10px;'>Premium Universal Summarizer</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    with st.expander("Advanced Settings"):
        st.subheader("Model & Performance")
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Choose AI Model", options=model_options)
        
        st.divider()
        summary_length = st.select_slider("Length", options=["Short", "Medium", "Long"], value="Medium")
        summary_format = st.radio("Format", options=["Bullet Points", "Paragraph", "ELI5", "Actionable Insights"])
        output_language = st.selectbox("Output Language", options=["English", "Spanish", "French", "German", "Hindi", "Chinese", "Japanese"])
        
        st.divider()
        focus_choice = st.selectbox("Focus Area", options=["General", "Technical Specs", "Pricing", "Pros & Cons", "Custom..."])
        focus_area = st.text_input("Custom Focus") if focus_choice == "Custom..." else (focus_choice if focus_choice != "General" else "")

    # History
    st.divider()
    st.header("History")
    history_items = get_history(limit=10)
    for i, item in enumerate(history_items[:3]):
        display_name = item['source'][:25] + "..." if len(item['source']) > 25 else item['source']
        if st.button(f"🔗 {display_name}", key=f"hist_{i}", use_container_width=True):
            st.session_state.last_summary = item['summary']
            st.session_state.context = item['context']
            st.rerun()

    # About
    st.divider()
    with st.expander("About Vellum"):
        st.markdown("**Vellum v2.1 (SOA)**\nA premium orchestration layer delivered as a Service.")

# Main Input
input_method = st.radio("Choose Input Method", ["URL", "Upload File", "Topic Research"], horizontal=True)
generic_url = st.text_input("URL") if input_method == "URL" else ""
uploaded_file = st.file_uploader("Upload File") if input_method == "Upload File" else None
research_topic = st.text_input("Enter a Topic") if input_method == "Topic Research" else ""

# Session State
if "messages" not in st.session_state: st.session_state.messages = []
if "context" not in st.session_state: st.session_state.context = None
if "last_summary" not in st.session_state: st.session_state.last_summary = None

# Summarize Logic
if st.button("Summarize", type="primary"):
    if not groq_api_key:
        st.error("Please provide a Groq API Key.")
    else:
        with st.status("🚀 Vellum Processing...", expanded=True) as status:
            headers = {"x-api-key": groq_api_key}
            try:
                full_response = ""
                placeholder = st.empty()
                
                if input_method == "URL":
                    payload = {
                        "url": generic_url,
                        "model": selected_model,
                        "length": summary_length,
                        "format": summary_format,
                        "language": output_language,
                        "focus_area": focus_area
                    }
                    with httpx.stream("POST", f"{API_URL}/summarize/url", json=payload, headers=headers, timeout=60.0) as r:
                        for chunk in r.iter_text():
                            full_response += chunk
                            placeholder.markdown(f'<div class="result-card"><h3>📝 Summary</h3>{full_response}▌</div>', unsafe_allow_html=True)
                
                elif input_method == "Upload File" and uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {
                        "model": selected_model, "length": summary_length, "format": summary_format,
                        "language": output_language, "focus_area": focus_area
                    }
                    with httpx.stream("POST", f"{API_URL}/summarize/file", files=files, data=data, headers=headers, timeout=60.0) as r:
                        for chunk in r.iter_text():
                            full_response += chunk
                            placeholder.markdown(f'<div class="result-card"><h3>📝 Summary</h3>{full_response}▌</div>', unsafe_allow_html=True)

                elif input_method == "Topic Research":
                    r = requests.post(f"{API_URL}/research", data={"topic": research_topic}, headers=headers)
                    res = r.json()
                    st.session_state.context = res["content"]
                    # Now summarize the research content
                    # (Simplified for now, could be a chain call)
                    status.update(label="✅ Research Complete!", state="complete")
                    st.rerun()

                if full_response:
                    placeholder.markdown(f'<div class="result-card"><h3>📝 Summary</h3>{full_response}</div>', unsafe_allow_html=True)
                    st.session_state.last_summary = full_response
                    # For simplicity, we'll assume content is stored or needed for chat
                    # We might need an endpoint to get the extracted context too
                    save_history(generic_url or uploaded_file.name, full_response, "Context stored in API")
                
                status.update(label="✅ Summary Ready!", state="complete")
            except Exception as e:
                st.error(f"API Error: {e}")

# Display Results & Chat (Interview)
if st.session_state.last_summary:
    st.markdown(f'<div class="result-card"><h3>📝 Summary</h3>{st.session_state.last_summary}</div>', unsafe_allow_html=True)
    
    # Interview Section
    if st.session_state.context:
        st.divider()
        st.header("💬 Interview Document")
        chat_container = st.container()
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        if query := st.chat_input("Ask a follow-up..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_container:
                with st.chat_message("user"): st.markdown(query)
                with st.chat_message("assistant"):
                    resp = ""
                    p = st.empty()
                    headers = {"x-api-key": groq_api_key}
                    with httpx.stream("POST", f"{API_URL}/chat", json={"query": query, "context": st.session_state.context, "model": selected_model}, headers=headers) as r:
                        for chunk in r.iter_text():
                            resp += chunk
                            p.markdown(resp + "▌")
                    p.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})