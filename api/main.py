from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import io
from core.engine import VellumEngine

app = FastAPI(title="Vellum API", version="2.1.0")
engine = VellumEngine()

class SummarizeURLRequest(BaseModel):
    url: str
    model: str = "llama-3.3-70b-versatile"
    length: str = "Medium"
    format: str = "Bullet Points"
    language: str = "English"
    focus_area: Optional[str] = ""

class ChatRequest(BaseModel):
    query: str
    context: str
    model: str = "llama-3.3-70b-versatile"

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.1.0"}

@app.post("/summarize/url")
async def summarize_url(request: SummarizeURLRequest, x_api_key: str = Header(...)):
    try:
        content = engine.extract_from_url(request.url)
        if not content:
            raise HTTPException(status_code=400, detail="Could not extract content from URL")
        
        return StreamingResponse(
            engine.get_summary_stream(
                content, x_api_key, request.model, request.length, request.format, request.language, request.focus_area
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/file")
async def summarize_file(
    file: UploadFile = File(...),
    model: str = Form("llama-3.3-70b-versatile"),
    length: str = Form("Medium"),
    format: str = Form("Bullet Points"),
    language: str = Form("English"),
    focus_area: Optional[str] = Form(""),
    x_api_key: str = Header(...)
):
    try:
        content = ""
        file_bytes = await file.read()
        file_ext = file.filename.split(".")[-1].lower()
        
        if file_ext == "pdf":
            content = engine.extract_from_pdf(file_bytes)
        elif file_ext in ["mp3", "wav", "m4a", "mp4", "mov"]:
            content = engine.extract_from_audio_visual(file_bytes, file_ext, x_api_key)
        elif file_ext in ["jpg", "jpeg", "png"]:
            content = engine.extract_from_image(file_bytes, file_ext, x_api_key)
        else:
            content = file_bytes.decode("utf-8")
            
        if not content:
            raise HTTPException(status_code=400, detail="Could not extract content from file")
            
        return StreamingResponse(
            engine.get_summary_stream(
                content, x_api_key, model, length, format, language, focus_area
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research")
async def research_topic(topic: str = Form(...), x_api_key: str = Header(...)):
    citations, content = engine.conduct_research(topic)
    return {"citations": citations, "content": content}

@app.post("/chat")
async def chat_interview(request: ChatRequest, x_api_key: str = Header(...)):
    return StreamingResponse(
        engine.get_chat_stream(request.query, request.context, x_api_key, request.model),
        media_type="text/event-stream"
    )

@app.get("/history")
def get_history(limit: int = 10):
    return engine.get_history(limit)

@app.post("/history/save")
def save_history(source: str, summary: str, context: str):
    engine.save_to_history(source, summary, context)
    return {"status": "saved"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
