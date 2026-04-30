import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import InteractiveMindMap from './MindMap';
import './index.css';

function App() {
  // Input State
  const [inputMethod, setInputMethod] = useState("URL");
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [topic, setTopic] = useState('');
  
  // Settings State
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [model, setModel] = useState('llama-3.3-70b-versatile');
  const [summaryLength, setSummaryLength] = useState('Medium');
  const [summaryFormat, setSummaryFormat] = useState('Paragraph');
  const [outputLanguage, setOutputLanguage] = useState('English');
  const [focusArea, setFocusArea] = useState('General');
  const [customFocus, setCustomFocus] = useState('');

  // Execution State
  const [summary, setSummary] = useState('');
  const [context, setContext] = useState('');
  const [loading, setLoading] = useState(false);
  const [mindMapData, setMindMapData] = useState(null);
  const [mindMapLoading, setMindMapLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');
  const [audioLoading, setAudioLoading] = useState(false);

  // History State
  const [history, setHistory] = useState([]);
  const [searchHistory, setSearchHistory] = useState('');
  const [currentHistoryId, setCurrentHistoryId] = useState(null);

  // Chat State
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  
  // Refs
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history?limit=50');
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
        return data;
      }
    } catch (e) {
      console.error("Failed to fetch history:", e);
    }
    return [];
  };

  const clearHistory = async () => {
    try {
      await fetch('/api/history', { method: 'DELETE' });
      setHistory([]);
      alert("History Cleared!");
    } catch (e) {
      console.error("Failed to clear history:", e);
    }
  };

  const loadHistoryItem = (item) => {
    setSummary(item.summary);
    setContext(item.context);
    setMindMapData(null);
    setAudioUrl('');
    setCurrentHistoryId(item.id);
    if (item.chat_history) {
      try {
        setChatMessages(JSON.parse(item.chat_history));
      } catch (e) {
        setChatMessages([]);
      }
    } else {
      setChatMessages([]);
    }
  };

  const handleSummarize = async () => {
    if (inputMethod === "URL" && !url) return alert("⚠️ Please enter a URL.");
    if (inputMethod === "Upload File" && !file) return alert("⚠️ Please upload a file.");
    if (inputMethod === "Topic Research" && !topic) return alert("⚠️ Please enter a topic.");

    setLoading(true);
    setSummary('');
    setMindMapData(null);
    setAudioUrl('');
    setChatMessages([]);
    setCurrentHistoryId(null);
    setContext(''); // In a real scenario, we might want the backend to return context too, but streaming makes it tricky. We'll rely on history for context right now.
    
    try {
      const formData = new FormData();
      formData.append('input_method', inputMethod);
      formData.append('model', model);
      formData.append('summary_length', summaryLength);
      formData.append('summary_format', summaryFormat);
      formData.append('output_language', outputLanguage);
      formData.append('focus_area', focusArea === 'Custom...' ? customFocus : focusArea);

      if (inputMethod === "URL") formData.append('url', url);
      if (inputMethod === "Upload File") formData.append('file', file);
      if (inputMethod === "Topic Research") formData.append('topic', topic);

      const response = await fetch('/api/summarize', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      let fullText = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        setSummary(prev => prev + chunk);
      }
      
      // Fetch history again to update context and history list
      const newHistory = await fetchHistory();
      if (newHistory.length > 0) {
        setCurrentHistoryId(newHistory[0].id);
      }

    } catch (error) {
      console.error("Error summarizing:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleMindMap = async () => {
    if (!summary) return;
    setMindMapLoading(true);
    try {
      const res = await fetch('/api/mindmap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary: summary })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setMindMapData(data);
    } catch (e) {
      alert(`Mind Map Failed: ${e.message}`);
    } finally {
      setMindMapLoading(false);
    }
  };

  const handleAudio = async () => {
    if (!summary) return;
    setAudioLoading(true);
    try {
      const res = await fetch('/api/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: summary })
      });
      if (!res.ok) throw new Error("Audio generation failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (e) {
      alert(`Audio Failed: ${e.message}`);
    } finally {
      setAudioLoading(false);
    }
  };

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([summary], {type: 'text/markdown'});
    element.href = URL.createObjectURL(file);
    element.download = "summary.md";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
    alert("Copied to clipboard!");
  };

  const handleChat = async () => {
    if (!chatInput.trim() || chatLoading) return;
    
    // We need context to chat. The context is usually saved in history.
    // If the user just generated a summary, the newest history item contains the context.
    let currentContext = context;
    if (!currentContext && history.length > 0) {
      currentContext = history[0].context;
      setContext(currentContext);
    }
    
    if (!currentContext) {
      alert("No document context available to chat with.");
      return;
    }

    const query = chatInput;
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: query }]);
    setChatLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context: currentContext, query: query })
      });

      if (!res.ok) throw new Error("Chat failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      
      const initialMessages = [...chatMessages, { role: 'user', content: query }];
      setChatMessages(prev => [...prev, { role: 'assistant', content: '' }]);
      
      let fullAssistantMessage = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullAssistantMessage += chunk;
        
        setChatMessages(prev => {
          const newMessages = [...prev];
          const lastIndex = newMessages.length - 1;
          newMessages[lastIndex] = {
            ...newMessages[lastIndex],
            content: newMessages[lastIndex].content + chunk
          };
          return newMessages;
        });
        
        if (chatEndRef.current) {
           chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }

      if (currentHistoryId) {
        const finalChat = [...initialMessages, { role: 'assistant', content: fullAssistantMessage }];
        fetch('/api/chat/save', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify({ history_id: currentHistoryId, chat_messages: finalChat })
        }).catch(e => console.error("Failed to save chat", e));
      }
    } catch (e) {
      alert(`Chat Error: ${e.message}`);
    } finally {
      setChatLoading(false);
    }
  };

  const handleChatKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleChat();
    }
  };

  // Group History
  const filteredHistory = history.filter(h => h.source.toLowerCase().includes(searchHistory.toLowerCase()) || h.summary.toLowerCase().includes(searchHistory.toLowerCase()));
  const today = new Date().toISOString().split('T')[0];
  const historyToday = filteredHistory.filter(h => h.timestamp.startsWith(today));
  const historyEarlier = filteredHistory.filter(h => !h.timestamp.startsWith(today));

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <div className="sidebar">
        <h2>Settings</h2>
        
        <div className="settings-section">
          <div 
            className="settings-header" 
            onClick={() => setSettingsOpen(!settingsOpen)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', padding: '8px 0' }}
          >
            <h3 style={{ margin: 0, fontSize: '1rem', color: '#FAFAFA' }}>Advanced Settings</h3>
            <span style={{ fontSize: '0.8rem', color: '#888' }}>{settingsOpen ? '▼' : '▶'}</span>
          </div>
          
          {settingsOpen && (
            <div className="settings-content" style={{ marginTop: '16px' }}>
              <div className="input-group">
                <label>AI Model</label>
                <select value={model} onChange={e => setModel(e.target.value)}>
                  <option value="llama-3.3-70b-versatile">Llama 3.3 70B (Recommended)</option>
                  <option value="llama-3.1-8b-instant">Llama 3.1 8B (Fastest)</option>
                  <option value="mixtral-8x7b-32768">Mixtral 8x7b (Context)</option>
                </select>
              </div>

              <div className="input-group">
                <label>Length</label>
                <select value={summaryLength} onChange={e => setSummaryLength(e.target.value)}>
                  <option value="Short">Short</option>
                  <option value="Medium">Medium</option>
                  <option value="Long">Long</option>
                </select>
              </div>

              <div className="input-group">
                <label>Format</label>
                <select value={summaryFormat} onChange={e => setSummaryFormat(e.target.value)}>
                  <option value="Paragraph">Paragraph</option>
                  <option value="Bullet Points">Bullet Points</option>
                  <option value="ELI5">ELI5</option>
                  <option value="Actionable Insights">Actionable Insights</option>
                </select>
              </div>

              <div className="input-group">
                <label>Language</label>
                <select value={outputLanguage} onChange={e => setOutputLanguage(e.target.value)}>
                  <option value="English">English</option>
                  <option value="Spanish">Spanish</option>
                  <option value="French">French</option>
                  <option value="German">German</option>
                  <option value="Hindi">Hindi</option>
                </select>
              </div>

              <div className="input-group">
                <label>Focus Lens</label>
                <select value={focusArea} onChange={e => setFocusArea(e.target.value)}>
                  <option value="General">General</option>
                  <option value="Technical Specs">Technical Specs</option>
                  <option value="Pricing & Plans">Pricing & Plans</option>
                  <option value="Pros & Cons">Pros & Cons</option>
                  <option value="Custom...">Custom...</option>
                </select>
                {focusArea === "Custom..." && (
                  <input type="text" value={customFocus} onChange={e => setCustomFocus(e.target.value)} placeholder="Enter custom focus..." style={{marginTop: '10px'}}/>
                )}
              </div>
            </div>
          )}
        </div>

        <hr className="divider" />
        
        <div className="history-section">
          <h3>History</h3>
          <input 
            type="text" 
            placeholder="Search History..." 
            value={searchHistory} 
            onChange={e => setSearchHistory(e.target.value)}
            style={{marginBottom: '10px'}}
          />
          
          <div className="history-list">
            {historyToday.length > 0 && <h4>Today</h4>}
            {historyToday.map((item, i) => (
              <button key={i} className="history-item-btn" onClick={() => loadHistoryItem(item)}>
                {item.source.substring(0, 25)}{item.source.length > 25 ? '...' : ''}
              </button>
            ))}
            
            {historyEarlier.length > 0 && <h4>Earlier</h4>}
            {historyEarlier.map((item, i) => (
              <button key={i} className="history-item-btn" onClick={() => loadHistoryItem(item)}>
                {item.source.substring(0, 25)}{item.source.length > 25 ? '...' : ''}
              </button>
            ))}
          </div>

          <button className="secondary-btn clear-btn" onClick={clearHistory}>Clear All</button>
        </div>
      </div>
      
      {/* MAIN CONTENT */}
      <div className="main-content">
        <header>

          <div>
            <h1 className="title">Vellum</h1>
            <p className="subtitle">Premium Universal Summarizer</p>
          </div>
        </header>

        {/* Segmented Control */}
        <div className="segmented-control">
          {["URL", "Upload File", "Topic Research"].map(method => (
            <button 
              key={method}
              className={`segment-btn ${inputMethod === method ? 'active' : ''}`}
              onClick={() => setInputMethod(method)}
            >
              {method}
            </button>
          ))}
        </div>

        {/* Input Section */}
        <div className="input-section">
          {inputMethod === "URL" && (
             <input type="text" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://www.anywebsite.com/anyarticle" className="main-input" />
          )}
          {inputMethod === "Upload File" && (
             <input type="file" onChange={e => setFile(e.target.files[0])} className="main-input" accept=".pdf,.txt,.mp3,.wav,.m4a,.mp4,.mov,.jpg,.jpeg,.png" />
          )}
          {inputMethod === "Topic Research" && (
             <input type="text" value={topic} onChange={e => setTopic(e.target.value)} placeholder="e.g. Latest advancements in AI Agents" className="main-input" />
          )}
          
          <button onClick={handleSummarize} disabled={loading} className="primary-btn">
            {loading ? "Processing..." : "Summarize"}
          </button>
        </div>

        {/* Summary Result */}
        {summary && (
          <>
            <div className="result-card">
              <h3>Summary</h3>
              <div className="markdown-content">
                <ReactMarkdown>{summary + (loading ? '▌' : '')}</ReactMarkdown>
              </div>
            </div>

            {/* Post Summary Toolbar */}
            {!loading && (
              <div className="toolbar">
                <button onClick={handleMindMap} disabled={mindMapLoading} className="tool-btn">
                  {mindMapLoading ? "Generating Map..." : "Mind Map"}
                </button>
                <button onClick={handleAudio} disabled={audioLoading} className="tool-btn">
                  {audioLoading ? "Generating Audio..." : "Listen"}
                </button>
                <button onClick={handleDownload} className="tool-btn">Download</button>
                <button onClick={handleCopy} className="tool-btn">Copy</button>
              </div>
            )}
            
            {/* Audio Player */}
            {audioUrl && (
              <div className="audio-player">
                <audio controls src={audioUrl} autoPlay></audio>
              </div>
            )}

            {/* Mind Map Renderer */}
            {mindMapData && (
              <div className="mindmap-container">
                <h3>Visual Mind Map</h3>
                <InteractiveMindMap data={mindMapData} />
              </div>
            )}

            {/* Chat Interface */}
            <hr className="divider" style={{margin: '40px 0'}} />
            <div className="chat-interface">
              <h3>Interview Document</h3>
              <div className="chat-window">
                {chatMessages.length === 0 && <p className="chat-placeholder">Ask follow-up questions about this document...</p>}
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`chat-bubble ${msg.role}`}>
                    <ReactMarkdown>{msg.content + (msg.role === 'assistant' && chatLoading && i === chatMessages.length - 1 ? '▌' : '')}</ReactMarkdown>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
              <div className="chat-input-row">
                <input 
                  type="text" 
                  value={chatInput} 
                  onChange={e => setChatInput(e.target.value)} 
                  onKeyPress={handleChatKeyPress}
                  placeholder="Ask a question..."
                  disabled={chatLoading}
                />
                <button onClick={handleChat} disabled={chatLoading} className="primary-btn small">
                  Send
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;
