import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FaRobot, FaSearch, FaLightbulb, FaFileAlt, FaPaperclip, FaBook, FaArrowRight } from "react-icons/fa";
import { Send, Upload, X } from "lucide-react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [hideHeader, setHideHeader] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [streamingIndex, setStreamingIndex] = useState(-1);
  const [streamingText, setStreamingText] = useState("");
  const [messageQueue, setMessageQueue] = useState([]);
  const [pdfFile, setPdfFile] = useState(null);
  const messagesEndRef = useRef(null);

  const backendUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";
  const endpoint = "/api/chat";
  const url = `${backendUrl}${endpoint}`;

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (messageQueue.length > 0 && streamingIndex === -1) {
      const nextMessage = messageQueue[0];
      setStreamingIndex(nextMessage.index);
      streamMessage(nextMessage.text, nextMessage.index);
      setMessageQueue(prev => prev.slice(1));
    }
  }, [messageQueue, streamingIndex]);

  const streamMessage = (message, index) => {
    const words = message.split(" ");
    let currentWord = 0;

    const streamInterval = setInterval(() => {
      if (currentWord <= words.length) {
        setStreamingText(words.slice(0, currentWord).join(" "));
        currentWord++;
      } else {
        clearInterval(streamInterval);
        setStreamingIndex(-1);
        setStreamingText("");
        setMessages(prev => {
          const updated = [...prev];
          updated[index].text = message;
          return updated;
        });
      }
    }, 20);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setPdfFile(e.target.files[0]);
    }
  };

  const removeFile = () => {
    setPdfFile(null);
  };

  const renderFunctionCallResult = (result) => {
    if (!result) return null;

    if (result.function_calls && result.function_calls.length > 0) {
      return (
        <div className="function-call-result">
          {result.function_calls.map((call, index) => (
            <div key={index}>
              <div className="function-call-header">
                {call.function_name === "search_arxiv" && <FaSearch className="function-call-icon" />}
                {call.function_name === "get_design_recommendations" && <FaLightbulb className="function-call-icon" />}
                {call.function_name === "search_document" && <FaBook className="function-call-icon" />}
                <span>{call.description}</span>
              </div>
              {call.function_name === "search_arxiv" && call.result && (
                <div className="search-results">
                  {call.result.map((paper, idx) => (
                    <div key={idx} className="search-result-item">
                      <div className="search-result-title">{paper.title}</div>
                      <div className="search-result-authors">{paper.authors}</div>
                      <div className="search-result-abstract">{paper.abstract}</div>
                    </div>
                  ))}
                </div>
              )}
              {call.function_name === "get_design_recommendations" && call.result && (
                <div className="design-recommendations">
                  {call.result.map((rec, idx) => (
                    <div key={idx} className="recommendation-item">
                      <div className="recommendation-title">{rec.title}</div>
                      <div className="recommendation-content">{rec.content}</div>
                    </div>
                  ))}
                </div>
              )}
              {call.function_name === "search_document" && call.result && (
                <div className="search-results">
                  {call.result.map((doc, idx) => (
                    <div key={idx} className="search-result-item">
                      <div className="search-result-content">{doc.content}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    return null;
  };

  const renderWelcomeMessage = () => (
    <div className="welcome-message">
      <div className="welcome-header">
        <FaRobot className="welcome-icon" />
        <h2>Welcome to Your AI Research Assistant</h2>
      </div>
      <p className="welcome-description">
        I'm specialized in structural engineering and architectural design research. 
        I can help you explore academic papers, get design recommendations, and analyze documents.
      </p>
      
      <div className="welcome-capabilities">
        <div className="capability-item">
          <FaSearch className="capability-icon" />
          <div>
            <h3>Search Research Papers</h3>
            <p>Find and analyze papers from arXiv on structural engineering, architecture, and related fields</p>
          </div>
        </div>
        
        <div className="capability-item">
          <FaLightbulb className="capability-icon" />
          <div>
            <h3>Get Design Recommendations</h3>
            <p>Receive evidence-based design suggestions for your structural and architectural projects</p>
          </div>
        </div>
        
        <div className="capability-item">
          <FaFileAlt className="capability-icon" />
          <div>
            <h3>Analyze Documents</h3>
            <p>Upload PDF documents for comprehensive analysis and summarization</p>
          </div>
        </div>
      </div>
      
      <div className="welcome-examples">
        <h3>Try asking me:</h3>
        <div className="example-queries">
          <div className="example-query" onClick={() => setInput("Find recent papers on seismic design of tall buildings")}>
            <FaArrowRight className="example-icon" />
            "Find recent papers on seismic design of tall buildings"
          </div>
          <div className="example-query" onClick={() => setInput("What are the best practices for foundation design in soft soil?")}>
            <FaArrowRight className="example-icon" />
            "What are the best practices for foundation design in soft soil?"
          </div>
          <div className="example-query" onClick={() => setInput("Recommend sustainable materials for high-rise construction")}>
            <FaArrowRight className="example-icon" />
            "Recommend sustainable materials for high-rise construction"
          </div>
        </div>
      </div>
    </div>
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput("");
    setLoading(true);
    setHideHeader(true);

    try {
      const formData = new FormData();
      formData.append("message", input);
      if (pdfFile) {
        formData.append("document", pdfFile);
      }

      const res = await axios.post(url, formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const response = res.data;
      const newMessage = {
        text: response.message || "No response received",
        sender: "bot",
        streaming: true,
        result: response.result,
        isPdfSummary: !!pdfFile
      };

      setMessages(prev => {
        const updatedMessages = [...prev, newMessage];
        setMessageQueue([{
          text: newMessage.text,
          index: prev.length,
          streaming: true
        }]);
        return updatedMessages;
      });

      // Clear the PDF file after processing
      if (pdfFile) {
        setPdfFile(null);
      }

    } catch (err) {
      console.error(err);
      const errorMessage = {
        text: err.response?.data?.error || "An error occurred while processing your request.",
        sender: "error",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const formatMessage = (message) => {
    // Basic markdown-like formatting
    return message
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className="App">
      {!hideHeader && (
        <header className="App-header">
          <h1>AI Research Assistant</h1>
          <p style={{ margin: "0.5rem 0 0 0", color: "#8B949E", fontSize: "0.9rem" }}>
            Your intelligent companion for structural engineering & architectural research
          </p>
        </header>
      )}
      <div className="chat-container">
        <div className="messages scrollbar-custom">
          {messages.length === 0 && !loading && renderWelcomeMessage()}
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.sender === "bot" && (
                <FaRobot className="bot-icon" />
              )}
              <div className="message-content">
                {message.isPdfSummary && (
                  <div className="summary-badge">
                    <FaFileAlt />
                    Document Summary
                  </div>
                )}
                <div 
                  dangerouslySetInnerHTML={{
                    __html: formatMessage(
                      index === streamingIndex && message.streaming 
                        ? streamingText 
                        : message.text
                    )
                  }}
                />
                {message.result && renderFunctionCallResult(message.result)}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message bot">
              <FaRobot className="bot-icon" />
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="input-form">
          {pdfFile && (
            <div className="file-upload-container">
              <div className="file-input-label file-selected">
                <FaPaperclip />
                <span>PDF uploaded:</span>
                <span className="file-name">{pdfFile.name}</span>
                <button 
                  type="button" 
                  onClick={removeFile}
                  style={{ 
                    background: "none", 
                    border: "none", 
                    color: "#8B949E", 
                    cursor: "pointer",
                    marginLeft: "0.5rem"
                  }}
                >
                  <X size={16} />
                </button>
              </div>
            </div>
          )}
          {!pdfFile && (
            <div className="file-upload-container">
              <label className="file-input-label">
                <Upload size={16} />
                Upload PDF Document
                <input
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="file-input"
                />
              </label>
            </div>
          )}
          <div className="input-container">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about research papers, get design recommendations, or upload a PDF to summarize..."
              disabled={loading}
              className="text-input"
              rows="1"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="send-button"
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;