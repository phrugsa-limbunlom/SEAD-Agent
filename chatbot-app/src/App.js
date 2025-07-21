import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FaRobot, FaSearch, FaLightbulb, FaFileAlt, FaPaperclip, FaBook, FaArrowRight, FaExternalLinkAlt, FaUser, FaGlobe, FaQuestionCircle } from "react-icons/fa";
import { Send, X } from "lucide-react";
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

  // New component for displaying sources in Perplexity style
  const SourcesDisplay = ({ sources }) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="sources-grid">
        {sources.map((sourceStr, index) => {
          try {
            const source = JSON.parse(sourceStr);
            return (
              <div key={source.id || index} className="source-card">
                <div className="source-header">
                  <div className="source-icon">
                    {source.type === "ARXIV PAPER" && <FaFileAlt />}
                    {source.type === "DOCUMENT" && <FaBook />}
                    {source.type === "WEB" && <FaGlobe />}
                  </div>
                  <div className="source-info">
                    <div className="source-type">{source.type}</div>
                    <div className="source-number">{index + 1}</div>
                  </div>
                </div>
                <div className="source-content">
                  <h4 className="source-title">{source.title}</h4>
                  {source.authors && (
                    <div className="source-authors">
                      <FaUser className="author-icon" />
                      {source.authors}
                    </div>
                  )}
                  {source.published && (
                    <div className="source-published">
                      Published: {source.published}
                    </div>
                  )}
                </div>
                <div className="source-footer">
                  <a 
                    href={source.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="source-link"
                  >
                    <FaExternalLinkAlt />
                    View Source
                  </a>
                </div>
              </div>
            );
          } catch (e) {
            console.error("Error parsing source:", e);
            return null;
          }
        })}
      </div>
    );
  };

  // New component for displaying tool calls
  const ToolCallsDisplay = ({ toolCalls }) => {
    if (!toolCalls || toolCalls.length === 0) return null;

    return (
      <div className="tool-calls-display">
        <h3>Tools Used</h3>
        <div className="tool-calls-grid">
          {toolCalls.map((call, index) => (
            <div key={index} className="tool-call-card">
              <div className="tool-call-header">
                <div className="tool-call-icon">
                  {call.function_name === "search_arxiv" && <FaSearch />}
                  {call.function_name === "get_design_recommendations" && <FaLightbulb />}
                  {call.function_name === "search_document" && <FaBook />}
                </div>
                <div className="tool-call-info">
                  <div className="tool-call-name">{call.display_name}</div>
                  <div className="tool-call-description">{call.description}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // New component for Perplexity-style response display
  const PerplexityResponse = ({ message, isStreaming, streamingText }) => {
    const [activeTab, setActiveTab] = useState("answer");
    const result = message.result;

    if (!result) return null;

    const displayText = isStreaming ? streamingText : (result.final_response || message.text);
    const sources = result.sources || [];
    const toolCalls = result.tool_calls || [];

    return (
      <div className="perplexity-response">
        {/* Initial Response */}
        {result.initial_response && (
          <div className="initial-response">
            <div className="initial-response-content">
              {result.initial_response}
            </div>
          </div>
        )}

        {/* Tool Calls Display */}
        <ToolCallsDisplay toolCalls={toolCalls} />

        {/* Tab Navigation */}
        <div className="response-tabs">
          <button 
            className={`tab-button ${activeTab === "answer" ? "active" : ""}`}
            onClick={() => setActiveTab("answer")}
          >
            <FaRobot className="tab-icon" />
            Answer
          </button>
          <button 
            className={`tab-button ${activeTab === "sources" ? "active" : ""}`}
            onClick={() => setActiveTab("sources")}
          >
            <FaBook className="tab-icon" />
            Sources
            {sources.length > 0 && <span className="source-count">{sources.length}</span>}
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === "answer" && (
            <div className="answer-content">
              <div 
                className="formatted-response"
                dangerouslySetInnerHTML={{
                  __html: formatMessage(displayText)
                }}
              />
            </div>
          )}
          
          {activeTab === "sources" && (
            <div className="sources-content">
              {sources.length > 0 ? (
                <SourcesDisplay sources={sources} />
              ) : (
                <div className="no-sources">
                  <FaBook className="no-sources-icon" />
                  <p>No sources available for this response.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderWelcomeMessage = () => (
    <div className="welcome-message">
      <div className="welcome-header">
        <FaRobot className="welcome-icon" />
        <h2>Welcome to the AI Research Assistant</h2>
      </div>
      <p className="welcome-description">
        This system specializes in structural engineering and architectural design research. 
        It provides comprehensive support for exploring academic papers, generating design recommendations, and analyzing documents.
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
            <p>Receive evidence-based design suggestions for structural and architectural projects</p>
          </div>
        </div>
        
        <div className="capability-item">
          <FaFileAlt className="capability-icon" />
          <div>
            <h3>Analyze Documents</h3>
            <p>Upload PDF documents for comprehensive analysis and summarization</p>
          </div>
        </div>
        
        <div className="capability-item">
          <FaQuestionCircle className="capability-icon" />
          <div>
            <h3>Q&A Assistance</h3>
            <p>Ask questions about structural engineering concepts, building codes, design standards, and get expert-level answers</p>
          </div>
        </div>
      </div>
      
      <div className="welcome-examples">
        <h3>Example queries:</h3>
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
          <div className="example-query" onClick={() => setInput("What are the key differences between moment frames and braced frames?")}>
            <FaArrowRight className="example-icon" />
            "What are the key differences between moment frames and braced frames?"
          </div>
        </div>
      </div>
    </div>
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() && !pdfFile) return;

    const userMessage = { 
      text: input || (pdfFile ? `Uploaded document: ${pdfFile.name}` : ""), 
      sender: "user",
      hasFile: !!pdfFile,
      fileName: pdfFile ? pdfFile.name : null
    };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput("");
    // Clear the PDF file immediately after adding to chat
    if (pdfFile) {
      setPdfFile(null);
    }
    setLoading(true);
    setHideHeader(true);

    try {
      const formData = new FormData();
      formData.append("message", input || `Please analyze and summarize this document: ${pdfFile.name}`);
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
        if (response.result && response.result.final_response) {
          setMessageQueue([{
            text: response.result.final_response,
            index: prev.length,
            streaming: true
          }]);
        }
        return updatedMessages;
      });

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
    // Enhanced markdown-like formatting
    return message
      .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/^#{4}\s+(.*?)$/gm, '<h4>$1</h4>')
      .replace(/^#{3}\s+(.*?)$/gm, '<h3>$1</h3>')
      .replace(/^#{2}\s+(.*?)$/gm, '<h2>$1</h2>')
      .replace(/^#{1}\s+(.*?)$/gm, '<h1>$1</h1>')
      .replace(/^\s*[-*]\s+(.*)$/gm, '• $1')
      .replace(/^\s*\d+\.\s+(.*)$/gm, '$1')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className="App">
      {!hideHeader && (
        <header className="App-header">
          <h1>AI Research Assistant</h1>
          <p style={{ margin: "0.5rem 0 0 0", color: "#8B949E", fontSize: "0.9rem" }}>
            Intelligent companion for structural engineering & architectural research
          </p>
        </header>
      )}
      <div className="chat-container">
        <div className="messages scrollbar-custom">
          {messages.length === 0 && !loading && renderWelcomeMessage()}
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.sender === "user" && (
                <div className="user-message-container">
                  <div className="user-avatar">
                    <FaUser />
                  </div>
                  <div className="user-message-content">
                    <div className="user-name">You</div>
                    <div className="user-text">{message.text}</div>
                    {message.hasFile && (
                      <div className="file-attachment">
                        <FaFileAlt className="file-icon" />
                        <span className="file-name">{message.fileName}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {message.sender === "bot" && (
                <div className="bot-message-container">
                  <div className="bot-avatar">
                    <FaRobot />
                  </div>
                  <div className="bot-message-content">
                    <div className="bot-name">AI Assistant</div>
                    {message.result && (message.result.sources || message.result.tool_calls) ? (
                      <PerplexityResponse 
                        message={message} 
                        isStreaming={index === streamingIndex && message.streaming}
                        streamingText={streamingText}
                      />
                    ) : (
                      <div className="simple-response">
                        <div 
                          dangerouslySetInnerHTML={{
                            __html: formatMessage(
                              index === streamingIndex && message.streaming 
                                ? streamingText 
                                : message.text
                            )
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}
              {message.sender === "error" && (
                <div className="error-message-container">
                  <div className="error-content">
                    <div className="error-text">{message.text}</div>
                  </div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message bot">
              <div className="bot-message-container">
                <div className="bot-avatar">
                  <FaRobot />
                </div>
                <div className="bot-message-content">
                  <div className="bot-name">AI Assistant</div>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-container">
            <label className="paperclip-button">
              <FaPaperclip />
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                disabled={loading}
                className="file-input"
              />
            </label>
            
            {pdfFile && (
              <div className="file-preview">
                <span className="file-name">{pdfFile.name}</span>
                <button 
                  type="button" 
                  onClick={removeFile}
                  className="remove-file-btn"
                >
                  <X size={16} />
                </button>
              </div>
            )}
            
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Search research papers, request design recommendations, or upload a PDF for analysis..."
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