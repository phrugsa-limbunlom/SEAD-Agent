import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FaRobot } from "react-icons/fa";
import { Send } from "lucide-react";
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

  const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
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
      formData.append("input_text", input);
      if (pdfFile) {
        formData.append("pdf", pdfFile);
      }

      const res = await axios.post(url, formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const response = res.data;
      const newMessages = [];
      const queuedMessages = [];

      if (response.summary) {
        newMessages.push({ text: response.summary, sender: "bot", streaming: true });
      } else if (response.message) {
        newMessages.push({ text: response.message, sender: "bot", streaming: true });
      }

      setMessages(prev => {
        const updatedMessages = [...prev, ...newMessages];
        queuedMessages.push(...newMessages.map((msg, idx) => ({
          text: msg.text,
          index: prev.length + idx,
          streaming: true
        })));
        setMessageQueue(queuedMessages);
        return updatedMessages;
      });

    } catch (err) {
      console.error(err);
      const errorMessage = {
        text: err.response?.data?.error || "An error occurred.",
        sender: "error",
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      {!hideHeader && (
        <header className="App-header">
          <h1>Arxiv Agent</h1>
        </header>
      )}
      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.sender === "bot" && <FaRobot className="bot-icon" />}
              {index === streamingIndex && message.streaming ? streamingText : message.text}
            </div>
          ))}
          {loading && (
            <div className="message bot">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            disabled={loading}
            className="file-input"
          />
          <div>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask or summarize a paper..."
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
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