import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { getProductSummary } from "../api";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function ChatBot({ forecastResult, isVisible, onClose, groupInfo }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [isFocused, setIsFocused] = useState(false);
  const [productSummary, setProductSummary] = useState(null);
  const messagesEndRef = useRef(null);

  // Fetch per-product sales ranking once, if a grouping column was used at
  // upload, so the chatbot can answer "which products sell well" with real
  // numbers instead of guessing.
  useEffect(() => {
    if (groupInfo?.group_col) {
      getProductSummary(
        groupInfo.filename,
        groupInfo.date_col,
        groupInfo.target_col,
        groupInfo.group_col,
        50
      )
        .then(setProductSummary)
        .catch((err) => console.error("ChatBot: failed to load product summary", err));
    }
  }, [groupInfo?.group_col]);

  // Initialize session on first load
  useEffect(() => {
    if (!sessionId) {
      const newSessionId = `session_${crypto.randomUUID()}`;
      setSessionId(newSessionId);
      
      // Add welcome message
      setMessages([
        {
          type: "bot",
          content: "👋 Hi! I'm your AI forecasting assistant. Ask me anything about this forecast - model selection, performance, predictions, or recommendations!"
        }
      ]);
    }
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    
    // Add user message to UI
    setMessages(prev => [...prev, { type: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/chat/ask`, {
        question: userMessage,
        session_id: sessionId,
        forecast_data: forecastResult, // Send forecast data with each message
        product_summary: productSummary, // Per-product ranking, if available
      });

      // Add bot response
      setMessages(prev => [...prev, { type: "bot", content: response.data.response }]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { 
        type: "bot", 
        content: "❌ Sorry, I encountered an error. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        type: "bot",
        content: "🔄 Conversation reset. Ask me anything about this forecast!"
      }
    ]);
    axios.post(`${API_BASE}/api/chat/reset/${sessionId}`).catch(console.error);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-4 right-4 w-96 h-[600px] bg-white rounded-lg shadow-2xl flex flex-col border-2 border-blue-200 z-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-t-lg flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🤖</span>
          <div>
            <h3 className="font-bold text-lg">Forecast Assistant</h3>
            <p className="text-xs opacity-90">Powered by AI</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="hover:bg-blue-500 p-2 rounded transition"
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.length === 0 ? (
          <div className="text-center text-gray-400 mt-8">
            <p className="text-4xl mb-2">💬</p>
            <p className="text-sm">Start by asking about the forecast!</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-xs px-4 py-3 rounded-lg ${
                  msg.type === "user"
                    ? "bg-blue-500 text-white rounded-br-none"
                    : "bg-white text-gray-800 border border-gray-200 rounded-bl-none"
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 px-4 py-3 rounded-lg rounded-bl-none">
              <div className="flex gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: "0.1s"}}></span>
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: "0.2s"}}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-3 space-y-2 bg-white rounded-b-lg">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Ask about the forecast..."
            className={`flex-1 p-2 border rounded resize-none focus:outline-none focus:border-blue-500 transition text-sm ${
              isFocused ? "border-blue-500 shadow-sm" : "border-gray-300"
            }`}
            rows="2"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
            className="px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400 transition text-sm font-medium self-end"
            title="Send (Enter)"
          >
            📤
          </button>
        </div>
        <button
          onClick={handleReset}
          className="w-full text-xs py-1 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded transition"
        >
          🔄 Reset Conversation
        </button>
      </div>
    </div>
  );
}
