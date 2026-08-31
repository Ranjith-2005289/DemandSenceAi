import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { getProductSummary } from "../api";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bot, X, Send, RefreshCw, MessageSquare } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function ChatBot({ forecastResult, isVisible, onClose, groupInfo }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [productSummary, setProductSummary] = useState(null);
  const messagesEndRef = useRef(null);

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

  useEffect(() => {
    if (!sessionId) {
      const newSessionId = `session_${crypto.randomUUID()}`;
      setSessionId(newSessionId);
      
      setMessages([
        {
          type: "bot",
          content: "Hi! I'm your AI forecasting assistant. Ask me anything about this forecast - model selection, performance, predictions, or recommendations!"
        }
      ]);
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    
    setMessages(prev => [...prev, { type: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/chat/ask`, {
        question: userMessage,
        session_id: sessionId,
        forecast_data: forecastResult,
        product_summary: productSummary,
      });

      setMessages(prev => [...prev, { type: "bot", content: response.data.response }]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { 
        type: "bot", 
        content: "Sorry, I encountered an error. Please try again." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        type: "bot",
        content: "Conversation reset. Ask me anything about this forecast!"
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
    <Card className="fixed bottom-4 right-4 w-96 h-[600px] shadow-2xl flex flex-col border-slate-200 z-50 overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-4 flex flex-row items-center justify-between rounded-t-lg">
        <div className="flex items-center gap-3">
          <Bot className="h-6 w-6 text-blue-100" />
          <div>
            <CardTitle className="text-lg font-bold">Forecast Assistant</CardTitle>
            <p className="text-xs text-blue-100 font-medium">Powered by AI</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="text-white hover:bg-white/20 h-8 w-8 rounded-full"
        >
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <MessageSquare className="h-10 w-10 mb-2 opacity-20" />
            <p className="text-sm font-medium">Ask a question to get started</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                  msg.type === "user"
                    ? "bg-blue-600 text-white rounded-br-sm"
                    : "bg-white text-slate-800 border border-slate-200 rounded-bl-sm"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-bl-sm shadow-sm flex items-center gap-1.5 h-10">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></span>
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: "0.15s"}}></span>
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: "0.3s"}}></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </CardContent>

      <CardFooter className="border-t border-slate-200 p-3 flex flex-col gap-2 bg-white rounded-b-lg">
        <div className="flex w-full items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about the forecast..."
            className="flex-1 min-h-[44px] max-h-32 p-3 border border-slate-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-sm shadow-sm"
            rows="1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={isLoading || !input.trim()}
            className="h-[44px] w-[44px] rounded-xl bg-blue-600 hover:bg-blue-700 shadow-sm shrink-0 p-0"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          className="w-full text-xs text-slate-500 hover:text-blue-700 hover:bg-blue-50"
        >
          <RefreshCw className="mr-2 h-3 w-3" />
          Reset Conversation
        </Button>
      </CardFooter>
    </Card>
  );
}
