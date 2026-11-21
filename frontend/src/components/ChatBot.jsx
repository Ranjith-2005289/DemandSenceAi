import { useEffect, useRef, useState } from "react";
import { askChat, getProductSummary, resetChat, errorMessage } from "../api";
import { Icon } from "./ui";
import Markdown from "./Markdown";

const SUGGESTIONS = [
  "Why was this model selected?",
  "How reliable is this forecast?",
  "How much safety stock should I hold?",
  "What does MASE mean here?",
];

const GREETING = {
  type: "bot",
  content:
    "Hi — I'm your forecasting assistant. Ask me why a model won, how confident to be in the numbers, or how to turn the forecast into stock and staffing decisions.",
};

export default function ChatBot({ forecastResult, isVisible, onClose, groupInfo }) {
  const [messages, setMessages] = useState([GREETING]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => `session_${crypto.randomUUID()}`);
  const [productSummary, setProductSummary] = useState(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  // Per-group ranking, so "which products sell best" is answered from real numbers.
  useEffect(() => {
    if (!groupInfo?.group_col) return;
    getProductSummary(groupInfo.filename, groupInfo.date_col, groupInfo.target_col, groupInfo.group_col, 50)
      .then(setProductSummary)
      .catch(() => setProductSummary(null));
  }, [groupInfo?.filename, groupInfo?.group_col, groupInfo?.target_col]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  useEffect(() => {
    if (isVisible) setTimeout(() => inputRef.current?.focus(), 50);
  }, [isVisible]);

  const send = async (text) => {
    const question = (text ?? input).trim();
    if (!question || isLoading) return;
    setInput("");
    setMessages((prev) => [...prev, { type: "user", content: question }]);
    setIsLoading(true);
    try {
      const data = await askChat({ question, session_id: sessionId, forecast_data: forecastResult, product_summary: productSummary });
      setMessages((prev) => [...prev, { type: "bot", content: data.response }]);
    } catch (err) {
      setMessages((prev) => [...prev, { type: "bot", error: true, content: errorMessage(err, "Sorry — I couldn't answer that. Please try again.") }]);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setMessages([{ type: "bot", content: "Conversation cleared. What would you like to know about this forecast?" }]);
    resetChat(sessionId).catch(() => {});
  };

  if (!isVisible) return null;

  const onlyGreeting = messages.length === 1;

  return (
    <div className="fixed inset-x-3 bottom-3 z-50 flex h-[min(640px,calc(100vh-6rem))] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl animate-fade-in sm:inset-x-auto sm:right-6 sm:w-[420px]">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-600 text-white"><Icon.Sparkles width={18} height={18} /></span>
          <div>
            <p className="text-sm font-semibold text-slate-900">Forecast assistant</p>
            <p className="text-xs text-slate-500">Answers use this run's results</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" onClick={reset} className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700" title="Clear conversation">
            <Icon.Refresh width={16} height={16} />
          </button>
          <button type="button" onClick={onClose} className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700" title="Close">
            <Icon.X width={16} height={16} />
          </button>
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto bg-slate-50/60 px-4 py-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.type === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[90%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                m.type === "user"
                  ? "rounded-br-md bg-brand-600 text-white"
                  : m.error
                  ? "rounded-bl-md border border-rose-200 bg-rose-50 text-rose-800"
                  : "rounded-bl-md border border-slate-200 bg-white text-slate-700"
              }`}
            >
              {m.type === "bot" && !m.error ? <Markdown text={m.content} /> : <p className="whitespace-pre-wrap">{m.content}</p>}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-1.5 rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3">
              {[0, 0.15, 0.3].map((d) => (
                <span key={d} className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: `${d}s` }} />
              ))}
            </div>
          </div>
        )}
        {onlyGreeting && (
          <div className="flex flex-wrap gap-2 pt-1">
            {SUGGESTIONS.map((s) => (
              <button key={s} type="button" onClick={() => send(s)}
                className="rounded-full border border-brand-200 bg-white px-3 py-1.5 text-xs font-medium text-brand-700 transition hover:bg-brand-50">
                {s}
              </button>
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form onSubmit={(e) => { e.preventDefault(); send(); }} className="border-t border-slate-100 bg-white p-3">
        <div className="flex items-end gap-2 rounded-xl bg-slate-50 p-1.5 ring-1 ring-inset ring-slate-200 focus-within:ring-2 focus-within:ring-brand-500">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Ask about the forecast…"
            rows={1}
            className="max-h-32 flex-1 resize-none border-0 bg-transparent px-2 py-1.5 text-sm text-slate-800 outline-none placeholder:text-slate-400"
          />
          <button type="submit" disabled={isLoading || !input.trim()}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand-600 text-white transition hover:bg-brand-700 disabled:bg-slate-300" title="Send">
            <Icon.Send width={16} height={16} />
          </button>
        </div>
        <p className="mt-1.5 px-1 text-[11px] text-slate-400">AI answers can be wrong — check the numbers against the leaderboard.</p>
      </form>
    </div>
  );
}
