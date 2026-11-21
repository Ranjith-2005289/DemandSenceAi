import { useEffect, useRef, useState } from "react";
import { ArrowUp, RotateCcw, X } from "lucide-react";
import { askChat, errorMessage, resetChat } from "@/api";
import { useAppState } from "@/state/AppState";
import { IconButton } from "@/components/ui";
import { modelLabel } from "@/lib/format";
import { llmUnavailableReason } from "@/lib/insights";
import Markdown from "./Markdown";

const SUGGESTIONS = [
  "Why was this model selected?",
  "How reliable is this forecast?",
  "What should I plan for next period?",
  "Which products need attention?",
];

const GREETING = {
  role: "assistant",
  content: "Ask about this forecast: model choice, accuracy, the outlook, or what to plan for. Answers are based on this run's results.",
};

/** Docked right-hand panel. Stays mounted while hidden so the conversation is kept. */
export default function AssistantPanel({ open, onClose }) {
  const { view, products } = useAppState();
  const [chatId] = useState(() => `session_${crypto.randomUUID()}`);
  const [messages, setMessages] = useState([GREETING]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const send = async (text) => {
    const q = (text ?? input).trim();
    if (!q || busy || !view) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setBusy(true);
    try {
      const res = await askChat({ question: q, session_id: chatId, forecast_data: view, product_summary: products.data });
      const text = String(res.response || "");
      if (text.startsWith("Error generating response")) {
        setMessages((m) => [...m, { role: "assistant", content: `${llmUnavailableReason(text)} Ask your administrator to check the server configuration.`, error: true }]);
      } else {
        setMessages((m) => [...m, { role: "assistant", content: text }]);
      }
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", content: errorMessage(err, "The assistant couldn't answer."), error: true }]);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setMessages([GREETING]);
    resetChat(chatId).catch(() => {});
  };

  return (
    <aside
      className={
        open
          ? "flex h-full w-[380px] max-w-full shrink-0 flex-col border-l border-line bg-surface max-lg:fixed max-lg:bottom-0 max-lg:right-0 max-lg:top-12 max-lg:z-40 max-lg:h-auto max-lg:shadow-pop"
          : "hidden"
      }
      aria-label="Assistant"
    >
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-line px-4">
        <div>
          <p className="text-[13px] font-semibold text-ink">Assistant</p>
          {view && (
            <p className="text-2xs text-ink-3">
              {view.group ? `${view.group.group_col}: ${view.group.group_value}` : "All data"} · {modelLabel(view.best_model?.best_model_name)}
            </p>
          )}
        </div>
        <div className="flex items-center">
          <IconButton icon={RotateCcw} label="Clear conversation" onClick={reset} />
          <IconButton icon={X} label="Close assistant" onClick={onClose} />
        </div>
      </div>

      <div className="scroll-thin flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] whitespace-pre-wrap rounded-md bg-brand px-3 py-2 text-[13px] text-white">{m.content}</div>
            </div>
          ) : (
            <div key={i} className="text-[13px] leading-relaxed text-ink">
              <p className="mb-1 text-2xs font-medium uppercase tracking-wide text-ink-3">Assistant</p>
              <div className={m.error ? "text-bad" : ""}>
                <Markdown text={m.content} />
              </div>
            </div>
          )
        )}
        {busy && (
          <div className="flex items-center gap-1.5 py-1" aria-label="Assistant is typing">
            {[0, 0.15, 0.3].map((d) => (
              <span key={d} className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-3" style={{ animationDelay: `${d}s` }} />
            ))}
          </div>
        )}
        {messages.length === 1 && (
          <div className="space-y-1.5 pt-1">
            <p className="text-2xs font-medium uppercase tracking-wide text-ink-3">Suggested</p>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => send(s)}
                className="block w-full rounded border border-line px-3 py-1.5 text-left text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:bg-surface-hover hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form
        className="border-t border-line p-3"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <div className="flex items-end gap-2 rounded border border-line-strong bg-surface px-2.5 py-1.5 focus-within:border-brand">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask a question…"
            className="max-h-32 min-h-[24px] flex-1 resize-none bg-transparent py-0.5 text-[13px] text-ink placeholder:text-ink-3 focus:shadow-none"
            style={{ outline: "none" }}
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            aria-label="Send"
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded bg-brand text-white transition-colors hover:bg-brand-hover disabled:opacity-40"
          >
            <ArrowUp className="h-3.5 w-3.5" />
          </button>
        </div>
        <p className="mt-1.5 text-2xs text-ink-3">Enter to send · Shift+Enter for a new line</p>
      </form>
    </aside>
  );
}
