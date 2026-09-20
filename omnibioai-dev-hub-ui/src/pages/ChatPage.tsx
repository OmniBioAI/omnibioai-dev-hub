import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { ragStream } from "../api/client";
import type { AskResult } from "../api/client";
import Citations from "../components/Citations";

interface Message {
  role: "user" | "bot";
  text: string;
  result?: AskResult;
  streaming?: boolean;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "bot",
      text: "Hello! I'm Ask OmniBioAI. I answer only from OmniBioAI's published documentation, with sources. If the documentation doesn't cover your question, I'll tell you rather than guess.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");
    setLoading(true);

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setMessages((prev) => [...prev, { role: "bot", text: "", streaming: true }]);

    // Only the VERIFIED answer is ever rendered. Raw model tokens are ignored on purpose:
    // an unverified answer must never appear as an OmniBioAI documentation answer.
    await ragStream(
      query,
      () => {},
      (fullContent?: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "bot") {
            updated[updated.length - 1] = { ...last, text: fullContent ?? last.text, streaming: false };
          }
          return updated;
        });
        setLoading(false);
      },
      (err) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "bot", text: `Error: ${err}`, streaming: false };
          return updated;
        });
        setLoading(false);
      },
      (result) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "bot") {
            updated[updated.length - 1] = { ...last, result, text: result.content, streaming: false };
          }
          return updated;
        });
      }
    );
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <>
      <div className="page-header">
        <div className="page-title">Ask OmniBioAI</div>
        <div className="page-sub">Answers only from published documentation · every answer cites its sources</div>
      </div>

      <div className="surface" style={{ flex: 1, display: "flex", flexDirection: "column", padding: 0, overflow: "hidden", minHeight: 0, height: "calc(100vh - 180px)" }}>
        <div className="chat-messages">
          {messages.map((msg, i) => {
            const noAnswer = msg.role === "bot" && msg.result && !msg.result.grounded;
            return (
              <div key={i} className={`chat-bubble ${msg.role}${noAnswer ? " no-answer" : ""}`}>
                {msg.role === "bot" ? (
                  noAnswer ? (
                    <div style={{ fontSize: 13 }} role="status" data-answer-status={msg.result!.answer_status}>
                      <div className="no-answer-title">No trusted documentation answer</div>
                      {msg.text}
                    </div>
                  ) : (
                    <div style={{ fontSize: 13 }} data-grounded={msg.result?.grounded ? "true" : undefined}>
                      {msg.streaming && !msg.text ? (
                        <span className="searching">Searching the documentation…</span>
                      ) : (
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                      )}
                      {msg.streaming && <span className="cursor-blink" />}
                    </div>
                  )
                ) : (
                  <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "var(--sans)", fontSize: 13 }}>
                    {msg.text}
                  </pre>
                )}
                {msg.result?.grounded && <Citations citations={msg.result.citations} />}
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="chat-input-row">
          <textarea
            className="chat-input"
            placeholder="Ask OmniBioAI... (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            rows={1}
          />
          <button className="btn-send" onClick={send} disabled={loading}>
            {loading ? <><span className="spinner" />Searching</> : "Ask →"}
          </button>
        </div>
      </div>
    </>
  );
}