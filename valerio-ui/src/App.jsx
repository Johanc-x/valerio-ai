// src/App.jsx
import { useState } from "react";
import { ask } from "./lib/api";  // 👈 quitamos callCalc
import { Send } from "lucide-react";

const MODELS = [
  { 
    key: "black-scholes",
    label: "Black-Scholes",
    template: "black scholes S=100 K=105 r=0.02 sigma=0.25 T=1.0 type=call"
  },
  { 
    key: "var",
    label: "VaR (Simple)",
    template: "VaR simple with returns=[-0.02,-0.01,0.01,0.015] confidence=0.95"
  }, 
  { 
    key: "var-montecarlo",
    label: "VaR (Monte Carlo)",
    template: "VaR Monte Carlo alpha=0.05 horizon=5 sims=10000 amount=200000"
  }, 
  { 
    key: "montecarlo",
    label: "Monte Carlo Simulation",
    template: "monte carlo S0=100 mu=0.05 sigma=0.2 T=1.0 steps=252 sims=10000"
  },
  { 
    key: "capm",
    label: "CAPM",
    template: "CAPM rf=0.02 beta=1.1 rm=0.08"
  },
  { 
    key: "markowitz",
    label: "Markowitz",
    template: "Markowitz returns=[0.1,0.15,0.2] cov=[[0.005,-0.010,0.004],[-0.010,0.040,-0.002],[0.004,-0.002,0.023]]"
  },
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const handleAsk = async () => {
    if (!question.trim()) return;

    setMessages(prev => [
      ...prev,
      { id: Date.now(), role: "user", text: question }
    ]);

    try {
      setMessages(prev => [
        ...prev,
        { id: Date.now() + 1, role: "system", text: "Valerio AI is processing your query..." }
      ]);

      const res = await ask(question);   // 🔑 ahora todo va por ask()

      const msg = res.message ?? res.answer ?? JSON.stringify(res);

      setMessages(prev => [
        ...prev.filter(m => m.role !== "system"),
        { id: Date.now() + 2, role: "valerio", text: msg, graph: res.graph }
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { id: Date.now() + 3, role: "error", text: `❌ Error: ${err.message}` }
      ]);
    }
    setQuestion("");
  };

  const handleKeyDown = e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-neutral-900 to-neutral-800 text-neutral-100 flex font-[Inter]">
      {/* Sidebar fija */}
      <aside className="fixed left-0 top-0 h-full w-64 flex flex-col bg-neutral-900/95 p-6 shadow-xl border-r border-neutral-800 z-10">
        <h1 className="text-2xl font-extrabold tracking-widest mb-10">
          VALERIO <span className="text-gray-300">AI</span>
        </h1>
        <h2 className="uppercase text-xs font-semibold text-neutral-300 mb-5 tracking-wider">
          Models
        </h2>
        <div className="flex flex-col gap-3">
          {MODELS.map(m => (
            <button
              key={m.key}
              onClick={() => setQuestion(m.template)}  // 👈 solo escribe en textarea
              className="text-left text-sm font-medium text-neutral-200 hover:text-violet-400 transition transform hover:-translate-y-0.5"
              title={`Use ${m.label}`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col justify-between px-10 py-8 ml-64">
        <div className="flex-1 overflow-y-auto space-y-6 mb-6">
          {messages.map(m => (
            <div
              key={m.id}
              className={`p-2 transition transform hover:-translate-y-1 hover:shadow-lg cursor-pointer max-w-2xl ${
                m.role === "user"
                  ? "text-neutral-200 ml-auto text-right"
                  : m.role === "valerio"
                  ? "text-neutral-100 text-left"
                  : m.role === "error"
                  ? "text-red-400 font-semibold"
                  : "text-neutral-400"
              }`}
            >
              <p className="whitespace-pre-wrap">{m.text}</p>
              {m.graph && (
                <img
                  src={`data:image/png;base64,${m.graph}`}
                  alt="Graph"
                  className="mt-4 rounded-lg shadow-lg border border-neutral-700"
                />
              )}
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="sticky bottom-0 py-4">
          <div className="relative max-w-3xl mx-auto w-full">
            <textarea
              className="w-full p-5 pr-14 rounded-2xl bg-neutral-800/80 text-neutral-100 placeholder:text-neutral-500 text-lg border border-neutral-700 focus:outline-none focus:ring-2 focus:ring-violet-500 resize-none"
              rows={2}
              placeholder="Type your question..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            {question.trim() && (
              <button
                onClick={handleAsk}
                className="absolute right-3 bottom-3 p-3 bg-violet-600 hover:bg-violet-500 rounded-full shadow-lg transition"
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
