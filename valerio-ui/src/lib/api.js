const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

// --- Para el agente (ML, OpenAI, financieros) ---
export async function ask(question) {
  const r = await fetch(`${BASE}/valerio/ask`, {   // 👈 ojo: tu routes_openai.py está montado en /ask
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question }), // ✅ debe ser "question"
  });

  if (!r.ok) {
    let detail = "";
    try {
      const err = await r.json();
      detail = err.detail ?? "";
    } catch {}
    throw new Error(`HTTP ${r.status} ${detail}`);
  }
  return r.json();
}

// --- Healthcheck ---
export async function health() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
