import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from server.app import app


LANDING_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ProcureRL</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe3f0;
      --blue: #1d4ed8;
      --teal: #0f766e;
      --amber: #b45309;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 35%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 35%),
        var(--bg);
      color: var(--ink);
    }
    .shell {
      max-width: 1040px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }
    .hero {
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.08), rgba(15, 118, 110, 0.10));
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 28px;
      padding: 36px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    .eyebrow {
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(29, 78, 216, 0.12);
      color: var(--blue);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    h1 {
      margin: 18px 0 14px;
      font-size: clamp(2.2rem, 6vw, 4rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .lede {
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: 1.08rem;
      line-height: 1.7;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-top: 26px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 1rem;
    }
    .card p, .card li {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    ul {
      padding-left: 18px;
      margin: 0;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #eff6ff;
      color: #1e3a8a;
      padding: 2px 6px;
      border-radius: 8px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 28px;
    }
    .action {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 12px 16px;
      border-radius: 999px;
      text-decoration: none;
      font-weight: 600;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .action.primary {
      background: var(--blue);
      color: white;
      box-shadow: 0 10px 20px rgba(29, 78, 216, 0.25);
    }
    .action.secondary {
      background: white;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .action:hover {
      transform: translateY(-1px);
    }
    .footer {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.95rem;
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <span class="eyebrow">OpenEnv Hackathon 2026</span>
      <h1>ProcureRL</h1>
      <p class="lede">
        ProcureRL is an OpenEnv-compatible procurement negotiation environment where a buyer LLM
        learns to negotiate over price, delivery, and quality against a stable scripted seller
        under real policy constraints.
      </p>
      <div class="grid">
        <article class="card">
          <h2>Why it matters</h2>
          <p>
            Real procurement is not just price haggling. Agents must reason with incomplete
            information, market signals, competitor pressure, and a hard budget ceiling.
          </p>
        </article>
        <article class="card">
          <h2>Core API</h2>
          <ul>
            <li><code>POST /reset</code></li>
            <li><code>POST /step</code></li>
            <li><code>GET /state/{session_id}</code></li>
            <li><code>GET /health</code></li>
          </ul>
        </article>
        <article class="card">
          <h2>Training setup</h2>
          <p>
            Qwen2.5-3B-Instruct + Unsloth + TRL/GRPO with shaped reward, scenario-aware prompting,
            and saved training plots from real runs.
          </p>
        </article>
      </div>
      <div class="actions">
        <a class="action primary" href="/health">Check health</a>
        <a class="action secondary" href="https://github.com/ShivenduShivu/ProcureRL">GitHub repo</a>
        <a class="action secondary" href="https://colab.research.google.com/drive/1E3w2Uac9HYaPov4_lOiFdqVxk7lMlb1e?usp=sharing">Training notebook</a>
      </div>
      <p class="footer">
        The root page is a human-friendly landing page. Judges and clients can still call the JSON
        environment endpoints directly.
      </p>
    </section>
  </main>
</body>
</html>
"""

ROBOTS_TXT = "User-agent: *\nAllow: /\n"


def run(host: str = "0.0.0.0", port: int = 7860):
    class Handler(BaseHTTPRequestHandler):
        def _send_text(self, status_code: int, body: str, content_type: str):
            payload = body.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status_code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(200, LANDING_PAGE, "text/html")
                return
            if parsed.path == "/robots.txt":
                self._send_text(200, ROBOTS_TXT, "text/plain")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status_code, payload = app._dispatch("GET", parsed.path)
            self._send_json(status_code, payload)

        def do_POST(self):
            parsed = urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            status_code, response_payload = app._dispatch("POST", parsed.path, json_body=payload)
            self._send_json(status_code, response_payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
