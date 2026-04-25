import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def _load_target(target: str):
    module_name, attr = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def run(app, host: str = "127.0.0.1", port: int = 8000):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
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
