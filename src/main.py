"""Small standard-library web server for the ticket triage workspace."""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.classifier import ClassificationError, MAX_TICKET_CHARS, RateLimitError, classify_ticket

ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    server_version = "TriageDesk"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path not in ("/", "/index.html"):
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return
        self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/classify":
            self._send(404, b'{"error":"Not found"}', "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, b'{"error":"Invalid request."}', "application/json; charset=utf-8")
            return
        if length <= 0 or length > MAX_TICKET_CHARS * 4:
            self._send(413 if length > MAX_TICKET_CHARS * 4 else 400,
                       b'{"error":"Request must contain a ticket within the size limit."}',
                       "application/json; charset=utf-8")
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {"ticket_text"}:
                raise ValueError
            result = classify_ticket(payload["ticket_text"])
        except json.JSONDecodeError:
            self._send(400, b'{"error":"Request body must be valid JSON."}', "application/json; charset=utf-8")
            return
        except ValueError as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")
            return
        except ClassificationError as exc:
            status = 429 if isinstance(exc, RateLimitError) else 503
            self._send(status, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")
            return
        except Exception as exc:
            # Keep unexpected implementation/provider errors out of the response and
            # never log the ticket body or provider response.
            print(f"Unexpected classification failure: {type(exc).__name__}", file=sys.stderr)
            self._send(500, b'{"error":"Classification failed unexpectedly. Please retry."}',
                       "application/json; charset=utf-8")
            return
        self._send(200, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        # Avoid default request logs that include client supplied paths or details.
        return


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Triage Desk available at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
