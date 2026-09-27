"""Expose the shared classifier handler as a Vercel Python Function."""

from src.main import Handler


class handler(Handler):
    """Handle POST /api/classify with the same logic used by local development."""

    def do_GET(self) -> None:
        self._send(
            405,
            b'{"error":"Open the site homepage to classify a ticket; this API endpoint accepts POST only."}',
            "application/json; charset=utf-8",
        )