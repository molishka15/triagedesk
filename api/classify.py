"""Expose the shared classifier handler as a Vercel Python Function."""

from src.main import Handler


class handler(Handler):
    """Handle POST /api/classify with the same logic used by local development."""

