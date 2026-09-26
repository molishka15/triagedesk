# Triage Desk

A small support operations workspace that classifies a customer ticket into urgency, category, sentiment, named entities, confidence, and concise reasoning. Tickets are sent to the configured AI provider for analysis and are not persisted by this app.

## Features

- Responsive ticket input and scannable classification results.
- OpenRouter Chat Completions API integration with JSON output mode and independent strict JSON Schema output and independent validation.
- Independent validation of every model result, including enums, exact fields, confidence range, and the 20-word reasoning limit.
- Input size checks, user-safe provider errors, loading/error states, and no ticket-content logging.
- No database, account system, browser-side API keys, or retained ticket history.

## Architecture

- `src/main.py`: standard-library threaded HTTP server and `/api/classify` endpoint.
- `src/classifier.py`: provider configuration, classification instructions, schema, request, and independent validation.
- `src/index.html`: self-contained responsive support-agent UI.
- `tests/test_classifier.py`: contract and provider behavior tests using mocked HTTP responses.

The web server binds to `127.0.0.1` by default. Run behind a properly configured TLS reverse proxy before exposing it to a network. Authentication and production-grade rate limiting are not included; deploy behind your organization's access controls if shared access is required.

## Requirements

- Python 3.10 or newer
- An OpenRouter API key with access to the configured model

## Setup and run

```powershell
cd ticket_classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENROUTER_API_KEY = "your-key"
python -m src.main
```

Open <http://127.0.0.1:8000>. On macOS/Linux, activate the environment with `source .venv/bin/activate` and set the key with `export OPENROUTER_API_KEY="your-key"`.

Copy `.env.example` as a reference for the required settings. The application intentionally does not read `.env` files; provide configuration through the process environment or your deployment secret manager.

## Configuration

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Server-side provider credential. Never put this in frontend code. |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o-mini` | Model name accepted by the OpenRouter Chat Completions API. |
| `HOST` | No | `127.0.0.1` | Bind address. |
| `PORT` | No | `8000` | HTTP listening port. |

## API

`POST /api/classify`

Request:

```json
{"ticket_text":"Our team cannot access the dashboard."}
```

Success returns exactly the classification contract. Errors return `{"error":"..."}` with a 4xx or 5xx response. Ticket text must be a non-empty string of at most 10,000 characters. The API does not log request bodies or provider response contents.

## Validation and AI behavior

The provider receives a compact triage instruction and the raw ticket as a separate user message. OpenRouter strict JSON Schema output is requested, then the application independently validates every field, enum, confidence score, and reasoning length. Invalid output becomes a controlled error and is never repaired by inventing data.

The ticket is treated as untrusted input, and instructions embedded in it must not override the classifier. The prompt asks the model to ignore those instructions. Prompt injection resistance is not a formal security boundary; review classifications before consequential actions.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover strict JSON Schema output and independent validation, missing entities, malformed enums, unexpected fields, confidence bounds, reasoning length, missing credentials, input limits, strict OpenRouter JSON Schema request configuration, malformed model output, and provider errors. They do not make live API calls.

## Privacy and security

- Configure credentials only on the server process, preferably through a secrets manager.
- Ticket text is sent to the configured AI provider for classification; check your organization's provider and data-handling requirements before use.
- This application does not store tickets. Avoid adding request-body logs, browser persistence, or ticket analytics without an explicit retention policy.
- The built-in server has no user authentication, durable rate limiting, or TLS. It defaults to loopback for local use. Do not expose it publicly as-is.
- Provider details, credentials, and stack traces are not returned to users.

## Troubleshooting

- **Classification is not configured:** set `OPENROUTER_API_KEY` in the environment where Python runs.
- **Service is busy:** wait briefly and retry; provider rate limits are surfaced as a safe retry message.
- **Invalid classification:** retry; the app rejects malformed results rather than displaying unvalidated model data.
- **Port is in use:** set `PORT` to another available port.
