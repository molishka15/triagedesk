# Triage Desk

A support ticket triage workspace that classifies urgency, category, sentiment, entities, confidence, and concise reasoning. Ticket text is sent to the configured Gemini API for analysis and is not stored by this application.

## Features

- Responsive ticket input and classification results.
- Official Google Gemini API integration with structured JSON output and independent validation.
- Server-side credentials only; ticket bodies and provider responses are not logged.
- Local Python server and Vercel Python Function deployments.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Run locally

~~~powershell
cd ticket_classifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-Google-AI-Studio-key"
$env:GEMINI_MODEL = "gemini-3.7-flash"
python -m src.main
~~~

Open http://127.0.0.1:8000. The app reads GEMINI_API_KEY and GEMINI_MODEL from its process environment; it does not load .env files. On macOS/Linux, use export GEMINI_API_KEY="your-key" and export GEMINI_MODEL="gemini-3.7-flash" before starting the server.

## Deploy to Vercel

The repository includes a Python Function for POST /api/classify and serves the page from src/index.html. Set Vercel's project Root Directory to the repository root. In **Project Settings > Environment Variables**, add:

- GEMINI_API_KEY: your Google AI Studio key. Keep it private; never commit it to GitHub.
- GEMINI_MODEL: gemini-3.7-flash.

Select Production and Preview as needed, save, then redeploy. Vercel applies environment-variable changes to new deployments. See the [Vercel Python runtime guide](https://vercel.com/docs/functions/runtimes/python) and [environment variable guide](https://vercel.com/docs/environment-variables).

## Configuration

| Variable | Required | Example | Purpose |
|---|---:|---|---|
| GEMINI_API_KEY | Yes | Set privately in the server environment | Google Gemini API credential. |
| GEMINI_MODEL | Yes | gemini-3.7-flash | Gemini model ID. Read from the environment; not hard-coded in Python. |
| HOST | No | 127.0.0.1 | Local server bind address. |
| PORT | No | 8000 | Local server port. |

.env.example is a reference template only. Copy its settings into the server environment; do not put a real API key in that file or in GitHub.

## API

POST /api/classify

Request:

~~~json
{"ticket_text":"Our team cannot access the dashboard."}
~~~

Success returns exactly the classification schema. Errors return {"error":"..."}. Ticket text must be a non-empty string of at most 10,000 characters.

## Validation and privacy

The application requests Gemini structured JSON output using the schema in src/classifier.py, then independently validates the result. Ticket text is treated as untrusted input. The app has no database or ticket history and does not log ticket bodies. Review AI classifications before consequential actions.

## Tests

~~~powershell
python -m unittest discover -s tests -v
~~~

Tests use mocked provider responses and do not make live API calls.

## Troubleshooting

- **Classification is not configured:** set GEMINI_API_KEY in the environment where the Python server runs.
- **Model is not configured:** set GEMINI_MODEL=gemini-3.7-flash in the server environment.
- **Vercel still reports a missing key:** ensure the variable is set in the right Vercel project and deployment environment, then redeploy.
- **Rate limit:** retry later; the free tier has usage limits.
- **Port is in use:** set PORT to another available port.
