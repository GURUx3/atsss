# ATS Analyzer

This repository contains a **FastAPI backend** and a **Next.js frontend** implementing a resume analysis tool.

## Backend (Python)

Located in `backend/`.

### Setup & run

```bash
cd backend
python -m venv .venv             # create a virtualenv
. ./.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# start server (default port 5000, can be overridden)
PORT=5000 uvicorn backend.main:app --reload
```

The service exposes:

* `POST /api/analyze` – upload a PDF/DOCX file (`file` field) to analyze a resume.
* `GET  /api/jobs?role={role}` – fetches live job listings from Adzuna for the given role.
* additional endpoints for roles and health.

The app will listen on `http://localhost:5000` unless you specify a different port via the `PORT` environment variable.

CORS is enabled so the frontend (which runs on Next.js port 3000 by default) can access the API.

## Frontend (Next.js)

Located in `ATS-Analysis/`.

### Setup & run

```bash
cd ATS-Analysis
npm install
npm run dev
```

By default the frontend uses a **Next.js rewrite proxy** so that all client requests to `/api/*` are transparently forwarded to whatever backend service you configure (see `ATS-Analysis/next.config.ts`). This avoids CORS headaches and means you can simply call `/api/analyze` from the browser.

If you do need to talk to a backend running on a different host/port, there are two options:

```bash
# proxy target override (no `/api` suffix):
NEXT_PUBLIC_API_PROXY="http://localhost:3001" npm run dev

# or bypass proxy entirely with a full URL (including `/api`):
NEXT_PUBLIC_API_BASE_URL="http://localhost:3001/api" npm run dev
```

The `lib/api.ts` file will pick the appropriate value and compose the request URLs accordingly.
## Synchronisation details

* Backend now defaults to port **5000** and the API client uses `http://localhost:5000/api` as the default base URL.  This synchronises the dev environment without extra configuration.
* The backend `main.py` entry point reads `PORT` env var so the port can be changed if needed.
* CORS middleware allows all origins.
* Types defined in `lib/api.ts` align with the subset of fields the UI consumes; you can extend them as backend models grow.

Start both services simultaneously (one in each terminal) and upload a resume through the UI to see live analysis and suggested jobs.
# atsss
