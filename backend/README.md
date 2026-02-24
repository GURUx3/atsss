# ATS-Analyzer Backend

This service powers the resume analysis and job‑fetching API consumed by the Next.js frontend.

## Features

* Accepts PDF/DOCX resumes and extracts plain text using `pdfplumber` / `python-docx`.
* Computes an ATS score against predefined role skills.
* Performs naive candidate metadata extraction (name, email, phone, organization, location).
* Queries the Adzuna job API to return live listings for a predicted role.
* CORS enabled so the frontend (running on port 3000) can reach it.

## Quickstart

```bash
# create & activate venv (Windows example)
python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt

# start development server (auto-reloads)
# default port is 5000, override with PORT env var if needed:
PORT=5000 uvicorn backend.main:app --reload
```

The service listens on `http://localhost:<PORT>` (default 5000) with these endpoints:

* `POST /api/analyze` &ndash; multipart file upload (`file` key) returns analysis JSON.  Files are limited to 10 MB by default (use `MAX_FILE_SIZE_MB` env var to adjust).
* `GET  /api/jobs?role=Data%20Scientist` &ndash; returns paged job results.

Keep the backend running while you start the Next.js application.
