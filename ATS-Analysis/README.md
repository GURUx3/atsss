# ATS-Analysis Intelligence Platform

A minimal, high-precision resume intelligence platform built for modern enterprise recruitment workflows. This system provides deep structural analysis of candidate profiles, skill-gap detection, and live market alignment.

## Core Features

- **Structural Parsing**: Advanced extraction of experience hierarchies from PDF/DOCX source files.
- **ATS Calibration**: Keyword-density mapping and alignment scoring based on recruiter-defined critical skills.
- **Market Velocity**: Real-time matching of candidate profiles to live job listings and role requirements.
- **Enterprise Aesthetic**: Sharp, minimal monochrome UI optimized for high-density information review.

## Tech Stack

- **Framework**: Next.js 14+ (App Router)
- **Styling**: Tailwind CSS (Utility-first, structured grid)
- **Typography**: Inter (Sans-serif) / Geist Mono (Display)
- **Icons**: Lucide React
- **API**: Integration with ATS-Analyzer Backend Service

## Design Principles

- **Minimalist**: Maximum whitespace, essential visual elements only.
- **Structured**: 12-column grid alignment, 8px spacing scale.
- **Sharp**: Maximum 4px border-radius across all components.
- **Monochrome**: Black & white primary palette with slate/gray neutrals.

## Project Structure

```text
ATS-Analyzer/
├── frontend/
│   ├── app/
│   │   ├── globals.css      # Core design system & theme
│   │   ├── layout.tsx       # Root layout (No Navbar)
│   │   └── page.tsx         # Main entry point (Sectional layout)
│   ├── components/
│   │   ├── Hero.tsx         # Enterprise CTA section
│   │   ├── FileUpload.tsx   # Precision upload panel
│   │   ├── ResultsDashboard.tsx # Analysis report (Split layout)
│   │   ├── Insights.tsx     # Capability grid
│   │   └── Footer.tsx       # Minimal platform footer
│   ├── lib/
│   │   └── api.ts           # Backend service integration
│   ├── public/              # Static assets
│   ├── .gitignore           # DevOps exclusions
│   ├── README.md            # Platform documentation
│   └── package.json         # Dependencies & scripts
└── backend/                 # Backend analysis engine (reference)
```

## Setup Instructions

1.  **Clone & Install**:
    ```bash
    git clone git@github.com:GURUx3/ATS-Analysis.git
    cd ATS-Analysis/frontend
    npm install
    ```

2.  **Environment Setup**:
    Ensure the backend service is running at `http://localhost:5000` (the default).

    During development the Next.js server will automatically forward any request
    starting with `/api` to the backend using a rewrite rule defined in
    `next.config.ts` – no special configuration is required and CORS is not
    needed.

    If you need to run the backend on a different port or host, set
    `NEXT_PUBLIC_API_PROXY`. e.g.:
    ```bash
    NEXT_PUBLIC_API_PROXY="http://localhost:3001" npm run dev
    ```

    Alternatively, you may circumvent the proxy and point directly at a full
    URL by exporting `NEXT_PUBLIC_API_BASE_URL` (include `/api` in the value):
    ```bash
    NEXT_PUBLIC_API_BASE_URL="http://localhost:3001/api" npm run dev
    ```
    The code chooses `/api` by default when neither variable is defined.

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```

4.  **Production Build**:
    ```bash
    npm run build
    npm run start
    ```

## Git Deployment Commands

To initialize and push this project from scratch:

```bash
# Initialize repository
git init

# Add all production files
git add .

# Initial commit
git commit -m "feat: complete enterprise rebuild of ATS frontend"

# Add remote origin
git remote add origin git@github.com:GURUx3/ATS-Analysis.git

# Push to main branch
git branch -M main
git push -u origin main
```

---
*Precision Resume Intelligence Platform*
