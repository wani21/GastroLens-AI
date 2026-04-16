# GastroLens-AI Frontend

React + Vite + Tailwind CSS frontend for the GastroLens-AI gastrointestinal disease classifier.

## Features

- **Landing page** with project overview, model metrics, and class information
- **Image analysis** — drag-and-drop upload, preview, and probability bars for all 4 classes
- **Video analysis** — frame sampling, dominant class detection, and per-frame breakdown
- Minimalist black/white design with emerald accent

## Tech Stack

- **Vite** — fast dev server and build tool
- **React 19** + React Router — UI and routing
- **Tailwind CSS v3** — utility-first styling
- **Axios** — HTTP client for backend calls

## Prerequisites

- Node.js 18+ (tested on Node 24)
- Backend running at `http://localhost:8000` (see `../backend/README.md`)

## Setup

```bash
cd frontend
npm install
```

## Run the Dev Server

```bash
npm run dev
```

Open http://localhost:5173

## Build for Production

```bash
npm run build
npm run preview    # test the production build locally
```

## Configuration

Backend URL defaults to `http://localhost:8000`. To change it, create a `.env` file:

```
VITE_API_URL=http://your-backend-host:8000
```

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.js            # Axios backend client
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── Dropzone.jsx         # Drag-and-drop file upload
│   │   └── ProbabilityBars.jsx  # Visual probability display
│   ├── pages/
│   │   ├── LandingPage.jsx
│   │   ├── ImagePage.jsx
│   │   └── VideoPage.jsx
│   ├── App.jsx                  # Routes + layout
│   ├── main.jsx                 # Entry point
│   └── index.css                # Tailwind directives
├── tailwind.config.js
├── postcss.config.js
└── vite.config.js
```
