# Chefly Frontend (Static assets + Flask proxy)

This directory serves **static assets** (`/static/css`, `/static/js`, images) and **proxies all HTML pages to the Flask backend**, which renders Jinja templates.

## Railway Settings

```text
Root Directory: frontend
Start Command: npm start
```

Required:

```text
API_BASE_URL=https://your-lightsail-backend-domain.com
```

With `API_BASE_URL` set, requests like `/generate.html` or `/upload.html` are proxied to Flask routes (`/generate`, `/upload`) and return fully rendered Jinja HTML — not raw template source.

## Local development

**Option A — Flask only (recommended):**

```bash
cd backend
source .venv/bin/activate
python index.py
```

Open `http://127.0.0.1:8080`

**Option B — Frontend proxy + Flask backend:**

```bash
# Terminal 1
cd backend && python index.py

# Terminal 2
cd frontend
API_BASE_URL=http://127.0.0.1:8080 npm start
```

Open `http://localhost:3000`

## Templates

All page markup lives in `backend/templates/` (Jinja). Do not add full HTML pages to `frontend/public/` — only `404.html` and static assets belong here.
