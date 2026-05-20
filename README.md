# Chefly Deployment Layout

This copy separates the deploy targets so the frontend can move to Railway without carrying the Python backend, ML models, database files, virtual environment, or Elastic Beanstalk artifacts.

## Structure

```text
backend/
  index.py                 Flask app for Lightsail or another Python host
  templates/               Original Flask templates, kept for backend compatibility
  static/                  Original backend static files
  requirements.txt
  Procfile
  .env.example

frontend/
  package.json             Railway start command
  server.js                Small static server that uses Railway's PORT
  public/                  Static frontend served by Railway
```

## Deploy Frontend To Railway

1. Push this folder to GitHub.
2. In Railway, create or update the service from the GitHub repo.
3. Set the service root directory to:

```text
frontend
```

4. Add this Railway variable:

```text
API_BASE_URL=https://your-lightsail-backend-domain.com
```

5. Redeploy and open the generated Railway domain.
6. In Cloudflare, point `cheflys.com` or `www.cheflys.com` to the Railway custom domain target.

## Backend Notes

The backend remains deployable on Lightsail. It now exposes `application = app` for WSGI hosts and includes CORS configuration so the Railway frontend can call it.

Set this backend environment variable on Lightsail:

```text
CORS_ORIGINS=https://cheflys.com,https://www.cheflys.com,https://your-railway-domain.up.railway.app
```

Do not commit real `.env`, database files, virtual environments, `.DS_Store`, or deployment zip files.
