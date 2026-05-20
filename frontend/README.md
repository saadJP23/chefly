# Chefly Frontend

Railway should deploy this directory only.

## Railway Settings

```text
Root Directory: frontend
Start Command: npm start
```

Add:

```text
API_BASE_URL=https://your-lightsail-backend-domain.com
```

The server injects that value at `/config.js`, so you can change the backend URL without rebuilding static HTML by hand.

## Local Run

```bash
npm start
```

Then open:

```text
http://localhost:3000
```
