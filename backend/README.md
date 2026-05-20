# Chefly Backend

This is the original Flask backend, separated from the Railway frontend.

## Environment

Copy `.env.example` to `.env` locally and set real values outside Git.

Required:

```text
SECRET_KEY
SQLALCHEMY_DATABASE_URI
CORS_ORIGINS
```

Optional integrations:

```text
SPOON_API_KEY
OPENAI_API_KEY
CLOUDINARY_CLOUD_NAME
CLOUDINARY_API_KEY
CLOUDINARY_API_SECRET
MAIL_*
```

## Start

```bash
pip install -r requirements.txt
gunicorn index:application
```

For Lightsail, keep using your existing backend deployment process. Make sure the deployed backend allows CORS from the Railway frontend domain.
