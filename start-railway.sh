#!/bin/sh
# Railway / Docker: always migrate before runserver. If this fails, the container exits
# and the deploy will show the error (check Railway "Deploy logs").
set -e
echo "[start] Running database migrations..."
python manage.py migrate --noinput
echo "[start] Migrations done. Starting web server on port ${PORT:-8000}..."
exec python manage.py runserver "0.0.0.0:${PORT:-8000}"
