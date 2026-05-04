#!/bin/sh
# Railway / Docker: always migrate before runserver. If this fails, the container exits
# and the deploy will show the error (check Railway "Deploy logs").
set -e
echo "[start] Running database migrations..."
python manage.py migrate --noinput
echo "[start] Migrations done. Starting Gunicorn on port ${PORT:-8000}..."
# runserver is dev-only; Railway often returns 502 with it. Gunicorn is the supported WSGI server.
exec gunicorn dmi_project.wsgi:application \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers 1 \
  --threads 2 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
