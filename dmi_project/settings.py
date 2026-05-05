import os
from pathlib import Path
from datetime import timedelta
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY', default='django-insecure-change-this-in-production')

DEBUG = config('DEBUG', default=True, cast=bool)

ALLOWED_HOSTS = [
    h.strip()
    for h in config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')
    if h.strip()
]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'analysis',
    'forum',
    'archive',
    'users',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'dmi_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'dmi_project.wsgi.application'

# Railway / managed Postgres often requires TLS; without sslmode=require, DB calls can 500.
_db_host = config('DB_HOST', default='localhost')
_ssl_flag = config('DATABASE_SSL_REQUIRE', default='').strip().lower()
if _ssl_flag in ('true', '1', 'yes'):
    _pg_ssl = True
elif _ssl_flag in ('false', '0', 'no'):
    _pg_ssl = False
else:
    _h = _db_host.lower()
    # Railway private Postgres (*.railway.internal): TLS usually breaks here — do not force SSL.
    # Public Railway DB hostnames use *.proxy.rlwy.net — those need sslmode=require.
    if 'railway.internal' in _h:
        _pg_ssl = False
    elif 'rlwy.net' in _h:
        _pg_ssl = True
    elif any(m in _h for m in ('neon.tech', 'amazonaws.com', 'supabase.co')):
        _pg_ssl = True
    else:
        _pg_ssl = False

_db = {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': config('DB_NAME', default='dmi_db'),
    'USER': config('DB_USER', default='user'),
    'PASSWORD': config('DB_PASSWORD', default='password'),
    'HOST': _db_host,
    'PORT': config('DB_PORT', default='5432'),
}
if _pg_ssl:
    _db['OPTIONS'] = {'sslmode': 'require'}

DATABASES = {'default': _db}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'users.User'

# JWT: access token valid 7 days so user stays logged in (no quick logout)
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=7),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=30),
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://dmi-frontend.vercel.app",
]
# Allow Vercel preview deployments too (e.g. dmi-frontend-git-*.vercel.app)
CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^https://.*\.vercel\.app$",
]

_cors_extra = config("CORS_EXTRA_ORIGINS", default="").strip()
if _cors_extra:
    CORS_ALLOWED_ORIGINS.extend(
        [o.strip() for o in _cors_extra.split(",") if o.strip()]
    )

# Allow credentials for CORS
CORS_ALLOW_CREDENTIALS = True

# For development, allow all origins (remove in production)
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True

# Django 4+ — trusted origins for HTTPS (admin, CSRF on cross-subdomain if needed)
CSRF_TRUSTED_ORIGINS = config(
    "CSRF_TRUSTED_ORIGINS",
    default="https://dmi-frontend.vercel.app,https://*.vercel.app,https://dmi-backend-production.up.railway.app",
).split(",")
CSRF_TRUSTED_ORIGINS = [o.strip() for o in CSRF_TRUSTED_ORIGINS if o.strip()]
_csrf_extra = config("CSRF_TRUSTED_ORIGINS_EXTRA", default="").strip()
if _csrf_extra:
    CSRF_TRUSTED_ORIGINS.extend(
        [o.strip() for o in _csrf_extra.split(",") if o.strip()]
    )

CELERY_BROKER_URL = config('REDIS_URL', default='redis://localhost:6379/0')
CELERY_RESULT_BACKEND = config('REDIS_URL', default='redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE

# Log tracebacks for 500s to stdout (Railway "Deploy Logs" / gunicorn).
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}

# ==============================================================================
# MODEL SELECTION
# ==============================================================================
# Image detection: ViT phase4_replay_best (only). File must be in ml_models/weights/
MODEL_NAME = config('MODEL_NAME', default='phase4_replay_best.pth')
TEXT_MODEL_NAME = config('TEXT_MODEL_NAME', default='deberta-v3-bestmodel.pth')
# Text inference: when DEBUG=True, default cpu so runserver does not die from CUDA OOM on first
# text analyze (laptop). Use TEXT_INFERENCE_DEVICE=cuda in .env when GPU + VRAM are enough.
_text_infer_default = "cpu" if DEBUG else ""
TEXT_INFERENCE_DEVICE = config(
    "TEXT_INFERENCE_DEVICE", default=_text_infer_default
).strip().lower()