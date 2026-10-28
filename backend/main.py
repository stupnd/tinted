import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.limits import MAX_UPLOAD_BYTES
from api.middleware import BodySizeLimitMiddleware
from api.routes import router

app = FastAPI(title="GlowMatch API")


# ── CORS ──────────────────────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS to the deployed frontend origin(s), comma-separated.
# Falls back to local dev origins so `npm run dev` works without config.

# With ALLOWED_ORIGINS unset we're in local dev, so accept any localhost port —
# `npm run dev -- -p 3002` shouldn't require reconfiguring the backend. Once
# ALLOWED_ORIGINS is set (i.e. in production) only those exact origins pass, and
# the localhost escape hatch is off.
DEV_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

_configured = os.environ.get("ALLOWED_ORIGINS", "").strip()
# Browsers send Origin without a trailing slash, so "https://x.app/" would never
# match. Strip it rather than fail silently.
allowed_origins = [o.strip().rstrip("/") for o in _configured.split(",") if o.strip()]

# Optional regex for origins that can't be listed up front, e.g. Vercel preview
# deployments: https://tinted-[a-z0-9]+-stupnds-projects\.vercel\.app
_configured_regex = os.environ.get("ALLOWED_ORIGIN_REGEX", "").strip()
dev_origin_regex = _configured_regex or (None if _configured else DEV_ORIGIN_REGEX)

if not _configured and not _configured_regex:
    print(
        "[cors] ALLOWED_ORIGINS is unset — allowing any localhost port (dev mode). "
        "Set it to your deployed origin before going public."
    )
else:
    print(f"[cors] allowing origins={allowed_origins} regex={_configured_regex or None}")

# Order matters: middleware added last is outermost, so CORS wraps the body-size
# check and a 413 still comes back with CORS headers the browser can read.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=dev_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)
