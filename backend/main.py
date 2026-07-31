import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import forecast, upload, chat, insights

app = FastAPI(title="Retail Forecasting Backend", version="0.1.0")

# Local dev origins always allowed; production frontend origin(s) come from
# the ALLOWED_ORIGINS env var (comma-separated) so this doesn't need a code
# change per deploy target.
_default_origins = [
    "http://localhost:3000", "http://localhost:5173", "http://localhost:5174",
    "http://localhost:5175", "http://127.0.0.1:5173",
]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(forecast.router)
app.include_router(chat.router)
app.include_router(insights.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "retail-forecasting-backend"}
