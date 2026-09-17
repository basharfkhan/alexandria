import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Book
from app.routers import auth, books, chat, me
from app.services.recommender import get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alexandria")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        try:
            get_service(db)  # warm the in-memory model so the first request is fast
            log.info("recommender loaded")
        except LookupError as exc:
            log.warning("%s", exc)
    yield


app = FastAPI(
    title="Alexandria API",
    version="0.1.0",
    description="Personalised book recommendations: hybrid content + collaborative filtering with online feedback.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LookupError)
async def catalog_missing(_: Request, exc: LookupError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", tags=["ops"])
def health():
    with SessionLocal() as db:
        n_books = db.scalar(select(func.count()).select_from(Book))
    return {"status": "ok", "books": n_books}


app.include_router(auth.router)
app.include_router(books.router)
app.include_router(me.router)
app.include_router(chat.router)
