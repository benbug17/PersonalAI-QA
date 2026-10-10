"""
FastAPI backend for the Voice Learning Assistant.

Pipeline for /voice:  audio upload -> STT (asr.py) -> LLM (groq_client.py)
                      -> save history (db.py) -> TTS (tts.py)

Run locally:  uvicorn api:app --reload
"""

import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

import asr
import db
import groq_client
import tts

logger = logging.getLogger("personalai")

MAX_QUERY_CHARS = 2000
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB
HASH_RE = re.compile(r"^[a-f0-9]{64}$")

# token -> user_id. In-memory on purpose: simple, and tests can reset it.
SESSIONS: Dict[str, int] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="PersonalAI", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- schemas
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=72)

    @field_validator("username")
    @classmethod
    def _username_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("username must not be blank")
        return v

    @field_validator("password")
    @classmethod
    def _password_bcrypt_limit(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:  # bcrypt ignores/rejects bytes past 72
            raise ValueError("password must be at most 72 bytes")
        return v


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=72)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    speak: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


# ---------------------------------------------------------------- helpers
bearer_scheme = HTTPBearer(auto_error=False, description="Token returned by POST /auth/login")
_UNAUTHORIZED = {"WWW-Authenticate": "Bearer"}


def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> int:
    if creds is None or not creds.credentials.strip():
        raise HTTPException(status_code=401, detail="Missing bearer token", headers=_UNAUTHORIZED)
    user_id = SESSIONS.get(creds.credentials.strip())
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token", headers=_UNAUTHORIZED)
    return user_id


def _synthesize(text: str) -> Optional[str]:
    """Return an /audio URL, or None if TTS failed (text reply is still valid)."""
    path = tts.text_to_speech(text)
    if not path:
        return None
    return f"/audio/{os.path.splitext(os.path.basename(path))[0]}"


def _answer(user_id: int, query: str, speak: bool) -> dict:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=503, detail="LLM is not configured")

    response = groq_client.get_learning_assistant_response(query)
    if not response:
        raise HTTPException(status_code=502, detail="LLM request failed")

    try:
        db.save_conversation(user_id, query, response)
    except Exception:  # history is best-effort; never lose the answer
        logger.exception("Could not save conversation")

    return {
        "query": query,
        "response": response,
        "audio_url": _synthesize(response) if speak else None,
    }


# ---------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest):
    user_id = db.create_user(body.username, body.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="Username already exists")
    return {"user_id": user_id}


@app.post("/auth/login")
def login(body: LoginRequest):
    user_id = db.authenticate_user(body.username, body.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = user_id
    return {"token": token, "user_id": user_id}


@app.post("/chat")
def chat(body: ChatRequest, user_id: int = Depends(current_user)):
    return _answer(user_id, body.query, body.speak)


@app.post("/voice")
def voice(file: UploadFile = File(...), user_id: int = Depends(current_user)):
    audio = file.file.read(MAX_AUDIO_BYTES + 1)
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large")

    transcript = asr.transcribe_audio(audio)
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio")

    result = _answer(user_id, transcript, speak=True)
    result["transcript"] = result.pop("query")
    return result


@app.get("/history")
def history(limit: int = Query(10, ge=1, le=100), user_id: int = Depends(current_user)):
    return {"items": db.get_user_history(user_id, limit=limit)}


@app.get("/audio/{audio_hash}")
def audio(audio_hash: str, _: int = Depends(current_user)):
    # Strict allow-list: blocks path traversal like ../../etc/passwd
    if not HASH_RE.match(audio_hash):
        raise HTTPException(status_code=400, detail="Invalid audio id")
    path = os.path.join(tts.TTS_CACHE_DIR, f"{audio_hash}.mp3")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg")
