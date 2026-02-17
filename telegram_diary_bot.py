#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shanti Telegram Bot — Personal Assistant + Diary (Railway-ready)

Fixes in THIS version:
- ✅ add missing `import asyncio` (fixes NameError)
- ✅ add global error handler + register it (removes "No error handlers are registered")
"""

import os
import re
import io
import json
import math
import asyncio  # ✅ FIX #1: needed for asyncio.to_thread(...)
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional, Any, Dict, List, Tuple

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from anthropic import Anthropic
from tavily import TavilyClient
from pyrogram import Client as PyroClient

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from openai import OpenAI


# ---------------- Logging ----------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("shanti")


# ---------------- Environment ----------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
TELEGRAM_API_ID = int(os.environ.get("TELEGRAM_API_ID", "0") or "0")
TELEGRAM_API_HASH = os.environ.get("TELEGRAM_API_HASH", "").strip()

ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID")
USER_TIMEZONE = os.environ.get("USER_TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(USER_TIMEZONE)

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip() or "claude-sonnet-4-5"

WHISPER_MAX_BYTES = 24 * 1024 * 1024
BOT_API_LIMIT = 20 * 1024 * 1024
CHUNK_SEC = 10 * 60  # 10 minutes


# ---------------- Clients ----------------
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# ---------------- Storage ----------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

PDF_DIR = Path("temp_pdf")
PDF_DIR.mkdir(exist_ok=True)

SCHEDULE_FILE = DATA_DIR / "schedule.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
PRICE_WATCHES_FILE = DATA_DIR / "price_watches.json"
CHAT_IDS_FILE = DATA_DIR / "chat_ids.json"

last_transcriptions: Dict[str, Dict[str, Any]] = {}


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"load_json failed {path}: {e}")
    return default


def save_json(path: Path, data):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_chat_id(uid: int, cid: int):
    d = load_json(CHAT_IDS_FILE, {})
    d[str(uid)] = cid
    save_json(CHAT_IDS_FILE, d)


def get_chat_id(uid: int) -> Optional[int]:
    d = load_json(CHAT_IDS_FILE, {})
    v = d.get(str(uid))
    return int(v) if v is not None else None


def tnow() -> datetime:
    return datetime.now(TZ)


def today_key() -> str:
    return tnow().strftime("%Y-%m-%d")


# ---------------- Access control ----------------
def is_allowed(uid: int) -> bool:
    return ALLOWED_USER_ID is None or str(uid) == str(ALLOWED_USER_ID)


# ---------------- Schedule / Reminders / Watches ----------------
def load_schedule() -> Dict[str, list]:
    return load_json(SCHEDULE_FILE, {})


def save_schedule(s: Dict[str, list]):
    save_json(SCHEDULE_FILE, s)


def load_reminders() -> List[dict]:
    return load_json(REMINDERS_FILE, [])


def save_reminders(r: List[dict]):
    save_json(REMINDERS_FILE, r)


def load_watches() -> List[dict]:
    return load_json(PRICE_WATCHES_FILE, [])


def save_watches(w: List[dict]):
    save_json(PRICE_WATCHES_FILE, w)


def today_sched_text() -> str:
    s = load_schedule()
    items = s.get(today_key(), [])
    if not items:
        return f"No plans for today ({today_key()})."
    lines = [f"Schedule for {today_key()}:"]
    for i, it in enumerate(items, 1):
        t = it.get("time", "")
        task = it.get("task", "")
        notes = it.get("notes", "")
        line = f"{i}. [{t or '?'}] {task}"
        if notes:
            line += f" — {notes}"
        lines.append(line)
    return "\n".join(lines)


def week_sched_text() -> str:
    s = load_schedule()
    lines = ["This week:"]
    found = False
    for d in range(7):
        day = tnow() + timedelta(days=d)
        k = day.strftime("%Y-%m-%d")
        items = s.get(k, [])
        if items:
            found = True
            lines.append(f"\n{day.strftime('%A %b %d')}:")
            for i, it in enumerate(items, 1):
                t = it.get("time", "")
                task = it.get("task", "")
                lines.append(f"  {i}. [{t}] {task}" if t else f"  {i}. {task}")
    if not found:
        lines.append("Nothing scheduled.")
    return "\n".join(lines)


def reminders_text() -> str:
    active = [r for r in load_reminders() if not r.get("sent")]
    if not active:
        return "No active reminders."
    lines = ["Reminders:"]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. [{r.get('datetime', '?')}] {r.get('message', '')}")
    return "\n".join(lines)


def watches_text() -> str:
    w = load_watches()
    if not w:
        return "No price watches."
    lines = ["Watches:"]
    for i, x in enumerate(w, 1):
        lines.append(f"{i}. {x.get('description', '?')} — last: {x.get('last_checked', 'never')}")
    return "\n".join(lines)


# ---------------- Tavily search ----------------
def web_search(query: str) -> str:
    if not tavily_client:
        return "Search is disabled (TAVILY_API_KEY not set)."
    try:
        r = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        out = ""
        if r.get("answer"):
            out += f"Summary: {r['answer']}\n\n"
        out += "Sources:\n"
        for i, x in enumerate(r.get("results", []), 1):
            c = (x.get("content", "") or "")[:320]
            out += f"{i}. {x.get('title', '')}\n   {x.get('url', '')}\n   {c}\n\n"
        return out.strip()
    except Exception as e:
        return f"Search failed: {e}"


def should_search_heuristic(text: str) -> bool:
    t = text.lower()
    triggers = [
        "flight", "flights", "ticket", "tickets", "cheap", "price", "prices", "cost",
        "hotel", "booking", "airbnb", "train", "bus", "schedule", "timetable",
        "weather", "forecast", "news", "today", "latest", "open now", "hours",
        "best", "recommend", "compare", "top", "deadline", "law", "visa",
    ]
    return any(k in t for k in triggers)


# ---------------- PDF generation ----------------
def generate_pdf(title: str, sections: List[Dict[str, str]]) -> str:
    path = os.path.join(str(PDF_DIR), f"shanti_{tnow().strftime('%Y%m%d_%H%M%S')}.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=20, spaceAfter=12, alignment=TA_CENTER)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=6, spaceBefore=12, textColor="#333333")
    body_style = ParagraphStyle("CustomBody", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=8, alignment=TA_LEFT)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=9, textColor="#888888", alignment=TA_CENTER, spaceAfter=20)

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story = []
    story.append(Paragraph(esc(title), title_style))
    story.append(Paragraph(f"Generated by Shanti | {tnow().strftime('%B %d, %Y at %H:%M')} ({USER_TIMEZONE})", meta_style))
    story.append(Spacer(1, 10))

    for section in sections:
        heading = section.get("heading", "")
        body = section.get("body", "")
        if heading:
            story.append(Paragraph(esc(heading), heading_style))
        if body:
            for para in body.split("\n"):
                para = para.strip()
                if not para:
                    story.append(Spacer(1, 6))
                    continue
                safe = esc(para)
                while "**" in safe:
                    safe = safe.replace("**", "<b>", 1)
                    if "**" in safe:
                        safe = safe.replace("**", "</b>", 1)
                    else:
                        safe += "</b>"
                if safe.startswith("- ") or safe.startswith("* "):
                    safe = f"&#8226; {safe[2:]}"
                story.append(Paragraph(safe, body_style))

    doc.build(story)
    logger.info(f"PDF generated: {path}")
    return path


# ---------------- Audio detection + file info ----------------
AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".webm", ".aac", ".wma", ".mpga", ".opus", ".mov"}
AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/ogg", "audio/wav", "audio/flac",
    "audio/webm", "audio/aac", "audio/x-m4a", "audio/m4a", "audio/opus",
    "audio/x-wav", "audio/mpeg3", "video/mp4", "audio/mp4a-latm", "video/quicktime",
}


def is_audio_message(msg) -> bool:
    if msg.voice or msg.audio or msg.video_note or msg.video:
        return True
    if msg.document:
        doc = msg.document
        mime = (doc.mime_type or "").lower()
        name = (doc.file_name or "").lower()
        if mime in AUDIO_MIMES or mime.startswith("audio/") or mime.startswith("video/"):
            return True
        if any(name.endswith(e) for e in AUDIO_EXTS):
            return True
    return False


def get_media_info(msg) -> Tuple[Optional[str], str, str, int]:
    if msg.voice:
        v = msg.voice
        return v.file_id, v.mime_type or "audio/ogg", "voice.ogg", v.file_size or 0
    if msg.audio:
        a = msg.audio
        return a.file_id, a.mime_type or "audio/mpeg", a.file_name or "audio.mp3", a.file_size or 0
    if msg.video_note:
        vn = msg.video_note
        return vn.file_id, "video/mp4", "videonote.mp4", vn.file_size or 0
    if msg.video:
        vd = msg.video
        return vd.file_id, vd.mime_type or "video/mp4", "video.mp4", vd.file_size or 0
    if msg.document:
        d = msg.document
        return d.file_id, d.mime_type or "", d.file_name or "file", d.file_size or 0
    return None, "", "file", 0


def get_ext(mime: str = "", name: str = "") -> str:
    if name:
        low = name.lower()
        for e in AUDIO_EXTS:
            if low.endswith(e):
                return e
    m = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/m4a": ".m4a",
        "audio/mp4a-latm": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
    }
    return m.get((mime or "").lower(), ".dat")


def fmt_bytes(b: int) -> str:
    if b > 1024 ** 3:
        return f"{b / 1024 ** 3:.1f}GB"
    if b > 1024 ** 2:
        return f"{b / 1024 ** 2:.1f}MB"
    return f"{b / 1024:.0f}KB"


# ---------------- Pyrogram for large downloads ----------------
pyro_ready = False
pyro_app: Optional[PyroClient] = None

if TELEGRAM_API_ID and TELEGRAM_API_HASH:
    pyro_app = PyroClient(
        name="shanti_dl",
        api_id=TELEGRAM_API_ID,
        api_hash=TELEGRAM_API_HASH,
        bot_token=TELEGRAM_TOKEN,
        workdir="data",
        no_updates=True,
    )
else:
    pyro_app = None


async def start_pyro():
    global pyro_ready
    if pyro_app and not pyro_ready:
        await pyro_app.start()
        pyro_ready = True
        me = await pyro_app.get_me()
        logger.info(f"Pyrogram started: @{me.username}")


async def stop_pyro():
    global pyro_ready
    if pyro_app and pyro_ready:
        await pyro_app.stop()
        pyro_ready = False


async def download_small(bot, file_id: str, dest: str) -> bool:
    try:
        f = await bot.get_file(file_id)
        await f.download_to_drive(dest)
        sz = os.path.getsize(dest)
        logger.info(f"Bot API download OK: {fmt_bytes(sz)}")
        return sz > 100
    except Exception as e:
        logger.warning(f"Bot API download failed: {e}")
        return False


async def download_pyro(chat_id: int, message_id: int, dest: str) -> bool:
    if not pyro_app:
        return False
    try:
        if not pyro_ready:
            await start_pyro()
        msg = await pyro_app.get_messages(chat_id, message_id)
        if not msg:
            logger.error("Pyrogram: no message")
            return False
        has_media = msg.audio or msg.voice or msg.document or msg.video or msg.video_note
        if not has_media:
            logger.error("Pyrogram: message has no media")
            return False
        result = await msg.download(file_name=dest)
        if result:
            if isinstance(result, str) and result != dest and os.path.exists(result):
                os.rename(result, dest)
        if os.path.exists(dest):
            sz = os.path.getsize(dest)
            logger.info(f"Pyrogram download OK: {fmt_bytes(sz)}")
            return sz > 100
        return False
    except Exception as e:
        logger.error(f"Pyrogram error: {type(e).__name__}: {e}")
        return False


async def download_media(bot, file_id: str, dest: str, chat_id: int, message_id: int, file_size: int) -> bool:
    logger.info(f"Downloading media: {fmt_bytes(file_size)}")
    if file_size > BOT_API_LIMIT:
        ok = await download_pyro(chat_id, message_id, dest)
        if ok:
            return True
        logger.warning("Pyrogram failed; trying Bot API fallback...")
        return await download_small(bot, file_id, dest)

    ok = await download_small(bot, file_id, dest)
    if ok:
        return True
    return await download_pyro(chat_id, message_id, dest)


# ---------------- ffmpeg chunking + transcription ----------------
def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=5)
        return True
    except Exception:
        return False


def split_audio_ffmpeg(path: str) -> List[str]:
    ts = datetime.now().strftime("%H%M%S")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(result.stdout)
        duration_sec = float(info["format"]["duration"])
        logger.info(f"Audio duration: {duration_sec:.0f}s ({duration_sec/60:.1f}min)")
    except Exception as e:
        logger.error(f"ffprobe failed: {e}")
        return [path]

    if duration_sec <= CHUNK_SEC:
        out = os.path.join(str(TEMP_DIR), f"conv_{ts}.mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-i", path, "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-y", out],
                capture_output=True, timeout=180
            )
            if os.path.exists(out):
                sz = os.path.getsize(out)
                if 100 < sz <= WHISPER_MAX_BYTES:
                    logger.info(f"Converted to mp3: {fmt_bytes(sz)}")
                    return [out]
        except Exception as e:
            logger.warning(f"Conversion failed: {e}")
            return [path]

    n = math.ceil(duration_sec / CHUNK_SEC)
    logger.info(f"Splitting into {n} chunks of {CHUNK_SEC}s")

    paths: List[str] = []
    for i in range(n):
        start = i * CHUNK_SEC
        out = os.path.join(str(TEMP_DIR), f"chunk_{ts}_{i:03d}.mp3")
        try:
            cmd = [
                "ffmpeg", "-i", path,
                "-ss", str(start),
                "-t", str(CHUNK_SEC),
                "-vn",
                "-acodec", "libmp3lame",
                "-ab", "128k",
                "-y", out
            ]
            subprocess.run(cmd, capture_output=True, timeout=240)

            if os.path.exists(out):
                sz = os.path.getsize(out)
                if sz > 100:
                    if sz > WHISPER_MAX_BYTES:
                        out2 = os.path.join(str(TEMP_DIR), f"chunk_{ts}_{i:03d}_lo.mp3")
                        subprocess.run(
                            ["ffmpeg", "-i", out, "-vn", "-acodec", "libmp3lame", "-ab", "64k", "-y", out2],
                            capture_output=True, timeout=240
                        )
                        try:
                            os.unlink(out)
                        except Exception:
                            pass
                        out = out2
                        logger.info(f"Re-encoded chunk {i+1}: {fmt_bytes(os.path.getsize(out))}")
                    paths.append(out)
        except subprocess.TimeoutExpired:
            logger.error(f"Chunk {i+1} ffmpeg timeout")
        except Exception as e:
            logger.error(f"Chunk {i+1} ffmpeg error: {e}")

    return paths or [path]


async def whisper_transcribe(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            txt = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return str(txt).strip() if txt else None
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return None


async def full_transcribe(path: str, status_cb=None) -> Optional[str]:
    chunks: List[str] = []
    created = False
    try:
        fs = os.path.getsize(path)
        logger.info(f"Transcription start: {fmt_bytes(fs)}")

        if fs <= WHISPER_MAX_BYTES:
            if status_cb:
                await status_cb("Transcribing (single file)…")
            r = await whisper_transcribe(path)
            if r:
                return r
            logger.warning("Direct whisper failed; will chunk")

        if not ffmpeg_exists():
            if status_cb:
                await status_cb("ffmpeg not found; trying best-effort transcription…")
            return await whisper_transcribe(path)

        if status_cb:
            await status_cb("Splitting into chunks (ffmpeg)…")

        chunks = split_audio_ffmpeg(path)
        created = len(chunks) > 1 or chunks[0] != path

        if status_cb:
            await status_cb(f"{len(chunks)} chunk(s). Transcribing…")

        parts: List[str] = []
        for i, cp in enumerate(chunks):
            if status_cb and len(chunks) > 1:
                await status_cb(f"Transcribing chunk {i+1}/{len(chunks)}…")
            t = await whisper_transcribe(cp)
            parts.append(t.strip() if t else f"[Chunk {i+1} failed]")
        return "\n\n".join(parts).strip()

    except Exception as e:
        logger.error(f"full_transcribe error: {type(e).__name__}: {e}")
        return None

    finally:
        if created:
            for cp in chunks:
                if cp != path:
                    try:
                        os.unlink(cp)
                    except Exception:
                        pass


# ---------------- Claude tool schema + Shanti respond ----------------
def tool_schema() -> dict:
    return {
        "name": "shanti_output",
        "description": "Return the assistant reply and structured actions for schedule, reminders, watches, pdf, and searches.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reply": {"type": "string"},
                "schedule_add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM 24h, can be empty if unknown"},
                            "task": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["date", "time", "task", "notes"],
                    },
                },
                "schedule_remove": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "task_keyword": {"type": "string"},
                        },
                        "required": ["date", "task_keyword"],
                    },
                },
                "schedule_edit": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string"},
                            "task_keyword": {"type": "string"},
                            "new_time": {"type": "string"},
                            "new_task": {"type": "string"},
                            "new_notes": {"type": "string"},
                        },
                        "required": ["date", "task_keyword", "new_time", "new_task", "new_notes"],
                    },
                },
                "reminder_add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "datetime": {"type": "string", "description": "YYYY-MM-DD HH:MM (user timezone)"},
                            "message": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "normal", "low"]},
                        },
                        "required": ["datetime", "message", "priority"],
                    },
                },
                "reminder_remove": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"keyword": {"type": "string"}},
                        "required": ["keyword"],
                    },
                },
                "price_watch_add": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "description": {"type": "string"},
                            "search_query": {"type": "string"},
                            "frequency": {"type": "string", "enum": ["daily", "weekly"]},
                        },
                        "required": ["description", "search_query", "frequency"],
                    },
                },
                "price_watch_clear": {"type": "boolean"},
                "pdf_request": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": ["summary", "transcription", "both"]},
                        "title": {"type": "string"},
                    },
                    "required": ["type", "title"],
                },
                "search_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "reply", "schedule_add", "schedule_remove", "schedule_edit",
                "reminder_add", "reminder_remove",
                "price_watch_add", "price_watch_clear",
                "pdf_request", "search_queries",
            ],
        },
        "strict": True,
    }


def build_system_context(uid: int, search_ctx: str = "") -> str:
    n = tnow()
    t = n.strftime("%A, %B %d, %Y")
    tm = n.strftime("%H:%M")
    tmrw = (n + timedelta(days=1)).strftime("%Y-%m-%d")
    in30 = (n + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    in2h = (n + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")

    sc = ""
    if search_ctx:
        sc = f"\n\n--- SEARCH RESULTS ---\n{search_ctx}\nUse these facts. Mention useful links if relevant.\n"

    return f"""You are Shanti — a warm, sharp personal assistant and diary organizer.
User timezone: {USER_TIMEZONE}
Now: {t} {tm}
Tomorrow date: {tmrw}
In 30 minutes: {in30}
In 2 hours: {in2h}

Always be useful, never generic.
If user implies timing ("tomorrow", "at 5", "in 2 hours", "remind me"): create reminder_add items.
If user plans tasks/events: create schedule_add items.
If user asks for flights/prices/news/weather/events/current info: add 1-3 search_queries.

Current schedule (today):
{today_sched_text()}

Current reminders:
{reminders_text()}

Current price watches:
{watches_text()}
{sc}
"""


async def shanti_respond(uid: int, user_text: str, pre_search_ctx: str = "") -> Dict[str, Any]:
    tool = tool_schema()
    system = build_system_context(uid, pre_search_ctx)

    def _call():
        resp = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1400,
            system=system,
            messages=[{"role": "user", "content": user_text}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "shanti_output"},
        )

        out = None
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "shanti_output":
                out = getattr(block, "input", None)
                break

        if not isinstance(out, dict):
            return {
                "reply": "I got you — do you want me to log this, set reminders, or build a plan?",
                "schedule_add": [], "schedule_remove": [], "schedule_edit": [],
                "reminder_add": [], "reminder_remove": [],
                "price_watch_add": [], "price_watch_clear": False,
                "pdf_request": None, "search_queries": [],
            }
        return out

    # ✅ FIX #1 uses asyncio.to_thread safely because asyncio is imported now
    return await asyncio.to_thread(_call)  # type: ignore


# ---------------- Apply actions ----------------
def apply_schedule_actions(out: Dict[str, Any]):
    s = load_schedule()
    changed = False

    for it in out.get("schedule_add", []) or []:
        d = it["date"]
        s.setdefault(d, []).append({"time": it.get("time", ""), "task": it.get("task", ""), "notes": it.get("notes", "")})
        s[d].sort(key=lambda x: x.get("time") or "99:99")
        changed = True

    for it in out.get("schedule_remove", []) or []:
        d = it["date"]
        kw = (it.get("task_keyword") or "").lower()
        if d in s and kw:
            before = len(s[d])
            s[d] = [x for x in s[d] if kw not in (x.get("task", "").lower())]
            if len(s[d]) != before:
                changed = True

    for it in out.get("schedule_edit", []) or []:
        d = it["date"]
        kw = (it.get("task_keyword") or "").lower()
        if d in s and kw:
            for x in s[d]:
                if kw in x.get("task", "").lower():
                    x["time"] = it.get("new_time", x.get("time", ""))
                    x["task"] = it.get("new_task", x.get("task", ""))
                    x["notes"] = it.get("new_notes", x.get("notes", ""))
                    changed = True
                    break
            s[d].sort(key=lambda x: x.get("time") or "99:99")

    if changed:
        save_schedule(s)


def apply_watch_actions(out: Dict[str, Any]):
    if out.get("price_watch_clear"):
        save_watches([])
        return

    adds = out.get("price_watch_add", []) or []
    if not adds:
        return

    w = load_watches()
    for it in adds:
        w.append({
            "description": it.get("description", ""),
            "search_query": it.get("search_query", ""),
            "frequency": it.get("frequency", "daily"),
            "created": tnow().isoformat(),
            "last_checked": "never",
            "last_result": "",
        })
    save_watches(w)


def apply_reminder_actions(out: Dict[str, Any], uid: int) -> List[Dict[str, Any]]:
    rems = load_reminders()
    scheduled: List[Dict[str, Any]] = []

    for it in out.get("reminder_remove", []) or []:
        kw = (it.get("keyword") or "").lower()
        if not kw:
            continue
        before = len(rems)
        rems = [r for r in rems if (kw not in (r.get("message", "").lower()) or r.get("sent"))]
        if len(rems) != before:
            logger.info("Removed reminder(s) by keyword")

    for it in out.get("reminder_add", []) or []:
        ds = it.get("datetime", "").strip()
        msg = it.get("message", "").strip()
        pr = it.get("priority", "normal")
        if not ds or not msg:
            continue
        try:
            rdt = datetime.strptime(ds, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except Exception:
            continue
        rems.append({"datetime": ds, "message": msg, "priority": pr, "user_id": uid, "sent": False, "created": tnow().isoformat()})
        scheduled.append({"datetime": rdt, "message": msg, "user_id": uid, "priority": pr})

    save_reminders(rems)
    return scheduled


# ---------------- Reminder jobs ----------------
async def fire_reminder(ctx):
    data = ctx.job.data
    try:
        pr = data.get("priority", "normal")
        head = "⏰ Reminder" if pr != "high" else "⏰‼️ Important reminder"
        await ctx.bot.send_message(chat_id=data["chat_id"], text=f"{head}\n\n{data['message']}")
    except Exception as e:
        logger.error(f"Reminder send failed: {e}")

    rems = load_reminders()
    for r in rems:
        if r.get("user_id") == data.get("user_id") and r.get("message") == data.get("message") and not r.get("sent"):
            r["sent"] = True
            break
    save_reminders(rems)


def schedule_reminder_job(app: Application, reminder: Dict[str, Any], chat_id: int):
    delay = max((reminder["datetime"] - tnow()).total_seconds(), 5)
    app.job_queue.run_once(
        fire_reminder,
        when=delay,
        data={"chat_id": chat_id, "message": reminder["message"], "user_id": reminder["user_id"], "priority": reminder.get("priority", "normal")},
        name=f"r_{reminder['user_id']}_{reminder['message'][:20]}",
    )


async def reminder_safety_check(ctx):
    rems = load_reminders()
    n = tnow()
    changed = False
    for r in rems:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except Exception:
            continue
        if rdt <= n:
            cid = get_chat_id(int(r.get("user_id")))
            if cid:
                try:
                    pr = r.get("priority", "normal")
                    head = "⏰ Reminder" if pr != "high" else "⏰‼️ Important reminder"
                    await ctx.bot.send_message(chat_id=cid, text=f"{head}\n\n{r.get('message', '!')}")
                    r["sent"] = True
                    changed = True
                except Exception:
                    pass
    if changed:
        save_reminders(rems)


# ---------------- ✅ FIX #2: Global error handler ----------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ I crashed on that message, but I’m back now. Try again.")
    except Exception:
        pass


# ---------------- PDF send helper ----------------
async def maybe_send_pdf(update: Update, pdf_req: Optional[dict], uid: int):
    if not pdf_req:
        return
    req_type = pdf_req.get("type", "summary")
    title = pdf_req.get("title", "Shanti Summary")

    data = last_transcriptions.get(str(uid), {})
    transcript = data.get("transcript", "")
    summary = data.get("summary", "")

    sections = []
    if req_type in ("transcription", "both") and transcript:
        sections.append({"heading": "Transcription", "body": transcript})
    if req_type in ("summary", "both") and summary:
        sections.append({"heading": "Summary", "body": summary})

    if not sections:
        await update.message.reply_text("No content for PDF yet. Send audio or ask for a summary first.")
        return

    try:
        path = generate_pdf(title, sections)
        with open(path, "rb") as f:
            await update.message.reply_document(document=f, filename=f"{title.replace(' ', '_')}.pdf", caption=f"PDF: {title}")
        os.unlink(path)
    except Exception as e:
        logger.error(f"PDF error: {e}")
        await update.message.reply_text(f"PDF failed: {str(e)[:200]}")


# ---------------- Telegram handlers ----------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return await update.message.reply_text("Private bot.")
    save_chat_id(uid, update.effective_chat.id)
    await update.message.reply_text(
        "👋 Hey! I'm Shanti — your personal assistant.\n\n"
        "Send me:\n"
        "• random thoughts (I’ll structure them)\n"
        "• tasks/plans (I’ll schedule)\n"
        "• reminders (I’ll remind)\n"
        "• audio/video notes (I’ll transcribe + summarize)\n\n"
        "Commands:\n"
        "/today /week /reminders /watches /checkprices /pdf\n"
        "/clear /clearschedule /clearreminders /clearwatches\n"
    )


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_chat_id(uid, update.effective_chat.id)
    await update.message.reply_text(today_sched_text())


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_chat_id(uid, update.effective_chat.id)
    await update.message.reply_text(week_sched_text())


async def cmd_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    await update.message.reply_text(reminders_text())


async def cmd_watches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    await update.message.reply_text(watches_text())


async def cmd_checkprices(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return

    ws = load_watches()
    if not ws:
        return await update.message.reply_text("No watches.")

    await update.message.reply_text(f"Checking {len(ws)} watch(es)…")
    await update.message.chat.send_action("typing")

    results = []
    for w in ws:
        q = w.get("search_query") or w.get("description") or ""
        r = web_search(q) if q else "No query."
        w["last_checked"] = tnow().isoformat()
        w["last_result"] = r[:900]
        results.append(f"{w.get('description','(watch)')}:\n{r}")

    save_watches(ws)

    combined = "\n---\n".join(results)
    out = await shanti_respond(uid, f"Here are the latest price watch results:\n\n{combined}")
    apply_schedule_actions(out)
    apply_watch_actions(out)
    scheduled = apply_reminder_actions(out, uid)

    cid = update.effective_chat.id
    for r in scheduled:
        schedule_reminder_job(context.application, r, cid)

    await update.message.reply_text(out["reply"])
    await maybe_send_pdf(update, out.get("pdf_request"), uid)


async def cmd_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    data = last_transcriptions.get(str(uid), {})
    if not data:
        return await update.message.reply_text("No transcription to export yet. Send audio first!")
    await update.message.reply_text("Generating PDF…")
    await maybe_send_pdf(update, {"type": "both", "title": data.get("title", "Audio Transcription")}, uid)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    last_transcriptions.pop(str(uid), None)
    await update.message.reply_text("Cleared last transcription buffer.")


async def cmd_clear_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_schedule({})
    await update.message.reply_text("Schedule cleared.")


async def cmd_clear_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_reminders([])
    for j in context.application.job_queue.jobs():
        if getattr(j, "name", "") and str(j.name).startswith("r_"):
            j.schedule_removal()
    await update.message.reply_text("Reminders cleared.")


async def cmd_clear_watches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_watches([])
    await update.message.reply_text("Watches cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    save_chat_id(uid, update.effective_chat.id)

    text = (update.message.text or "").strip()
    if not text:
        return

    await update.message.chat.send_action("typing")

    search_ctx = ""
    if should_search_heuristic(text) and tavily_client:
        search_ctx = web_search(text[:220])

    out = await shanti_respond(uid, text, pre_search_ctx=search_ctx)

    queries = (out.get("search_queries") or [])[:3]
    if queries and tavily_client:
        combined = "\n\n---\n\n".join([f"Query: {q}\n{web_search(q)}" for q in queries])
        out = await shanti_respond(uid, text, pre_search_ctx=combined)

    apply_schedule_actions(out)
    apply_watch_actions(out)
    scheduled = apply_reminder_actions(out, uid)

    cid = update.effective_chat.id
    for r in scheduled:
        schedule_reminder_job(context.application, r, cid)

    await update.message.reply_text(out["reply"])
    await maybe_send_pdf(update, out.get("pdf_request"), uid)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    if not is_audio_message(update.message):
        return

    cid = update.effective_chat.id
    mid = update.message.message_id
    save_chat_id(uid, cid)

    file_id, mime, fname, fsize = get_media_info(update.message)
    if not file_id:
        return

    ext = get_ext(mime, fname)
    tmp = os.path.join(str(TEMP_DIR), f"a_{uid}_{mid}{ext}")

    await update.message.reply_text(f"📥 Received: {fname} ({fmt_bytes(fsize)}). Downloading…")
    await update.message.chat.send_action("typing")

    try:
        ok = await download_media(
            bot=context.bot,
            file_id=file_id,
            dest=tmp,
            chat_id=cid,
            message_id=mid,
            file_size=fsize,
        )
        if not ok:
            return await update.message.reply_text(
                "Download failed. Try:\n"
                "1) resend\n2) smaller file\n3) Telegram voice note\n"
            )

        actual = os.path.getsize(tmp)
        if actual < 100:
            return await update.message.reply_text("File seems empty. Try again.")

        async def status(msg: str):
            try:
                await update.message.reply_text(msg)
                await update.message.chat.send_action("typing")
            except Exception:
                pass

        await update.message.reply_text(f"🧠 Transcribing… ({fmt_bytes(actual)})")
        transcript = await full_transcribe(tmp, status_cb=status)
        if not transcript:
            return await update.message.reply_text("❌ Transcription failed. Try another format or shorter clip.")

        prompt = (
            f"[TRANSCRIPT START]\n{transcript}\n[TRANSCRIPT END]\n\n"
            "Please:\n"
            "1) Summarize\n2) Extract tasks/reminders/schedule items\n3) Structure thoughts into themes\n"
            "Be concise and actionable."
        )

        search_ctx = ""
        if should_search_heuristic(transcript) and tavily_client:
            search_ctx = web_search(transcript[:220])

        out = await shanti_respond(uid, prompt, pre_search_ctx=search_ctx)

        queries = (out.get("search_queries") or [])[:3]
        if queries and tavily_client:
            combined = "\n\n---\n\n".join([f"Query: {q}\n{web_search(q)}" for q in queries])
            out = await shanti_respond(uid, prompt, pre_search_ctx=combined)

        apply_schedule_actions(out)
        apply_watch_actions(out)
        scheduled = apply_reminder_actions(out, uid)
        for r in scheduled:
            schedule_reminder_job(context.application, r, cid)

        last_transcriptions[str(uid)] = {
            "transcript": transcript,
            "summary": out["reply"],
            "title": f"Transcription — {fname}",
            "timestamp": tnow().isoformat(),
        }

        if len(transcript) <= 2500:
            await update.message.reply_text(f"📝 Transcription:\n{transcript}\n\n—\n\n{out['reply']}")
        else:
            await update.message.reply_text("📝 Transcription is long. Sending summary + actions now.")
            await update.message.reply_text(out["reply"])
            await update.message.reply_text("If you want the full transcription + summary as a PDF, use /pdf or say “send as PDF”.")

        await maybe_send_pdf(update, out.get("pdf_request"), uid)

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ---------------- Lifecycle hooks ----------------
async def post_init(app: Application):
    for f in TEMP_DIR.glob("*"):
        try:
            os.unlink(f)
        except Exception:
            pass
    for f in PDF_DIR.glob("*"):
        try:
            os.unlink(f)
        except Exception:
            pass

    try:
        if pyro_app:
            await start_pyro()
    except Exception as e:
        logger.error(f"Pyrogram failed to start: {e} (large files may fail)")

    rems = load_reminders()
    n = tnow()
    count = 0
    for r in rems:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except Exception:
            continue
        uid = int(r.get("user_id", 0) or 0)
        cid = get_chat_id(uid)
        if not cid:
            continue
        delay = max((rdt - n).total_seconds(), 5)
        app.job_queue.run_once(
            fire_reminder,
            when=delay,
            data={"chat_id": cid, "message": r.get("message", "!"), "user_id": uid, "priority": r.get("priority", "normal")},
            name=f"r_{uid}_{(r.get('message','')[:20])}",
        )
        count += 1

    app.job_queue.run_repeating(reminder_safety_check, interval=60, first=10)
    logger.info(f"Bot ready. Rescheduled {count} reminder(s).")

    if not ffmpeg_exists():
        logger.warning("ffmpeg/ffprobe not found. Long file chunking may not work.")


async def post_shutdown(app: Application):
    try:
        await stop_pyro()
    except Exception:
        pass
    logger.info("Bot shutdown.")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # ✅ FIX #2: register error handler so PTB doesn't complain and you get a user-visible error
    app.add_error_handler(on_error)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("reminders", cmd_reminders))
    app.add_handler(CommandHandler("watches", cmd_watches))
    app.add_handler(CommandHandler("checkprices", cmd_checkprices))
    app.add_handler(CommandHandler("pdf", cmd_pdf))

    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("clearschedule", cmd_clear_schedule))
    app.add_handler(CommandHandler("clearreminders", cmd_clear_reminders))
    app.add_handler(CommandHandler("clearwatches", cmd_clear_watches))

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.VIDEO | filters.VIDEO_NOTE | filters.Document.ALL, handle_media))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Starting Shanti…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()


