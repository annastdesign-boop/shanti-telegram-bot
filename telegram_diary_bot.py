import os
import json
import logging
import math
import io
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
from anthropic import Anthropic
from tavily import TavilyClient
from pydub import AudioSegment
from pyrogram import Client as PyroClient
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext,
)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Environment ---
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]
ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID")
USER_TIMEZONE = os.environ.get("USER_TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(USER_TIMEZONE)

WHISPER_MAX_BYTES = 24 * 1024 * 1024
CHUNK_DURATION_MS = 10 * 60 * 1000
BOT_API_LIMIT = 20 * 1024 * 1024

# --- Clients ---
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --- Pyrogram ---
pyro_app = PyroClient(
    name="shanti_dl",
    api_id=TELEGRAM_API_ID,
    api_hash=TELEGRAM_API_HASH,
    bot_token=TELEGRAM_TOKEN,
    workdir="data",
    no_updates=True,
)
pyro_ready = False


async def start_pyro():
    global pyro_ready
    if not pyro_ready:
        try:
            await pyro_app.start()
            pyro_ready = True
            me = await pyro_app.get_me()
            logger.info(f"Pyrogram started: @{me.username}")
        except Exception as e:
            logger.error(f"Pyrogram start failed: {e}")
            raise


async def stop_pyro():
    global pyro_ready
    if pyro_ready:
        try:
            await pyro_app.stop()
            pyro_ready = False
        except Exception:
            pass


# --- Storage ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)
PDF_DIR = Path("temp_pdf")
PDF_DIR.mkdir(exist_ok=True)
SCHEDULE_FILE = DATA_DIR / "schedule.json"
PRICE_WATCHES_FILE = DATA_DIR / "price_watches.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
CHAT_IDS_FILE = DATA_DIR / "chat_ids.json"

last_transcriptions: dict[str, dict] = {}


def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_chat_id(uid, cid):
    d = load_json(CHAT_IDS_FILE, {})
    d[str(uid)] = cid
    save_json(CHAT_IDS_FILE, d)


def get_chat_id(uid):
    return load_json(CHAT_IDS_FILE, {}).get(str(uid))


# --- Conversation ---
convos: dict[str, list[dict]] = {}
MAX_HIST = 40


def get_hist(uid):
    k = str(uid)
    if k not in convos:
        convos[k] = []
    return convos[k]


def add_msg(uid, role, content):
    h = get_hist(uid)
    h.append({"role": role, "content": content})
    if len(h) > MAX_HIST:
        convos[str(uid)] = h[-MAX_HIST:]


# --- Schedule ---
def load_schedule():
    return load_json(SCHEDULE_FILE, {})


def save_schedule(s):
    save_json(SCHEDULE_FILE, s)


def tnow():
    return datetime.now(TZ)


def today_key():
    return tnow().strftime("%Y-%m-%d")


def today_sched():
    s = load_schedule()
    items = s.get(today_key(), [])
    if not items:
        return f"No plans for today ({today_key()})."
    lines = [f"Schedule for {today_key()}:"]
    for i, it in enumerate(items, 1):
        l = f"{i}. [{it.get('time', '?')}] {it.get('task', '')}"
        if it.get("notes"):
            l += f" - {it['notes']}"
        lines.append(l)
    return "\n".join(lines)


def week_sched():
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
                lines.append(f"  {i}. [{t}] {it.get('task', '')}" if t else f"  {i}. {it.get('task', '')}")
    if not found:
        lines.append("Nothing scheduled.")
    return "\n".join(lines)


# --- Reminders ---
def load_rem():
    return load_json(REMINDERS_FILE, [])


def save_rem(r):
    save_json(REMINDERS_FILE, r)


def rem_text():
    active = [r for r in load_rem() if not r.get("sent")]
    if not active:
        return "No active reminders."
    lines = ["Reminders:"]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. [{r.get('datetime', '?')}] {r.get('message', '')}")
    return "\n".join(lines)


# --- Price watches ---
def load_pw():
    return load_json(PRICE_WATCHES_FILE, [])


def save_pw(w):
    save_json(PRICE_WATCHES_FILE, w)


def pw_text():
    w = load_pw()
    if not w:
        return "No price watches."
    lines = ["Watches:"]
    for i, x in enumerate(w, 1):
        lines.append(f"{i}. {x.get('description', '?')} - last: {x.get('last_checked', 'never')}")
    return "\n".join(lines)


# --- Tavily ---
def web_search(query):
    try:
        r = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
        t = ""
        if r.get("answer"):
            t += f"Summary: {r['answer']}\n\n"
        t += "Sources:\n"
        for i, x in enumerate(r.get("results", []), 1):
            c = x.get("content", "")[:300]
            t += f"{i}. {x.get('title', '')}\n   {x.get('url', '')}\n   {c}\n\n"
        return t
    except Exception as e:
        return f"Search failed: {e}"


SEARCH_P = """Decide if web search needed. ONE JSON only.
Yes: {"needs_search": true, "search_queries": ["q"]}
No: {"needs_search": false}
Search for: flights/events/hotels/news/weather/prices.
No search: scheduling/reminders/thoughts/conversation."""


def needs_search(msg):
    try:
        r = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=300,
            system=SEARCH_P, messages=[{"role": "user", "content": msg}])
        t = r.content[0].text.strip()
        if "{" in t:
            return json.loads(t[t.index("{"):t.rindex("}") + 1])
    except Exception:
        pass
    return {"needs_search": False}


def build_sys(uid, search_ctx=""):
    n = tnow()
    t = n.strftime("%A, %B %d, %Y")
    tm = n.strftime("%H:%M")
    tmrw = (n + timedelta(days=1)).strftime("%Y-%m-%d")
    in30 = (n + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    in2h = (n + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    ss = ""
    if search_ctx:
        ss = f"\n-- SEARCH RESULTS --\n{search_ctx}\nUse these. Cite prices/links.\n"

    return f"""You are Shanti - warm, intelligent personal assistant.
Date: {t} | Time: {tm} | TZ: {USER_TIMEZONE}

CAPABILITIES:

1. SCHEDULING:
   ```SCHEDULE_ADD
   {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "desc", "notes": "opt"}}
Tomorrow = {tmrw}

REMINDERS (remind/notify/alert):
```REMINDER_ADD
{{"datetime": "YYYY-MM-DD HH:MM", "message": "what"}}
```
In 30min = {in30} | In 2h = {in2h}
ALWAYS include REMINDER_ADD for reminders.

THOUGHTS: dumps -> themes, actions, ideas, emotional check-in.

AUDIO: Short -> brain dump. Long -> summary, actions, quotes.

SEARCH: Live results when provided. NEVER say cant search.

PDF EXPORT: If user asks for PDF/document/file of summary or transcription:
```PDF_REQUEST
{{"type": "summary|transcription|both", "title": "descriptive title"}}
```

PRICE WATCH:
```PRICE_WATCH
{{"description": "what", "search_query": "query", "frequency": "daily|weekly"}}
```

EDIT:
```SCHEDULE_REMOVE
{{"date": "YYYY-MM-DD", "task_keyword": "kw"}}
```
```SCHEDULE_EDIT
{{"date": "YYYY-MM-DD", "task_keyword": "kw", "new_time": "HH:MM", "new_task": "desc"}}
```
```REMINDER_REMOVE
{{"keyword": "kw"}}
```

TODAY:
{today_sched()}

REMINDERS:
{rem_text()}

WATCHES:
{pw_text()}

{ss}

RULES:
Never say As an AI. Never say cant search. No generic responses.
Schedule -> SCHEDULE_ADD. Reminder -> REMINDER_ADD. PDF -> PDF_REQUEST.
Valid JSON only. Warm, witty, concise. Emoji sparingly."""


def ask_claude(uid, msg):
    sd = needs_search(msg)
    sc = ""
    if sd.get("needs_search"):
        parts = [f"Search: {q}\n{web_search(q)}" for q in sd.get("search_queries", [])[:3]]
        sc = "\n---\n".join(parts)
    add_msg(uid, "user", msg)
    try:
        r = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096,
            system=build_sys(uid, sc), messages=get_hist(uid))
        t = r.content[0].text
        add_msg(uid, "assistant", t)
        proc_sched(t)
        proc_pw(t)
        rd = proc_rem(t, uid)
        pdf_req = proc_pdf_request(t)
        return cln(t), rd, pdf_req
    except Exception as e:
        logger.error(f"Claude: {e}")
        return f"Error: {str(e)[:200]}", None, None


def proc_sched(text):
    s = load_schedule()
    ch = False
    if "SCHEDULE_ADD" in text:
        try:
            d = json.loads(text.split("SCHEDULE_ADD")[1].split("```")[0].strip())
            s.setdefault(d["date"], []).append(
                {"time": d.get("time", ""), "task": d.get("task", ""), "notes": d.get("notes", "")})
            s[d["date"]].sort(key=lambda x: x.get("time", "99"))
            ch = True
        except Exception as e:
            logger.error(f"SA: {e}")
    if "SCHEDULE_REMOVE" in text:
        try:
            d = json.loads(text.split("SCHEDULE_REMOVE")[1].split("```")[0].strip())
            dk = d["date"]
            kw = d.get("task_keyword", "").lower()
            if dk in s:
                b = len(s[dk])
                s[dk] = [i for i in s[dk] if kw not in i.get("task", "").lower()]
                ch = len(s[dk]) < b
        except Exception as e:
            logger.error(f"SR: {e}")
    if "SCHEDULE_EDIT" in text:
        try:
            d = json.loads(text.split("SCHEDULE_EDIT")[1].split("```")[0].strip())
            dk = d["date"]
            kw = d.get("task_keyword", "").lower()
            if dk in s:
                for it in s[dk]:
                    if kw in it.get("task", "").lower():
                        if "new_time" in d:
                            it["time"] = d["new_time"]
                        if "new_task" in d:
                            it["task"] = d["new_task"]
                        ch = True
                        break
        except Exception as e:
            logger.error(f"SE: {e}")
    if ch:
        save_schedule(s)


def proc_rem(text, uid):
    rd = None
    if "REMINDER_ADD" in text:
        try:
            d = json.loads(text.split("REMINDER_ADD")[1].split("```")[0].strip())
            ds = d.get("datetime", "")
            m = d.get("message", "Reminder!")
            rdt = datetime.strptime(ds, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            rems = load_rem()
            rems.append({"datetime": ds, "message": m, "user_id": uid, "sent": False,
                         "created": tnow().isoformat()})
            save_rem(rems)
            rd = {"datetime": rdt, "message": m, "user_id": uid}
        except Exception as e:
            logger.error(f"RA: {e}")
    if "REMINDER_REMOVE" in text:
        try:
            kw = json.loads(
                text.split("REMINDER_REMOVE")[1].split("```")[0].strip()
            ).get("keyword", "").lower()
            rems = load_rem()
            b = len(rems)
            rems = [r for r in rems if kw not in r.get("message", "").lower() or r.get("sent")]
            if len(rems) < b:
                save_rem(rems)
        except Exception as e:
            logger.error(f"RR: {e}")
    return rd


def proc_pw(text):
    if "PRICE_WATCH" not in text:
        return
    try:
        d = json.loads(text.split("PRICE_WATCH")[1].split("```")[0].strip())
        w = load_pw()
        w.append({"description": d.get("description", ""), "search_query": d.get("search_query", ""),
                  "frequency": d.get("frequency", "daily"), "created": tnow().isoformat(),
                  "last_checked": "never", "last_result": ""})
        save_pw(w)
    except Exception as e:
        logger.error(f"PW: {e}")


def proc_pdf_request(text):
    if "PDF_REQUEST" not in text:
        return None
    try:
        block = text.split("PDF_REQUEST")[1].split("```")[0].strip()
        return json.loads(block)
    except Exception as e:
        logger.error(f"PDF_REQ: {e}")
        return None


def cln(text):
    c = text
    for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT",
                "REMINDER_ADD", "REMINDER_REMOVE", "PRICE_WATCH", "PDF_REQUEST"]:
        while f"```{tag}" in c:
            try:
                bf = c.split(f"```{tag}")[0]
                af = c.split(f"```{tag}")[1].split("```", 1)
                c = bf + (af[1] if len(af) > 1 else "")
            except IndexError:
                break
    while "\n\n\n" in c:
        c = c.replace("\n\n\n", "\n\n")
    return c.strip()


# --- PDF Generation ---
def generate_pdf(title, sections):
    path = os.path.join(str(PDF_DIR), f"shanti_{tnow().strftime('%Y%m%d_%H%M%S')}.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=12, alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=14, spaceAfter=6, spaceBefore=12,
        textColor="#333333",
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=11, leading=16, spaceAfter=8,
        alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"],
        fontSize=9, textColor="#888888", alignment=TA_CENTER,
        spaceAfter=20,
    )
    story = []
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(safe_title, title_style))
    story.append(Paragraph(
        f"Generated by Shanti | {tnow().strftime('%B %d, %Y at %H:%M')}",
        meta_style))
    story.append(Spacer(1, 10))
    for section in sections:
        heading = section.get("heading", "")
        body = section.get("body", "")
        if heading:
            safe_h = heading.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe_h, heading_style))
        if body:
            paragraphs = body.split("\n")
            for para in paragraphs:
                para = para.strip()
                if not para:
                    story.append(Spacer(1, 6))
                    continue
                safe = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                while "**" in safe:
                    safe = safe.replace("**", "<b>", 1)
                    if "**" in safe:
                        safe = safe.replace("**", "</b>", 1)
                    else:
                        safe += "</b>"
                if safe.startswith("- ") or safe.startswith("* "):
                    safe = f"&#8226; {safe[2:]}"
                try:
                    story.append(Paragraph(safe, body_style))
                except Exception:
                    plain = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Paragraph(plain, body_style))

    doc.build(story)
    logger.info(f"PDF generated: {path}")
    return path


# --- Audio helpers ---
AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".webm", ".aac", ".wma", ".mpga", ".opus"}
AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/ogg", "audio/wav", "audio/flac",
    "audio/webm", "audio/aac", "audio/x-m4a", "audio/m4a", "audio/opus",
    "audio/x-wav", "audio/mpeg3", "video/mp4", "audio/mp4a-latm",
}


def is_audio(msg):
    if msg.voice:
        return True
    if msg.audio:
        return True
    if msg.video_note:
        return True
    if msg.document:
        doc = msg.document
        mime = (doc.mime_type or "").lower()
        name = (doc.file_name or "").lower()
        if mime in AUDIO_MIMES or mime.startswith("audio/"):
            return True
        if any(name.endswith(e) for e in AUDIO_EXTS):
            return True
    return False


def get_audio_info(msg):
    if msg.voice:
        v = msg.voice
        return v.file_id, v.mime_type or "audio/ogg", "voice.ogg", v.file_size or 0
    if msg.audio:
        a = msg.audio
        return a.file_id, a.mime_type or "audio/mpeg", a.file_name or "audio.mp3", a.file_size or 0
    if msg.video_note:
        vn = msg.video_note
        return vn.file_id, "video/mp4", "videonote.mp4", vn.file_size or 0
    if msg.document:
        d = msg.document
        return d.file_id, d.mime_type or "", d.file_name or "file", d.file_size or 0
    return None, None, None, 0


def get_ext(mime="", name=""):
    if name:
        for e in AUDIO_EXTS:
            if name.lower().endswith(e):
                return e
    m = {
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "audio/aac": ".aac",
        "audio/ogg": ".ogg", "audio/opus": ".ogg", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/webm": ".webm",
        "video/mp4": ".mp4", "audio/mpeg3": ".mp3",
    }
    return m.get((mime or "").lower(), ".ogg")


def fmt(b):
    if b > 1024 ** 3:
        return f"{b / 1024 ** 3:.1f}GB"
    if b > 1024 ** 2:
        return f"{b / 1024 ** 2:.1f}MB"
    return f"{b / 1024:.0f}KB"


def dur_text(ms):
    s = ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# --- Download functions ---
async def download_small(bot, file_id, dest):
    try:
        f = await bot.get_file(file_id)
        await f.download_to_drive(dest)
        sz = os.path.getsize(dest)
        logger.info(f"Bot API download OK: {fmt(sz)}")
        return sz > 100
    except Exception as e:
        logger.warning(f"Bot API download failed: {e}")
        return False


async def download_pyro(chat_id, message_id, dest):
    try:
        if not pyro_ready:
            await start_pyro()
        logger.info(f"Pyrogram downloading msg {message_id} from chat {chat_id}")
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
            if isinstance(result, str) and result != dest:
                if os.path.exists(result):
                    os.rename(result, dest)
            if os.path.exists(dest):
                sz = os.path.getsize(dest)
                logger.info(f"Pyrogram download OK: {fmt(sz)}")
                return sz > 100
        logger.error(f"Pyrogram download returned: {result}")
        return False
    except Exception as e:
        logger.error(f"Pyrogram error: {type(e).__name__}: {e}")
        return False


async def download_audio_file(bot, file_id, dest, chat_id, message_id, file_size):
    logger.info(f"Downloading: {fmt(file_size)}, file_id={file_id[:20]}...")
    if file_size > BOT_API_LIMIT:
        logger.info(f"Large file ({fmt(file_size)}), using Pyrogram")
        ok = await download_pyro(chat_id, message_id, dest)
        if ok:
            return True
        logger.warning("Pyrogram failed, trying Bot API...")
        return await download_small(bot, file_id, dest)
    ok = await download_small(bot, file_id, dest)
    if ok:
        return True
    logger.info("Bot API failed, trying Pyrogram...")
    return await download_pyro(chat_id, message_id, dest)


# --- Chunking and transcription ---
def split_audio(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    fm = {
        "mp3": "mp3", "m4a": "mp4", "mp4": "mp4", "wav": "wav", "ogg": "ogg",
        "flac": "flac", "webm": "webm", "aac": "aac", "opus": "ogg",
    }.get(ext, "mp3")
    try:
        logger.info(f"Loading audio with format={fm}...")
        audio = AudioSegment.from_file(path, format=fm)
    except Exception as e1:
        logger.warning(f"Format {fm} failed: {e1}, trying auto-detect...")
        try:
            audio = AudioSegment.from_file(path)
        except Exception as e2:
            logger.error(f"Cannot load audio: {e2}")
            raise

    d = len(audio)
    fs = os.path.getsize(path)
    logger.info(f"Audio loaded OK: {dur_text(d)}, {fmt(fs)}")

    if fs <= WHISPER_MAX_BYTES and d <= CHUNK_DURATION_MS:
        logger.info("File small enough, no splitting needed")
        return [path]

    n = math.ceil(d / CHUNK_DURATION_MS)
    logger.info(f"Splitting {dur_text(d)} into {n} chunks of ~10min each")
    paths = []
    ts = datetime.now().strftime("%H%M%S")

    for i in range(n):
        s = i * CHUNK_DURATION_MS
        e = min((i + 1) * CHUNK_DURATION_MS, d)
        p = os.path.join(str(TEMP_DIR), f"chunk_{ts}_{i:03d}.mp3")
        logger.info(f"Exporting chunk {i+1}/{n}: {dur_text(e-s)}...")
        audio[s:e].export(p, format="mp3", bitrate="128k")
        cs = os.path.getsize(p)
        logger.info(f"Chunk {i+1} exported: {fmt(cs)}")

        if cs > WHISPER_MAX_BYTES:
            logger.warning(f"Chunk {i+1} too large ({fmt(cs)}), sub-splitting...")
            os.unlink(p)
            sd = CHUNK_DURATION_MS // 3
            for j in range(3):
                ss = s + j * sd
                se = min(s + (j + 1) * sd, e)
                if ss >= e:
                    break
                sp = os.path.join(str(TEMP_DIR), f"chunk_{ts}_{i:03d}_{j}.mp3")
                audio[ss:se].export(sp, format="mp3", bitrate="96k")
                logger.info(f"Sub-chunk {j+1}: {fmt(os.path.getsize(sp))}")
                paths.append(sp)
        else:
            paths.append(p)

    logger.info(f"Split complete: {len(paths)} chunks ready")
    return paths


async def full_transcribe(path, status_cb=None):
    chunks = []
    created = False
    try:
        fs = os.path.getsize(path)
        logger.info(f"Starting transcription: {fmt(fs)}")

        if fs <= WHISPER_MAX_BYTES:
            logger.info("Small file, direct Whisper...")
            r = await whisper_transcribe(path)
            if r:
                logger.info(f"Direct transcription OK: {len(r)} chars")
                return r
            logger.warning("Direct transcription failed, will try chunking")

        if status_cb:
            await status_cb("Splitting audio into chunks...")

        chunks = split_audio(path)
        created = len(chunks) > 1 or chunks[0] != path

        if created and status_cb:
            await status_cb(f"{len(chunks)} chunks ready. Starting transcription...")

        parts = []
        for i, cp in enumerate(chunks):
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)}: {fmt(os.path.getsize(cp))}")
            if status_cb and len(chunks) > 1:
                await status_cb(f"Transcribing chunk {i + 1}/{len(chunks)}...")
            t = await whisper_transcribe(cp)
            if t:
                parts.append(t.strip())
                logger.info(f"Chunk {i+1} OK: {len(t)} chars")
            else:
                parts.append(f"[Chunk {i + 1} failed]")
                logger.error(f"Chunk {i+1} transcription failed")

        result = "\n\n".join(parts)
        logger.info(f"Full transcription complete: {len(result)} chars from {len(chunks)} chunks")
        return result

    except Exception as e:
        logger.error(f"Transcription error: {type(e).__name__}: {e}")
        return None

    finally:
        if created:
            for cp in chunks:
                if cp != path:
                    try:
                        os.unlink(cp)
                    except Exception:
                        pass


async def whisper_transcribe(path):
    try:
        with open(path, "rb") as f:
            return openai_client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text")
    except Exception as e:
        logger.error(f"Whisper: {e}")
        return None


async def full_transcribe(path, status_cb=None):
    chunks = []
    created = False
    try:
        if os.path.getsize(path) <= WHISPER_MAX_BYTES:
            r = await whisper_transcribe(path)
            if r:
                return r
        if status_cb:
            await status_cb("Splitting audio into chunks...")
        chunks = split_audio(path)
        created = len(chunks) > 1 or chunks[0] != path
        if created and status_cb:
            await status_cb(f"{len(chunks)} chunks. Transcribing...")
        parts = []
        for i, cp in enumerate(chunks):
            if status_cb and len(chunks) > 1:
                await status_cb(f"Chunk {i + 1}/{len(chunks)}...")
            t = await whisper_transcribe(cp)
            parts.append(t.strip() if t else f"[Chunk {i + 1} failed]")
        return "\n\n".join(parts)
    finally:
        if created:
            for cp in chunks:
                if cp != path:
                    try:
                        os.unlink(cp)
                    except Exception:
                        pass


def is_allowed(uid):
    return ALLOWED_USER_ID is None or str(uid) == str(ALLOWED_USER_ID)


# --- Reminder jobs ---
async def fire_rem(ctx):
    d = ctx.job.data
    try:
        await ctx.bot.send_message(chat_id=d["chat_id"], text=f"REMINDER\n\n{d['message']}")
    except Exception as e:
        logger.error(f"Rem: {e}")
    rems = load_rem()
    for r in rems:
        if r.get("message") == d["message"] and r.get("user_id") == d["user_id"] and not r.get("sent"):
            r["sent"] = True
            break
    save_rem(rems)


def sched_rem(app, data, cid):
    delay = max((data["datetime"] - tnow()).total_seconds(), 5)
    app.job_queue.run_once(
        fire_rem, when=delay,
        data={"chat_id": cid, "message": data["message"], "user_id": data["user_id"]},
        name=f"r_{data['user_id']}_{data['message'][:20]}")


async def check_rem_bg(ctx):
    rems = load_rem()
    n = tnow()
    ch = False
    for r in rems:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except Exception:
            continue
        if rdt <= n:
            cid = get_chat_id(r.get("user_id"))
            if cid:
                try:
                    await ctx.bot.send_message(chat_id=cid, text=f"REMINDER\n\n{r.get('message', '!')}")
                    r["sent"] = True
                    ch = True
                except Exception:
                    pass
    if ch:
        save_rem(rems)


async def send_long(upd, text):
    for i in range(0, max(len(text), 1), 4000):
        await upd.message.reply_text(text[i:i + 4000])


async def maybe_send_pdf(update, pdf_req, uid):
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
        hist = get_hist(uid)
        for msg in reversed(hist):
            if msg["role"] == "assistant":
                sections.append({"heading": "Summary", "body": msg["content"]})
                break
    if not sections:
        await update.message.reply_text("No content for PDF.")
        return
    try:
        path = generate_pdf(title, sections)
        with open(path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"{title.replace(' ', '_')}.pdf",
                caption=f"PDF: {title}")
        os.unlink(path)
    except Exception as e:
        logger.error(f"PDF error: {e}")
        await update.message.reply_text(f"PDF failed: {str(e)[:200]}")


# --- Telegram Handlers ---
async def cmd_start(u, c):
    if not is_allowed(u.effective_user.id):
        return await u.message.reply_text("Private bot.")
    save_chat_id(u.effective_user.id, u.effective_chat.id)
    convos[str(u.effective_user.id)] = []
    await u.message.reply_text(
        "Hey! I'm Shanti.\n\n"
        "- Schedule naturally\n- Reminders\n- Structure thoughts\n"
        "- Audio ANY size - even 500MB+!\n- Hours-long recordings\n"
        "- PDF summaries - just ask!\n"
        "- Flights, events, prices\n- Price tracking\n\n"
        "/today /week /reminders /search /watches\n"
        "/checkprices /pdf /clear /clearschedule /clearreminders\n\nJust talk!")


async def cmd_today(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_chat_id(u.effective_user.id, u.effective_chat.id)
    await u.message.reply_text(today_sched())


async def cmd_week(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_chat_id(u.effective_user.id, u.effective_chat.id)
    await u.message.reply_text(week_sched())


async def cmd_rem_list(u, c):
    if not is_allowed(u.effective_user.id):
        return
    await u.message.reply_text(rem_text())


async def cmd_clear(u, c):
    if not is_allowed(u.effective_user.id):
        return
    convos[str(u.effective_user.id)] = []
    await u.message.reply_text("Cleared!")


async def cmd_clr_sched(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_schedule({})
    await u.message.reply_text("Schedule cleared.")


async def cmd_clr_rem(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_rem([])
    for j in c.application.job_queue.jobs():
        if hasattr(j, "name") and j.name and j.name.startswith("r_"):
            j.schedule_removal()
    await u.message.reply_text("Reminders cleared.")


async def cmd_search(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_chat_id(u.effective_user.id, u.effective_chat.id)
    q = " ".join(c.args) if c.args else ""
    if not q:
        return await u.message.reply_text("/search <query>")
    await u.message.chat.send_action("typing")
    r, _, _ = ask_claude(u.effective_user.id, f"Search \"{q}\":\n{web_search(q)}")
    await send_long(u, r)


async def cmd_watches(u, c):
    if not is_allowed(u.effective_user.id):
        return
    await u.message.reply_text(pw_text())


async def cmd_checkprices(u, c):
    if not is_allowed(u.effective_user.id):
        return
    ws = load_pw()
    if not ws:
        return await u.message.reply_text("No watches.")
    await u.message.reply_text(f"Checking {len(ws)}...")
    await u.message.chat.send_action("typing")
    res = []
    for w in ws:
        r = web_search(w.get("search_query", w.get("description", "")))
        w["last_checked"] = tnow().isoformat()
        w["last_result"] = r[:500]
        res.append(f"{w['description']}:\n{r}")
    save_pw(ws)
    r, _, _ = ask_claude(u.effective_user.id, "Price results:\n" + "\n---\n".join(res))
    await send_long(u, r)


async def cmd_clr_watches(u, c):
    if not is_allowed(u.effective_user.id):
        return
    save_pw([])
    await u.message.reply_text("Cleared.")


async def cmd_pdf(u, c):
    if not is_allowed(u.effective_user.id):
        return
    uid = u.effective_user.id
    data = last_transcriptions.get(str(uid), {})
    if not data:
        await u.message.reply_text("No transcription to export. Send audio first!")
        return
    await u.message.reply_text("Generating PDF...")
    sections = []
    if data.get("transcript"):
        sections.append({"heading": "Transcription", "body": data["transcript"]})
    if data.get("summary"):
        sections.append({"heading": "Summary", "body": data["summary"]})
    title = data.get("title", "Audio Transcription")
    try:
        path = generate_pdf(title, sections)
        with open(path, "rb") as f:
            await u.message.reply_document(
                document=f,
                filename=f"{title.replace(' ', '_')}.pdf",
                caption=f"PDF: {title}")
        os.unlink(path)
    except Exception as e:
        await u.message.reply_text(f"PDF failed: {str(e)[:200]}")


async def handle_text(u, c):
    if not is_allowed(u.effective_user.id):
        return
    uid = u.effective_user.id
    save_chat_id(uid, u.effective_chat.id)
    await u.message.chat.send_action("typing")
    r, rd, pdf_req = ask_claude(uid, u.message.text)
    if rd:
        sched_rem(c.application, rd, u.effective_chat.id)
    await send_long(u, r)
    if pdf_req:
        await maybe_send_pdf(u, pdf_req, uid)


async def handle_audio(u, c):
    if not is_allowed(u.effective_user.id):
        return
    if not is_audio(u.message):
        return
    uid = u.effective_user.id
    cid = u.effective_chat.id
    mid = u.message.message_id
    save_chat_id(uid, cid)

    file_id, mime, fname, fsize = get_audio_info(u.message)
    if not file_id:
        logger.error("No file_id found in audio message")
        return

    logger.info(f"AUDIO from {uid}: {fname} | {mime} | {fmt(fsize)} | msg={mid}")
    await u.message.reply_text(f"Audio received ({fmt(fsize)}). Downloading...")
    await u.message.chat.send_action("typing")

    ext = get_ext(mime, fname)
    tmp = os.path.join(str(TEMP_DIR), f"a_{uid}_{mid}{ext}")

    try:
        ok = await download_audio_file(
            bot=c.bot, file_id=file_id, dest=tmp,
            chat_id=cid, message_id=mid, file_size=fsize)

        if not ok:
            await u.message.reply_text(
                "Download failed. Please try:\n"
                "1. Send the file again\n"
                "2. Try a different format\n"
                "3. Send as a Telegram voice note")
            return

        actual = os.path.getsize(tmp)
        if actual < 100:
            await u.message.reply_text("File appears empty. Try again.")
            return

        logger.info(f"Downloaded: {fmt(actual)}")
        await u.message.reply_text(f"Downloaded ({fmt(actual)}). Transcribing...")

        async def status(msg):
            try:
                await u.message.reply_text(msg)
                await u.message.chat.send_action("typing")
            except Exception:
                pass

        transcript = await full_transcribe(tmp, status_cb=status)
        if not transcript:
            await u.message.reply_text("Transcription failed. Try a different format.")
            return

        logger.info(f"Transcribed: {len(transcript)} chars")

        if len(transcript) > 5000:
            vm = (f"[LONG AUDIO - {len(transcript)} chars]\n{transcript}\n[END]\n\n"
                  "Provide: 1) Summary 2) Action items 3) Key details 4) Scheduling items")
        else:
            vm = f"[VOICE NOTE]\n{transcript}\n[END]\n\nStructure and respond."

        await u.message.chat.send_action("typing")
        response, rd, pdf_req = ask_claude(uid, vm)

        last_transcriptions[str(uid)] = {
            "transcript": transcript,
            "summary": response,
            "title": f"Transcription - {fname}",
            "timestamp": tnow().isoformat(),
        }

        if rd:
            sched_rem(c.application, rd, cid)

        if len(transcript) > 3000:
            await u.message.reply_text("TRANSCRIPTION:")
            for i in range(0, len(transcript), 4000):
                await u.message.reply_text(transcript[i:i + 4000])
            await u.message.reply_text("---\nSUMMARY:")
            await send_long(u, response)
        else:
            await send_long(u, f"Transcription:\n{transcript}\n\n---\n\n{response}")

        await u.message.reply_text("Want this as PDF? Say 'send as PDF' or use /pdf")

        if pdf_req:
            await maybe_send_pdf(u, pdf_req, uid)

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


async def handle_doc(u, c):
    if not is_allowed(u.effective_user.id):
        return
    if u.message.document and is_audio(u.message):
        logger.info(f"Audio document: {u.message.document.file_name} ({u.message.document.mime_type})")
        await handle_audio(u, c)


# --- Startup and shutdown ---
async def post_init(app):
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
        await start_pyro()
    except Exception as e:
        logger.error(f"Pyrogram failed: {e} - large files won't work")

    rems = load_rem()
    n = tnow()
    cnt = 0
    for r in rems:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except Exception:
            continue
        cid = get_chat_id(r.get("user_id"))
        if not cid:
            continue
        app.job_queue.run_once(
            fire_rem, when=max((rdt - n).total_seconds(), 5),
            data={"chat_id": cid, "message": r.get("message", "!"), "user_id": r.get("user_id")},
            name=f"r_{r.get('user_id')}_{r.get('message', '')[:20]}")
        cnt += 1
    if cnt:
        logger.info(f"Rescheduled {cnt} reminders")
    app.job_queue.run_repeating(check_rem_bg, interval=60, first=10)
    logger.info("Bot ready!")


async def post_shutdown(app):
    await stop_pyro()
    logger.info("Bot shutdown.")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("reminders", cmd_rem_list))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("clearschedule", cmd_clr_sched))
    app.add_handler(CommandHandler("clearreminders", cmd_clr_rem))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("watches", cmd_watches))
    app.add_handler(CommandHandler("checkprices", cmd_checkprices))
    app.add_handler(CommandHandler("clearwatches", cmd_clr_watches))
    app.add_handler(CommandHandler("pdf", cmd_pdf))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE, handle_audio))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_doc))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Starting Shanti...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
