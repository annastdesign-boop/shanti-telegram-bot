import os
import json
import logging
import tempfile
import math
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
from anthropic import Anthropic
from tavily import TavilyClient
from pydub import AudioSegment
from pyrogram import Client as PyroClient
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Environment variables ────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Telegram API credentials (from https://my.telegram.org/apps)
TELEGRAM_API_ID = int(os.environ["TELEGRAM_API_ID"])
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]

ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID")

USER_TIMEZONE = os.environ.get("USER_TIMEZONE", "Europe/Berlin")
TZ = ZoneInfo(USER_TIMEZONE)

# ── Limits ───────────────────────────────────────────────────────────────────
WHISPER_MAX_BYTES = 24 * 1024 * 1024
CHUNK_DURATION_MS = 10 * 60 * 1000

# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Pyrogram client for downloading large files (MTProto = no size limit)
pyro_client = PyroClient(
    "shanti_bot",
    api_id=TELEGRAM_API_ID,
    api_hash=TELEGRAM_API_HASH,
    bot_token=TELEGRAM_TOKEN,
    workdir="data",
    no_updates=True,  # We don't need pyrogram to handle updates
)

# ── Persistent storage ──────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)
SCHEDULE_FILE = DATA_DIR / "schedule.json"
PRICE_WATCHES_FILE = DATA_DIR / "price_watches.json"
REMINDERS_FILE = DATA_DIR / "reminders.json"
CHAT_IDS_FILE = DATA_DIR / "chat_ids.json"


def load_json(path: Path, default=None):
    if default is None:
        default = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_chat_id(user_id: int, chat_id: int):
    data = load_json(CHAT_IDS_FILE, {})
    data[str(user_id)] = chat_id
    save_json(CHAT_IDS_FILE, data)


def get_chat_id(user_id: int) -> int | None:
    data = load_json(CHAT_IDS_FILE, {})
    return data.get(str(user_id))


# ── Conversation history ────────────────────────────────────────────────────
user_conversations: dict[str, list[dict]] = {}
MAX_HISTORY = 40


def get_history(user_id: int) -> list[dict]:
    uid = str(user_id)
    if uid not in user_conversations:
        user_conversations[uid] = []
    return user_conversations[uid]


def append_message(user_id: int, role: str, content: str):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY:
        user_conversations[str(user_id)] = history[-MAX_HISTORY:]


# ── Schedule ─────────────────────────────────────────────────────────────────
def load_schedule() -> dict:
    return load_json(SCHEDULE_FILE, {})


def save_schedule(schedule: dict):
    save_json(SCHEDULE_FILE, schedule)


def get_today_key() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def get_now() -> datetime:
    return datetime.now(TZ)


def get_today_schedule_text() -> str:
    schedule = load_schedule()
    today = get_today_key()
    items = schedule.get(today, [])
    if not items:
        return f"No plans for today ({today})."
    lines = [f"📅 Schedule for {today}:"]
    for i, item in enumerate(items, 1):
        t = item.get("time", "?")
        task = item.get("task", "")
        notes = item.get("notes", "")
        line = f"{i}. [{t}] {task}"
        if notes:
            line += f" — {notes}"
        lines.append(line)
    return "\n".join(lines)


def get_week_schedule_text() -> str:
    schedule = load_schedule()
    today = get_now()
    lines = ["📅 This week:"]
    found = False
    for d in range(7):
        day = today + timedelta(days=d)
        key = day.strftime("%Y-%m-%d")
        items = schedule.get(key, [])
        if items:
            found = True
            lines.append(f"\n{day.strftime('%A %b %d')}:")
            for i, item in enumerate(items, 1):
                t = item.get("time", "")
                line = f"  {i}. [{t}] {item.get('task', '')}" if t else f"  {i}. {item.get('task', '')}"
                lines.append(line)
    if not found:
        lines.append("Nothing scheduled.")
    return "\n".join(lines)


# ── Reminders ────────────────────────────────────────────────────────────────
def load_reminders() -> list:
    return load_json(REMINDERS_FILE, [])


def save_reminders(reminders: list):
    save_json(REMINDERS_FILE, reminders)


def get_reminders_text() -> str:
    active = [r for r in load_reminders() if not r.get("sent")]
    if not active:
        return "No active reminders."
    lines = ["⏰ Active Reminders:"]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. [{r.get('datetime', '?')}] {r.get('message', '')}")
    return "\n".join(lines)


# ── Price watches ────────────────────────────────────────────────────────────
def load_price_watches() -> list:
    return load_json(PRICE_WATCHES_FILE, [])


def save_price_watches(watches: list):
    save_json(PRICE_WATCHES_FILE, watches)


def get_price_watches_text() -> str:
    watches = load_price_watches()
    if not watches:
        return "No active price watches."
    lines = ["👀 Price Watches:"]
    for i, w in enumerate(watches, 1):
        lines.append(f"{i}. {w.get('description', '?')} — last: {w.get('last_checked', 'never')}")
    return "\n".join(lines)


# ── Tavily ───────────────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    try:
        logger.info(f"Search: {query}")
        resp = tavily_client.search(query=query, search_depth="advanced",
                                     max_results=5, include_answer=True)
        text = ""
        if resp.get("answer"):
            text += f"Summary: {resp['answer']}\n\n"
        text += "Sources:\n"
        for i, r in enumerate(resp.get("results", []), 1):
            c = r.get("content", "")[:300]
            text += f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {c}\n\n"
        return text
    except Exception as e:
        return f"Search failed: {str(e)[:200]}"


SEARCH_PROMPT = """Decide if web search is needed. Reply with ONE JSON object only.
Search needed: {"needs_search": true, "search_queries": ["query"]}
Not needed: {"needs_search": false}
Search for: flights, events, hotels, news, weather, prices, real-world data.
No search for: scheduling, reminders, thoughts, conversation, personal data."""


def needs_search(msg: str) -> dict:
    try:
        r = anthropic_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
            system=SEARCH_PROMPT, messages=[{"role": "user", "content": msg}])
        t = r.content[0].text.strip()
        if "{" in t:
            return json.loads(t[t.index("{"):t.rindex("}") + 1])
    except Exception:
        pass
    return {"needs_search": False}


def build_system_prompt(user_id: int, search_ctx: str = "") -> str:
    now = get_now()
    today = now.strftime("%A, %B %d, %Y")
    time = now.strftime("%H:%M")
    tmrw = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    in30 = (now + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    in2h = (now + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")

    ss = ""
    if search_ctx:
        ss = f"\n── SEARCH RESULTS ──\n{search_ctx}\nUse these. Cite prices/links/dates.\n"

    return f"""You are Shanti — warm, intelligent personal assistant. Personal, context-aware, proactive.

Date: {today} | Time: {time} | TZ: {USER_TIMEZONE}

── CAPABILITIES ──
1. SCHEDULING:
   ```SCHEDULE_ADD
   {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "desc", "notes": "opt"}}
Tomorrow = {tmrw}

REMINDERS (when user says remind/notify/alert):
REMINDER_ADDCopy{{"datetime": "YYYY-MM-DD HH:MM", "message": "what"}}
In 30min = {in30} | In 2h = {in2h}
ALWAYS include REMINDER_ADD for reminders.
THOUGHTS: Brain dumps → themes, actions, ideas, emotional check-in.
AUDIO: Short → brain dump. Long → summary, actions, key quotes.
SEARCH: Live results when provided. NEVER say you can't search.
PRICE WATCH:
PRICE_WATCHCopy{{"description": "what", "search_query": "query", "frequency": "daily|weekly"}}

EDIT:
SCHEDULE_REMOVECopy{{"date": "YYYY-MM-DD", "task_keyword": "kw"}}
SCHEDULE_EDITCopy{{"date": "YYYY-MM-DD", "task_keyword": "kw", "new_time": "HH:MM", "new_task": "desc"}}
REMINDER_REMOVECopy{{"keyword": "kw"}}


── TODAY ──
{get_today_schedule_text()}
── REMINDERS ──
{get_reminders_text()}
── WATCHES ──
{get_price_watches_text()}
{ss}
── RULES ──
Never say "As an AI..." | Never say can't search | No generic responses
Schedule → SCHEDULE_ADD | Reminder → REMINDER_ADD | Valid JSON only
Warm, witty, concise. Emoji sparingly."""
def ask_claude(user_id: int, msg: str):
sd = needs_search(msg)
sc = ""
if sd.get("needs_search"):
parts = []
for q in sd.get("search_queries", [])[:3]:
parts.append(f"Search: {q}\n{web_search(q)}")
sc = "\n---\n".join(parts)
Copyappend_message(user_id, "user", msg)
try:
    r = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=4096,
        system=build_system_prompt(user_id, sc),
        messages=get_history(user_id))
    t = r.content[0].text
    append_message(user_id, "assistant", t)
    process_schedule(t)
    process_price_watch(t)
    rd = process_reminder(t, user_id)
    return clean(t), rd
except Exception as e:
    logger.error(f"Claude error: {e}")
    return f"⚠️ Error: {str(e)[:200]}", None
def process_schedule(text: str):
sched = load_schedule()
changed = False
if "SCHEDULE_ADD" in text:         try:             d = json.loads(text.split("SCHEDULE_ADD")[1].split("")[0].strip())             sched.setdefault(d["date"], []).append(                 {"time": d.get("time", ""), "task": d.get("task", ""), "notes": d.get("notes", "")})             sched[d["date"]].sort(key=lambda x: x.get("time", "99"))             changed = True         except Exception as e:             logger.error(f"SCHEDULE_ADD: {e}")     if "SCHEDULE_REMOVE" in text:
try:
d = json.loads(text.split("SCHEDULE_REMOVE")[1].split("")[0].strip())
dk, kw = d["date"], d.get("task_keyword", "").lower()
if dk in sched:
b = len(sched[dk])
sched[dk] = [i for i in sched[dk] if kw not in i.get("task", "").lower()]
changed = len(sched[dk]) < b
except Exception as e:
logger.error(f"SCHEDULE_REMOVE: {e}")
if "SCHEDULE_EDIT" in text:         try:             d = json.loads(text.split("SCHEDULE_EDIT")[1].split("```")[0].strip())
dk, kw = d["date"], d.get("task_keyword", "").lower()
if dk in sched:
for item in sched[dk]:
if kw in item.get("task", "").lower():
if "new_time" in d: item["time"] = d["new_time"]
if "new_task" in d: item["task"] = d["new_task"]
changed = True
break
except Exception as e:
logger.error(f"SCHEDULE_EDIT: {e}")
if changed:
save_schedule(sched)
def process_reminder(text: str, uid: int) -> dict | None:
rd = None
if "REMINDER_ADD" in text:         try:             d = json.loads(text.split("REMINDER_ADD")[1].split("")[0].strip())             dt_s, msg = d.get("datetime", ""), d.get("message", "Reminder!")             rdt = datetime.strptime(dt_s, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)             rems = load_reminders()             rems.append({"datetime": dt_s, "message": msg, "user_id": uid, "sent": False,                          "created": get_now().isoformat()})             save_reminders(rems)             rd = {"datetime": rdt, "message": msg, "user_id": uid}         except Exception as e:             logger.error(f"REMINDER_ADD: {e}")     if "REMINDER_REMOVE" in text:
try:
kw = json.loads(text.split("REMINDER_REMOVE")[1].split("")[0].strip()).get("keyword", "").lower()
rems = load_reminders()
b = len(rems)
rems = [r for r in rems if kw not in r.get("message", "").lower() or r.get("sent")]
if len(rems) < b:
save_reminders(rems)
except Exception as e:
logger.error(f"REMINDER_REMOVE: {e}")
return rd
def process_price_watch(text: str):
if "PRICE_WATCH" not in text:         return     try:         d = json.loads(text.split("PRICE_WATCH")[1].split("```")[0].strip())
w = load_price_watches()
w.append({"description": d.get("description", ""), "search_query": d.get("search_query", ""),
"frequency": d.get("frequency", "daily"), "created": get_now().isoformat(),
"last_checked": "never", "last_result": ""})
save_price_watches(w)
except Exception as e:
logger.error(f"PRICE_WATCH: {e}")
def clean(text: str) -> str:
c = text
for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT",
"REMINDER_ADD", "REMINDER_REMOVE", "PRICE_WATCH"]:
while f"{tag}" in c:             try:                 before = c.split(f"{tag}")[0]
after = c.split(f"{tag}")[1].split("", 1)
c = before + (after[1] if len(after) > 1 else "")
except IndexError:
break
while "\n\n\n" in c:
c = c.replace("\n\n\n", "\n\n")
return c.strip()
── Audio ────────────────────────────────────────────────────────────────────
AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".webm", ".aac", ".wma", ".mpga"}
def is_audio_doc(doc) -> bool:
if not doc:
return False
mime = (doc.mime_type or "").lower()
if mime.startswith("audio/") or mime == "video/mp4":
return True
return any((doc.file_name or "").lower().endswith(e) for e in AUDIO_EXTS)
def get_ext(mime: str = "", name: str = "") -> str:
if name:
for e in AUDIO_EXTS:
if name.lower().endswith(e):
return e
m = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
"audio/x-m4a": ".m4a", "audio/m4a": ".m4a", "audio/aac": ".aac",
"audio/ogg": ".ogg", "audio/wav": ".wav", "audio/flac": ".flac",
"audio/webm": ".webm", "video/mp4": ".mp4"}
return m.get((mime or "").lower(), ".ogg")
def fmt_size(b: int) -> str:
if b > 1024 * 1024 * 1024:
return f"{b / (10243):.1f}GB"
if b > 1024 * 1024:
return f"{b / (10242):.1f}MB"
return f"{b / 1024:.0f}KB"
def duration_text(ms: int) -> str:
s = ms // 1000
h, s = divmod(s, 3600)
m, s = divmod(s, 60)
if h: return f"{h}h {m}m"
if m: return f"{m}m {s}s"
return f"{s}s"
── Pyrogram download (NO SIZE LIMIT) ────────────────────────────────────────
async def download_with_pyrogram(message_id: int, chat_id: int, dest_path: str,
progress_callback=None) -> bool:
"""Download any file from Telegram using MTProto — no size limit."""
try:
async with pyro_client:
# Get the message via pyrogram
msg = await pyro_client.get_messages(chat_id, message_id)
Copy        if not msg:
            logger.error("Pyrogram: message not found")
            return False

        # Download with progress
        path = await msg.download(
            file_name=dest_path,
            progress=progress_callback,
        )

        if path and os.path.exists(path):
            size = os.path.getsize(path)
            logger.info(f"Pyrogram downloaded: {fmt_size(size)}")
            # If pyrogram saved to different path, move it
            if path != dest_path:
                os.rename(path, dest_path)
            return True

        logger.error("Pyrogram download returned no path")
        return False

except Exception as e:
    logger.error(f"Pyrogram download error: {e}")
    return False
async def download_file(bot, file_id: str, dest_path: str,
message_id: int = None, chat_id: int = None,
file_size: int = 0) -> bool:
"""
Download file with automatic fallback:
1. Standard Bot API (fast, up to 20MB)
2. Pyrogram MTProto (any size, no limit)
"""
# Method 1: Standard Bot API (for files under 20MB)
if file_size <= 20 * 1024 * 1024:
try:
tg_file = await bot.get_file(file_id)
await tg_file.download_to_drive(dest_path)
logger.info(f"Bot API download OK: {fmt_size(os.path.getsize(dest_path))}")
return True
except Exception as e:
logger.warning(f"Bot API download failed: {e}")
Copy# Method 2: Pyrogram MTProto (no size limit)
if message_id and chat_id:
    logger.info(f"Using Pyrogram for large file ({fmt_size(file_size)})...")
    success = await download_with_pyrogram(message_id, chat_id, dest_path)
    if success:
        return True

# Method 3: Try Bot API anyway as last resort
try:
    tg_file = await bot.get_file(file_id)
    await tg_file.download_to_drive(dest_path)
    return True
except Exception as e:
    logger.error(f"All download methods failed: {e}")
    return False
── Chunking & transcription ────────────────────────────────────────────────
def split_audio(file_path: str) -> list[str]:
ext = os.path.splitext(file_path)[1].lower().lstrip(".")
fmt = {"mp3": "mp3", "m4a": "mp4", "mp4": "mp4", "wav": "wav", "ogg": "ogg",
"flac": "flac", "webm": "webm", "aac": "aac"}.get(ext, "mp3")
try:
audio = AudioSegment.from_file(file_path, format=fmt)
except Exception:
audio = AudioSegment.from_file(file_path)
Copydur = len(audio)
fsize = os.path.getsize(file_path)
logger.info(f"Audio: {duration_text(dur)}, {fmt_size(fsize)}")

if fsize <= WHISPER_MAX_BYTES and dur <= CHUNK_DURATION_MS:
    return [file_path]

n = math.ceil(dur / CHUNK_DURATION_MS)
logger.info(f"Splitting into {n} chunks")
paths = []
for i in range(n):
    s, e = i * CHUNK_DURATION_MS, min((i + 1) * CHUNK_DURATION_MS, dur)
    p = os.path.join(str(TEMP_DIR), f"c_{i:03d}_{os.getpid()}.mp3")
    audio[s:e].export(p, format="mp3", bitrate="128k")
    cs = os.path.getsize(p)
    if cs > WHISPER_MAX_BYTES:
        os.unlink(p)
        sd = CHUNK_DURATION_MS // 3
        for j in range(3):
            ss, se = s + j * sd, min(s + (j + 1) * sd, e)
            if ss >= e: break
            sp = os.path.join(str(TEMP_DIR), f"c_{i:03d}_{j}_{os.getpid()}.mp3")
            audio[ss:se].export(sp, format="mp3", bitrate="96k")
            paths.append(sp)
    else:
        paths.append(p)
return paths
async def transcribe_file(path: str) -> str:
try:
with open(path, "rb") as f:
return openai_client.audio.transcriptions.create(
model="whisper-1", file=f, response_format="text")
except Exception as e:
logger.error(f"Whisper: {e}")
return None
async def transcribe_chunked(file_path: str, status_cb=None) -> str:
chunks = []
created = False
try:
if os.path.getsize(file_path) <= WHISPER_MAX_BYTES:
r = await transcribe_file(file_path)
if r: return r
Copy    if status_cb:
        await status_cb("✂️ Splitting audio into chunks...")
    chunks = split_audio(file_path)
    created = len(chunks) > 1 or chunks[0] != file_path
    if created and status_cb:
        await status_cb(f"📝 {len(chunks)} chunks. Transcribing...")

    parts = []
    for i, cp in enumerate(chunks):
        if status_cb and len(chunks) > 1:
            await status_cb(f"🔄 Transcribing {i + 1}/{len(chunks)}...")
        t = await transcribe_file(cp)
        parts.append(t.strip() if t else f"[Chunk {i + 1} failed]")
    return "\n\n".join(parts)
finally:
    if created:
        for cp in chunks:
            if cp != file_path:
                try: os.unlink(cp)
                except OSError: pass
── Access control ───────────────────────────────────────────────────────────
def is_allowed(uid: int) -> bool:
return ALLOWED_USER_ID is None or str(uid) == str(ALLOWED_USER_ID)
── Reminder jobs ────────────────────────────────────────────────────────────
async def fire_reminder(context: CallbackContext):
d = context.job.data
try:
await context.bot.send_message(chat_id=d["chat_id"], text=f"⏰ REMINDER\n\n{d['message']}")
except Exception as e:
logger.error(f"Reminder send: {e}")
rems = load_reminders()
for r in rems:
if r.get("message") == d["message"] and r.get("user_id") == d["user_id"] and not r.get("sent"):
r["sent"] = True
break
save_reminders(rems)
def sched_reminder(app, data: dict, chat_id: int):
delay = max((data["datetime"] - get_now()).total_seconds(), 5)
app.job_queue.run_once(fire_reminder, when=delay,
data={"chat_id": chat_id, "message": data["message"], "user_id": data["user_id"]},
name=f"rem_{data['user_id']}_{data['message'][:20]}")
async def check_reminders_bg(context: CallbackContext):
rems = load_reminders()
now = get_now()
changed = False
for r in rems:
if r.get("sent"): continue
try:
rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
except (ValueError, KeyError): continue
if rdt <= now:
cid = get_chat_id(r.get("user_id"))
if cid:
try:
await context.bot.send_message(chat_id=cid, text=f"⏰ REMINDER\n\n{r.get('message', '!')}")
r["sent"] = True
changed = True
except Exception: pass
if changed:
save_reminders(rems)
async def send_long(update, text):
for i in range(0, max(len(text), 1), 4000):
await update.message.reply_text(text[i:i + 4000])
── Handlers ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id):
return await update.message.reply_text("⛔ Private.")
save_chat_id(update.effective_user.id, update.effective_chat.id)
user_conversations[str(update.effective_user.id)] = []
await update.message.reply_text(
"Hey! 👋 I'm Shanti.\n\n"
"• 📅 Schedule naturally\n• ⏰ Reminders\n• 🧠 Structure thoughts\n"
"• 🎤 Audio ANY size (mp3, m4a, wav — even 200MB+!)\n"
"• 📻 Long recordings (1h, 2h, 3h+)\n"
"• ✈️ Flights, events, prices\n• 👀 Price tracking\n\n"
"/today /week /reminders /search /watches\n"
"/checkprices /clear /clearschedule /clearreminders\n\nJust talk! 💬")
async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_chat_id(update.effective_user.id, update.effective_chat.id)
await update.message.reply_text(get_today_schedule_text())
async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_chat_id(update.effective_user.id, update.effective_chat.id)
await update.message.reply_text(get_week_schedule_text())
async def cmd_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
await update.message.reply_text(get_reminders_text())
async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
user_conversations[str(update.effective_user.id)] = []
await update.message.reply_text("🧹 Cleared!")
async def cmd_clear_sched(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_schedule({})
await update.message.reply_text("🗑️ Schedule cleared.")
async def cmd_clear_rem(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_reminders([])
for j in context.application.job_queue.jobs():
if hasattr(j, "name") and j.name and j.name.startswith("rem_"):
j.schedule_removal()
await update.message.reply_text("🗑️ Reminders cleared.")
async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_chat_id(update.effective_user.id, update.effective_chat.id)
q = " ".join(context.args) if context.args else ""
if not q: return await update.message.reply_text("/search <query>")
await update.message.chat.send_action("typing")
r, _ = ask_claude(update.effective_user.id, f"Search "{q}":\n{web_search(q)}")
await send_long(update, r)
async def cmd_watches(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
await update.message.reply_text(get_price_watches_text())
async def cmd_checkprices(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
ws = load_price_watches()
if not ws: return await update.message.reply_text("No watches.")
await update.message.reply_text(f"🔍 Checking {len(ws)}...")
await update.message.chat.send_action("typing")
res = []
for w in ws:
r = web_search(w.get("search_query", w.get("description", "")))
w["last_checked"] = get_now().isoformat()
w["last_result"] = r[:500]
res.append(f"{w['description']}:\n{r}")
save_price_watches(ws)
r, _ = ask_claude(update.effective_user.id, "Price results:\n" + "\n---\n".join(res))
await send_long(update, r)
async def cmd_clearwatches(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
save_price_watches([])
await update.message.reply_text("🗑️ Cleared.")
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
uid = update.effective_user.id
save_chat_id(uid, update.effective_chat.id)
await update.message.chat.send_action("typing")
r, rd = ask_claude(uid, update.message.text)
if rd: sched_reminder(context.application, rd, update.effective_chat.id)
await send_long(update, r)
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
uid = update.effective_user.id
cid = update.effective_chat.id
mid = update.message.message_id
save_chat_id(uid, cid)
Copyvoice, audio, doc = update.message.voice, update.message.audio, update.message.document
file_id = mime = fname = None
fsize = 0

if voice:
    file_id, mime, fname, fsize = voice.file_id, voice.mime_type or "audio/ogg", "voice.ogg", voice.file_size or 0
elif audio:
    file_id, mime, fname, fsize = audio.file_id, audio.mime_type or "", audio.file_name or "audio", audio.file_size or 0
elif doc and is_audio_doc(doc):
    file_id, mime, fname, fsize = doc.file_id, doc.mime_type or "", doc.file_name or "file", doc.file_size or 0
else:
    return

logger.info(f"Audio from {uid}: {fname} ({fmt_size(fsize)})")
await update.message.reply_text(f"🎤 Got audio ({fmt_size(fsize)}), downloading...")
await update.message.chat.send_action("typing")

ext = get_ext(mime, fname)
tmp = os.path.join(str(TEMP_DIR), f"a_{uid}_{mid}{ext}")

try:
    success = await download_file(
        context.bot, file_id, tmp,
        message_id=mid, chat_id=cid, file_size=fsize)

    if not success:
        await update.message.reply_text(
            f"⚠️ Download failed ({fmt_size(fsize)}).\n"
            "Please try sending again or compress the file.")
        return

    actual = os.path.getsize(tmp)
    if actual < 100:
        await update.message.reply_text("⚠️ File is empty. Try again.")
        return

    logger.info(f"Downloaded: {fmt_size(actual)}")
    await update.message.reply_text(f"✅ Downloaded ({fmt_size(actual)}). Transcribing...")

    async def status(msg):
        try:
            await update.message.reply_text(msg)
            await update.message.chat.send_action("typing")
        except Exception: pass

    transcript = await transcribe_chunked(tmp, status_cb=status)

    if not transcript:
        await update.message.reply_text("⚠️ Transcription failed.")
        return

    logger.info(f"Transcribed: {len(transcript)} chars")

    if len(transcript) > 5000:
        vm = (f"[LONG AUDIO — {len(transcript)} chars]\n{transcript}\n[END]\n\n"
              "Provide: 1) Summary 2) Action items 3) Key details 4) Scheduling items")
    else:
        vm = f"[VOICE NOTE]\n{transcript}\n[END]\n\nStructure and respond."

    await update.message.chat.send_action("typing")
    response, rd = ask_claude(uid, vm)
    if rd: sched_reminder(context.application, rd, cid)

    if len(transcript) > 3000:
        await update.message.reply_text("📝 TRANSCRIPTION:")
        for i in range(0, len(transcript), 4000):
            await update.message.reply_text(transcript[i:i + 4000])
        await update.message.reply_text("---\n📋 SUMMARY:")
        await send_long(update, response)
    else:
        await send_long(update, f"📝 Transcription:\n{transcript}\n\n---\n\n{response}")
finally:
    try: os.unlink(tmp)
    except OSError: pass
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not is_allowed(update.effective_user.id): return
if update.message.document and is_audio_doc(update.message.document):
await handle_voice(update, context)
async def post_init(application: Application):
for f in TEMP_DIR.glob("*"):
try: os.unlink(f)
except OSError: pass
Copyrems = load_reminders()
now = get_now()
n = 0
for r in rems:
    if r.get("sent"): continue
    try:
        rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    except (ValueError, KeyError): continue
    cid = get_chat_id(r.get("user_id"))
    if not cid: continue
    application.job_queue.run_once(fire_reminder,
        when=max((rdt - now).total_seconds(), 5),
        data={"chat_id": cid, "message": r.get("message", "!"), "user_id": r.get("user_id")},
        name=f"rem_{r.get('user_id')}_{r.get('message', '')[:20]}")
    n += 1
if n: logger.info(f"Rescheduled {n} reminders")
application.job_queue.run_repeating(check_reminders_bg, interval=60, first=10)
logger.info("Bot ready.")
def main():
app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
Copyapp.add_handler(CommandHandler("start", cmd_start))
app.add_handler(CommandHandler("today", cmd_today))
app.add_handler(CommandHandler("week", cmd_week))
app.add_handler(CommandHandler("reminders", cmd_reminders))
app.add_handler(CommandHandler("clear", cmd_clear))
app.add_handler(CommandHandler("clearschedule", cmd_clear_sched))
app.add_handler(CommandHandler("clearreminders", cmd_clear_rem))
app.add_handler(CommandHandler("search", cmd_search))
app.add_handler(CommandHandler("watches", cmd_watches))
app.add_handler(CommandHandler("checkprices", cmd_checkprices))
app.add_handler(CommandHandler("clearwatches", cmd_clearwatches))
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

logger.info("Starting...")
app.run_polling(allowed_updates=Update.ALL_TYPES)
if name == "main":
main()
