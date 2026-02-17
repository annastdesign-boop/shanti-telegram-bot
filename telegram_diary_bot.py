import os
import json
import logging
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

# ── Environment ──────────────────────────────────────────────────────────────
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

# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Pyrogram — started once at boot, stays connected
pyro_app = PyroClient(
    name="shanti_downloader",
    api_id=TELEGRAM_API_ID,
    api_hash=TELEGRAM_API_HASH,
    bot_token=TELEGRAM_TOKEN,
    workdir="data",
    no_updates=True,
)
pyro_started = False

# ── Storage ──────────────────────────────────────────────────────────────────
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


def save_chat_id(uid: int, cid: int):
    d = load_json(CHAT_IDS_FILE, {})
    d[str(uid)] = cid
    save_json(CHAT_IDS_FILE, d)


def get_chat_id(uid: int):
    return load_json(CHAT_IDS_FILE, {}).get(str(uid))


# ── Conversation ─────────────────────────────────────────────────────────────
convos: dict[str, list[dict]] = {}
MAX_HIST = 40


def get_hist(uid: int) -> list[dict]:
    k = str(uid)
    if k not in convos:
        convos[k] = []
    return convos[k]


def add_msg(uid: int, role: str, content: str):
    h = get_hist(uid)
    h.append({"role": role, "content": content})
    if len(h) > MAX_HIST:
        convos[str(uid)] = h[-MAX_HIST:]


# ── Schedule ─────────────────────────────────────────────────────────────────
def load_schedule(): return load_json(SCHEDULE_FILE, {})
def save_schedule(s): save_json(SCHEDULE_FILE, s)
def now(): return datetime.now(TZ)
def today_key(): return now().strftime("%Y-%m-%d")


def today_sched_text():
    s = load_schedule()
    items = s.get(today_key(), [])
    if not items:
        return f"No plans for today ({today_key()})."
    lines = [f"📅 Schedule for {today_key()}:"]
    for i, it in enumerate(items, 1):
        l = f"{i}. [{it.get('time', '?')}] {it.get('task', '')}"
        if it.get("notes"): l += f" — {it['notes']}"
        lines.append(l)
    return "\n".join(lines)


def week_sched_text():
    s = load_schedule()
    lines = ["📅 This week:"]
    found = False
    for d in range(7):
        day = now() + timedelta(days=d)
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


# ── Reminders ────────────────────────────────────────────────────────────────
def load_rem(): return load_json(REMINDERS_FILE, [])
def save_rem(r): save_json(REMINDERS_FILE, r)


def rem_text():
    active = [r for r in load_rem() if not r.get("sent")]
    if not active:
        return "No active reminders."
    lines = ["⏰ Reminders:"]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. [{r.get('datetime', '?')}] {r.get('message', '')}")
    return "\n".join(lines)


# ── Price watches ────────────────────────────────────────────────────────────
def load_pw(): return load_json(PRICE_WATCHES_FILE, [])
def save_pw(w): save_json(PRICE_WATCHES_FILE, w)


def pw_text():
    w = load_pw()
    if not w:
        return "No price watches."
    lines = ["👀 Watches:"]
    for i, x in enumerate(w, 1):
        lines.append(f"{i}. {x.get('description', '?')} — last: {x.get('last_checked', 'never')}")
    return "\n".join(lines)


# ── Tavily ───────────────────────────────────────────────────────────────────
def web_search(query):
    try:
        r = tavily_client.search(query=query, search_depth="advanced", max_results=5, include_answer=True)
        t = ""
        if r.get("answer"): t += f"Summary: {r['answer']}\n\n"
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
        r = anthropic_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
            system=SEARCH_P, messages=[{"role": "user", "content": msg}])
        t = r.content[0].text.strip()
        if "{" in t:
            return json.loads(t[t.index("{"):t.rindex("}") + 1])
    except Exception:
        pass
    return {"needs_search": False}


def build_sys(uid, search_ctx=""):
    n = now()
    t = n.strftime("%A, %B %d, %Y")
    tm = n.strftime("%H:%M")
    tmrw = (n + timedelta(days=1)).strftime("%Y-%m-%d")
    in30 = (n + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    in2h = (n + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    ss = f"\n── SEARCH RESULTS ──\n{search_ctx}\nUse these. Cite prices/links.\n" if search_ctx else ""

    return f"""You are Shanti — warm, intelligent personal assistant. Personal, context-aware, proactive.
Date: {t} | Time: {tm} | TZ: {USER_TIMEZONE}
── CAPABILITIES ──
1. SCHEDULING:
   ```SCHEDULE_ADD
   {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "desc", "notes": "opt"}}
Tomorrow = {tmrw}
2. REMINDERS (remind/notify/alert):
REMINDER_ADDCopy{{"datetime": "YYYY-MM-DD HH:MM", "message": "what"}}
In 30min = {in30} | In 2h = {in2h}
ALWAYS include REMINDER_ADD for reminders.
3. THOUGHTS: dumps → themes, actions, ideas, emotional check-in.
4. AUDIO: Short → brain dump. Long → summary, actions, quotes.
5. SEARCH: Live results when provided. NEVER say can't search.
6. PRICE WATCH:
PRICE_WATCHCopy{{"description": "what", "search_query": "query", "frequency": "daily|weekly"}}

EDIT:
SCHEDULE_REMOVECopy{{"date": "YYYY-MM-DD", "task_keyword": "kw"}}
SCHEDULE_EDITCopy{{"date": "YYYY-MM-DD", "task_keyword": "kw", "new_time": "HH:MM", "new_task": "desc"}}
REMINDER_REMOVECopy{{"keyword": "kw"}}


── TODAY ──
{today_sched_text()}
── REMINDERS ──
{rem_text()}
── WATCHES ──
{pw_text()}
{ss}
── RULES ──
Never "As an AI..." | Never say can't search | No generic responses
Schedule → SCHEDULE_ADD | Reminder → REMINDER_ADD | Valid JSON
Warm, witty, concise. Emoji sparingly."""
def ask_claude(uid, msg):
sd = needs_search(msg)
sc = ""
if sd.get("needs_search"):
parts = [f"Search: {q}\n{web_search(q)}" for q in sd.get("search_queries", [])[:3]]
sc = "\n---\n".join(parts)
add_msg(uid, "user", msg)
try:
r = anthropic_client.messages.create(model="claude-sonnet-4-20250514", max_tokens=4096,
system=build_sys(uid, sc), messages=get_hist(uid))
t = r.content[0].text
add_msg(uid, "assistant", t)
proc_sched(t); proc_pw(t)
rd = proc_rem(t, uid)
return cln(t), rd
except Exception as e:
logger.error(f"Claude: {e}")
return f"⚠️ Error: {str(e)[:200]}", None
def proc_sched(text):
s = load_schedule(); ch = False
if "SCHEDULE_ADD" in text:         try:             d = json.loads(text.split("SCHEDULE_ADD")[1].split("")[0].strip())             s.setdefault(d["date"], []).append({"time": d.get("time", ""), "task": d.get("task", ""), "notes": d.get("notes", "")})             s[d["date"]].sort(key=lambda x: x.get("time", "99")); ch = True         except Exception as e: logger.error(f"SA: {e}")     if "SCHEDULE_REMOVE" in text:
try:
d = json.loads(text.split("SCHEDULE_REMOVE")[1].split("")[0].strip())
dk, kw = d["date"], d.get("task_keyword", "").lower()
if dk in s:
b = len(s[dk]); s[dk] = [i for i in s[dk] if kw not in i.get("task", "").lower()]
ch = len(s[dk]) < b
except Exception as e: logger.error(f"SR: {e}")
if "SCHEDULE_EDIT" in text:         try:             d = json.loads(text.split("SCHEDULE_EDIT")[1].split("```")[0].strip())
dk, kw = d["date"], d.get("task_keyword", "").lower()
if dk in s:
for it in s[dk]:
if kw in it.get("task", "").lower():
if "new_time" in d: it["time"] = d["new_time"]
if "new_task" in d: it["task"] = d["new_task"]
ch = True; break
except Exception as e: logger.error(f"SE: {e}")
if ch: save_schedule(s)
def proc_rem(text, uid):
rd = None
if "REMINDER_ADD" in text:         try:             d = json.loads(text.split("REMINDER_ADD")[1].split("")[0].strip())             ds, m = d.get("datetime", ""), d.get("message", "Reminder!")             rdt = datetime.strptime(ds, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)             rems = load_rem()             rems.append({"datetime": ds, "message": m, "user_id": uid, "sent": False, "created": now().isoformat()})             save_rem(rems)             rd = {"datetime": rdt, "message": m, "user_id": uid}         except Exception as e: logger.error(f"RA: {e}")     if "REMINDER_REMOVE" in text:
try:
kw = json.loads(text.split("REMINDER_REMOVE")[1].split("")[0].strip()).get("keyword", "").lower()
rems = load_rem(); b = len(rems)
rems = [r for r in rems if kw not in r.get("message", "").lower() or r.get("sent")]
if len(rems) < b: save_rem(rems)
except Exception as e: logger.error(f"RR: {e}")
return rd
def proc_pw(text):
if "PRICE_WATCH" not in text: return     try:         d = json.loads(text.split("PRICE_WATCH")[1].split("```")[0].strip())
w = load_pw()
w.append({"description": d.get("description", ""), "search_query": d.get("search_query", ""),
"frequency": d.get("frequency", "daily"), "created": now().isoformat(),
"last_checked": "never", "last_result": ""})
save_pw(w)
except Exception as e: logger.error(f"PW: {e}")
def cln(text):
c = text
for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT", "REMINDER_ADD", "REMINDER_REMOVE", "PRICE_WATCH"]:
while f"{tag}" in c:             try:                 bf = c.split(f"{tag}")[0]; af = c.split(f"{tag}")[1].split("", 1)
c = bf + (af[1] if len(af) > 1 else "")
except: break
while "\n\n\n" in c: c = c.replace("\n\n\n", "\n\n")
return c.strip()
── Audio ────────────────────────────────────────────────────────────────────
AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".webm", ".aac", ".wma"}
def is_audio_doc(doc):
if not doc: return False
m = (doc.mime_type or "").lower()
if m.startswith("audio/") or m == "video/mp4": return True
return any((doc.file_name or "").lower().endswith(e) for e in AUDIO_EXTS)
def get_ext(mime="", name=""):
if name:
for e in AUDIO_EXTS:
if name.lower().endswith(e): return e
m = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
"audio/m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg", "audio/wav": ".wav",
"audio/flac": ".flac", "audio/webm": ".webm", "video/mp4": ".mp4"}
return m.get((mime or "").lower(), ".ogg")
def fmt(b):
if b > 10243: return f"{b / 10243:.1f}GB"
if b > 10242: return f"{b / 10242:.1f}MB"
return f"{b / 1024:.0f}KB"
def dur_text(ms):
s = ms // 1000; h, s = divmod(s, 3600); m, s = divmod(s, 60)
if h: return f"{h}h {m}m"
if m: return f"{m}m {s}s"
return f"{s}s"
── PYROGRAM DOWNLOAD — NO SIZE LIMIT ────────────────────────────────────────
async def ensure_pyro():
"""Start pyrogram if not already started."""
global pyro_started
if not pyro_started:
logger.info("Starting Pyrogram client...")
await pyro_app.start()
pyro_started = True
logger.info("Pyrogram client started OK")
async def download_via_pyrogram(chat_id: int, message_id: int, dest: str, progress_cb=None) -> bool:
"""Download file using Pyrogram MTProto — NO size limit."""
try:
await ensure_pyro()
logger.info(f"Pyrogram: getting message {message_id} from chat {chat_id}")
msg = await pyro_app.get_messages(chat_id, message_id)
if not msg:
logger.error("Pyrogram: message not found")
return False
Copy    logger.info("Pyrogram: starting download...")
    result = await msg.download(file_name=dest, progress=progress_cb)
    
    if result:
        # pyrogram might return a different path
        if isinstance(result, str) and result != dest:
            if os.path.exists(result):
                os.rename(result, dest)
        
        if os.path.exists(dest):
            sz = os.path.getsize(dest)
            logger.info(f"Pyrogram download complete: {fmt(sz)}")
            return sz > 0

    logger.error("Pyrogram: download returned nothing")
    return False
except Exception as e:
    logger.error(f"Pyrogram download error: {type(e).__name__}: {e}")
    return False
async def download_via_bot_api(bot, file_id: str, dest: str) -> bool:
"""Standard Bot API download — up to 20MB."""
try:
f = await bot.get_file(file_id)
await f.download_to_drive(dest)
sz = os.path.getsize(dest)
logger.info(f"Bot API download: {fmt(sz)}")
return sz > 0
except Exception as e:
logger.warning(f"Bot API download failed: {e}")
return False
async def download_audio(bot, file_id: str, dest: str, chat_id: int, message_id: int,
file_size: int, progress_cb=None) -> bool:
"""
Download audio file. Tries Bot API first (fast), falls back to Pyrogram (no limit).
"""
# Always try pyrogram for files over 15MB (don't even waste time with bot API)
if file_size > 15 * 1024 * 1024:
logger.info(f"Large file ({fmt(file_size)}), going straight to Pyrogram")
ok = await download_via_pyrogram(chat_id, message_id, dest, progress_cb)
if ok:
return True
# If pyrogram fails, try bot API as hail mary
logger.info("Pyrogram failed, trying Bot API as fallback...")
return await download_via_bot_api(bot, file_id, dest)
Copy# Small files: try Bot API first (faster)
ok = await download_via_bot_api(bot, file_id, dest)
if ok:
    return True

# Fallback to pyrogram
logger.info("Bot API failed, trying Pyrogram...")
return await download_via_pyrogram(chat_id, message_id, dest, progress_cb)
── Chunking & transcription ────────────────────────────────────────────────
def split_audio(path):
ext = os.path.splitext(path)[1].lower().lstrip(".")
fm = {"mp3": "mp3", "m4a": "mp4", "mp4": "mp4", "wav": "wav", "ogg": "ogg",
"flac": "flac", "webm": "webm", "aac": "aac"}.get(ext, "mp3")
try:
audio = AudioSegment.from_file(path, format=fm)
except Exception:
audio = AudioSegment.from_file(path)
Copyd = len(audio)
fs = os.path.getsize(path)
logger.info(f"Audio: {dur_text(d)}, {fmt(fs)}")

if fs <= WHISPER_MAX_BYTES and d <= CHUNK_DURATION_MS:
    return [path]

n = math.ceil(d / CHUNK_DURATION_MS)
logger.info(f"Splitting into {n} chunks")
paths = []
for i in range(n):
    s, e = i * CHUNK_DURATION_MS, min((i + 1) * CHUNK_DURATION_MS, d)
    p = os.path.join(str(TEMP_DIR), f"c_{os.getpid()}_{i:03d}.mp3")
    audio[s:e].export(p, format="mp3", bitrate="128k")
    cs = os.path.getsize(p)
    if cs > WHISPER_MAX_BYTES:
        os.unlink(p)
        sd = CHUNK_DURATION_MS // 3
        for j in range(3):
            ss, se = s + j * sd, min(s + (j + 1) * sd, e)
            if ss >= e: break
            sp = os.path.join(str(TEMP_DIR), f"c_{os.getpid()}_{i:03d}_{j}.mp3")
            audio[ss:se].export(sp, format="mp3", bitrate="96k")
            paths.append(sp)
    else:
        paths.append(p)
return paths
async def whisper(path):
try:
with open(path, "rb") as f:
return openai_client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
except Exception as e:
logger.error(f"Whisper: {e}")
return None
async def transcribe(path, status_cb=None):
chunks = []; created = False
try:
if os.path.getsize(path) <= WHISPER_MAX_BYTES:
r = await whisper(path)
if r: return r
Copy    if status_cb: await status_cb("✂️ Splitting audio...")
    chunks = split_audio(path)
    created = len(chunks) > 1 or chunks[0] != path
    if created and status_cb:
        await status_cb(f"📝 {len(chunks)} chunks. Transcribing...")

    parts = []
    for i, cp in enumerate(chunks):
        if status_cb and len(chunks) > 1:
            await status_cb(f"🔄 Chunk {i + 1}/{len(chunks)}...")
        t = await whisper(cp)
        parts.append(t.strip() if t else f"[Chunk {i + 1} failed]")
    return "\n\n".join(parts)
finally:
    if created:
        for cp in chunks:
            if cp != path:
                try: os.unlink(cp)
                except: pass
def is_allowed(uid): return ALLOWED_USER_ID is None or str(uid) == str(ALLOWED_USER_ID)
── Reminder jobs ────────────────────────────────────────────────────────────
async def fire_rem(ctx: CallbackContext):
d = ctx.job.data
try:
await ctx.bot.send_message(chat_id=d["chat_id"], text=f"⏰ REMINDER\n\n{d['message']}")
except Exception as e:
logger.error(f"Rem send: {e}")
rems = load_rem()
for r in rems:
if r.get("message") == d["message"] and r.get("user_id") == d["user_id"] and not r.get("sent"):
r["sent"] = True; break
save_rem(rems)
def sched_rem(app, data, cid):
delay = max((data["datetime"] - now()).total_seconds(), 5)
app.job_queue.run_once(fire_rem, when=delay,
data={"chat_id": cid, "message": data["message"], "user_id": data["user_id"]},
name=f"r_{data['user_id']}_{data['message'][:20]}")
async def check_rem_bg(ctx: CallbackContext):
rems = load_rem(); n = now(); ch = False
for r in rems:
if r.get("sent"): continue
try:
rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
except: continue
if rdt <= n:
cid = get_chat_id(r.get("user_id"))
if cid:
try:
await ctx.bot.send_message(chat_id=cid, text=f"⏰ REMINDER\n\n{r.get('message', '!')}")
r["sent"] = True; ch = True
except: pass
if ch: save_rem(rems)
async def send_long(upd, text):
for i in range(0, max(len(text), 1), 4000):
await upd.message.reply_text(text[i:i + 4000])
── Handlers ─────────────────────────────────────────────────────────────────
async def cmd_start(upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
if not is_allowed(upd.effective_user.id): return await upd.message.reply_text("⛔")
save_chat_id(upd.effective_user.id, upd.effective_chat.id)
convos[str(upd.effective_user.id)] = []
await upd.message.reply_text(
"Hey! 👋 I'm Shanti.\n\n"
"• 📅 Schedule naturally\n• ⏰ Reminders\n• 🧠 Structure thoughts\n"
"• 🎤 Audio ANY size — even 200MB+!\n• 📻 Long recordings (hours)\n"
"• ✈️ Flights, events, prices\n• 👀 Price tracking\n\n"
"/today /week /reminders /search /watches\n"
"/checkprices /clear /clearschedule /clearreminders\n\nJust talk! 💬")
async def cmd_today(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_chat_id(u.effective_user.id, u.effective_chat.id)
await u.message.reply_text(today_sched_text())
async def cmd_week(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_chat_id(u.effective_user.id, u.effective_chat.id)
await u.message.reply_text(week_sched_text())
async def cmd_rem(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
await u.message.reply_text(rem_text())
async def cmd_clear(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
convos[str(u.effective_user.id)] = []
await u.message.reply_text("🧹 Cleared!")
async def cmd_clr_sched(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_schedule({}); await u.message.reply_text("🗑️ Schedule cleared.")
async def cmd_clr_rem(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_rem([])
for j in c.application.job_queue.jobs():
if hasattr(j, "name") and j.name and j.name.startswith("r_"): j.schedule_removal()
await u.message.reply_text("🗑️ Reminders cleared.")
async def cmd_search(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_chat_id(u.effective_user.id, u.effective_chat.id)
q = " ".join(c.args) if c.args else ""
if not q: return await u.message.reply_text("/search <query>")
await u.message.chat.send_action("typing")
r, _ = ask_claude(u.effective_user.id, f"Search "{q}":\n{web_search(q)}")
await send_long(u, r)
async def cmd_watches(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
await u.message.reply_text(pw_text())
async def cmd_checkprices(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
ws = load_pw()
if not ws: return await u.message.reply_text("No watches.")
await u.message.reply_text(f"🔍 Checking {len(ws)}...")
await u.message.chat.send_action("typing")
res = []
for w in ws:
r = web_search(w.get("search_query", w.get("description", "")))
w["last_checked"] = now().isoformat(); w["last_result"] = r[:500]
res.append(f"{w['description']}:\n{r}")
save_pw(ws)
r, _ = ask_claude(u.effective_user.id, "Price results:\n" + "\n---\n".join(res))
await send_long(u, r)
async def cmd_clr_watches(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
save_pw([]); await u.message.reply_text("🗑️ Cleared.")
async def handle_text(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
uid = u.effective_user.id
save_chat_id(uid, u.effective_chat.id)
await u.message.chat.send_action("typing")
r, rd = ask_claude(uid, u.message.text)
if rd: sched_rem(c.application, rd, u.effective_chat.id)
await send_long(u, r)
async def handle_voice(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
uid = u.effective_user.id
cid = u.effective_chat.id
mid = u.message.message_id
save_chat_id(uid, cid)
Copyvoice, audio, doc = u.message.voice, u.message.audio, u.message.document
file_id = mime = fname = None; fsize = 0

if voice:
    file_id, mime, fname, fsize = voice.file_id, voice.mime_type or "audio/ogg", "voice.ogg", voice.file_size or 0
elif audio:
    file_id, mime, fname, fsize = audio.file_id, audio.mime_type or "", audio.file_name or "audio", audio.file_size or 0
elif doc and is_audio_doc(doc):
    file_id, mime, fname, fsize = doc.file_id, doc.mime_type or "", doc.file_name or "file", doc.file_size or 0
else:
    return

logger.info(f"Audio from {uid}: {fname} ({fmt(fsize)})")
await u.message.reply_text(f"🎤 Got audio ({fmt(fsize)}). Downloading...")
await u.message.chat.send_action("typing")

ext = get_ext(mime, fname)
tmp = os.path.join(str(TEMP_DIR), f"a_{uid}_{mid}{ext}")

try:
    # Download — uses Bot API for small, Pyrogram for large
    ok = await download_audio(
        bot=c.bot,
        file_id=file_id,
        dest=tmp,
        chat_id=cid,
        message_id=mid,
        file_size=fsize,
    )

    if not ok:
        await u.message.reply_text("⚠️ Download failed. Please try sending again.")
        return

    actual = os.path.getsize(tmp)
    if actual < 100:
        await u.message.reply_text("⚠️ File empty. Try again.")
        return

    logger.info(f"Downloaded OK: {fmt(actual)}")
    await u.message.reply_text(f"✅ Downloaded ({fmt(actual)}). Transcribing...")

    async def status(msg):
        try:
            await u.message.reply_text(msg)
            await u.message.chat.send_action("typing")
        except: pass

    transcript = await transcribe(tmp, status_cb=status)
    if not transcript:
        await u.message.reply_text("⚠️ Transcription failed.")
        return

    logger.info(f"Transcribed: {len(transcript)} chars")

    if len(transcript) > 5000:
        vm = f"[LONG AUDIO — {len(transcript)} chars]\n{transcript}\n[END]\n\nProvide: 1) Summary 2) Action items 3) Key details 4) Scheduling items"
    else:
        vm = f"[VOICE NOTE]\n{transcript}\n[END]\n\nStructure and respond."

    await u.message.chat.send_action("typing")
    response, rd = ask_claude(uid, vm)
    if rd: sched_rem(c.application, rd, cid)

    if len(transcript) > 3000:
        await u.message.reply_text("📝 TRANSCRIPTION:")
        for i in range(0, len(transcript), 4000):
            await u.message.reply_text(transcript[i:i + 4000])
        await u.message.reply_text("---\n📋 SUMMARY:")
        await send_long(u, response)
    else:
        await send_long(u, f"📝 Transcription:\n{transcript}\n\n---\n\n{response}")
finally:
    try: os.unlink(tmp)
    except: pass
async def handle_doc(u: Update, c: ContextTypes.DEFAULT_TYPE):
if not is_allowed(u.effective_user.id): return
if u.message.document and is_audio_doc(u.message.document):
await handle_voice(u, c)
async def post_init(app: Application):
for f in TEMP_DIR.glob("*"):
try: os.unlink(f)
except: pass
Copy# Start pyrogram at boot
try:
    await ensure_pyro()
    logger.info("Pyrogram ready for large file downloads")
except Exception as e:
    logger.error(f"Pyrogram init failed: {e} — large files won't work")

rems = load_rem(); n = now(); cnt = 0
for r in rems:
    if r.get("sent"): continue
    try:
        rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    except: continue
    cid = get_chat_id(r.get("user_id"))
    if not cid: continue
    app.job_queue.run_once(fire_rem, when=max((rdt - n).total_seconds(), 5),
        data={"chat_id": cid, "message": r.get("message", "!"), "user_id": r.get("user_id")},
        name=f"r_{r.get('user_id')}_{r.get('message', '')[:20]}")
    cnt += 1
if cnt: logger.info(f"Rescheduled {cnt} reminders")
app.job_queue.run_repeating(check_rem_bg, interval=60, first=10)
logger.info("Bot ready.")
async def post_shutdown(app: Application):
"""Stop pyrogram on shutdown."""
global pyro_started
if pyro_started:
try:
await pyro_app.stop()
pyro_started = False
logger.info("Pyrogram stopped.")
except Exception:
pass
def main():
app = (
Application.builder()
.token(TELEGRAM_TOKEN)
.post_init(post_init)
.post_shutdown(post_shutdown)
.build()
)
Copyapp.add_handler(CommandHandler("start", cmd_start))
app.add_handler(CommandHandler("today", cmd_today))
app.add_handler(CommandHandler("week", cmd_week))
app.add_handler(CommandHandler("reminders", cmd_rem))
app.add_handler(CommandHandler("clear", cmd_clear))
app.add_handler(CommandHandler("clearschedule", cmd_clr_sched))
app.add_handler(CommandHandler("clearreminders", cmd_clr_rem))
app.add_handler(CommandHandler("search", cmd_search))
app.add_handler(CommandHandler("watches", cmd_watches))
app.add_handler(CommandHandler("checkprices", cmd_checkprices))
app.add_handler(CommandHandler("clearwatches", cmd_clr_watches))
app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
app.add_handler(MessageHandler(filters.Document.ALL, handle_doc))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

logger.info("Starting...")
app.run_polling(allowed_updates=Update.ALL_TYPES)
if name == "main":
main()
