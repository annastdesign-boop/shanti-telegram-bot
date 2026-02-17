import os
import json
import logging
import tempfile
import math
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import openai
from anthropic import Anthropic
from tavily import TavilyClient
from pydub import AudioSegment
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


# ── Per-user conversation history ────────────────────────────────────────────
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


# ── Schedule helpers ─────────────────────────────────────────────────────────
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
        return f"No plans scheduled for today ({today})."
    lines = [f"📅 Schedule for {today}:"]
    for i, item in enumerate(items, 1):
        time_str = item.get("time", "no specific time")
        task = item.get("task", "")
        notes = item.get("notes", "")
        line = f"{i}. [{time_str}] {task}"
        if notes:
            line += f" — {notes}"
        lines.append(line)
    return "\n".join(lines)


def get_week_schedule_text() -> str:
    schedule = load_schedule()
    today = get_now()
    lines = ["📅 This week's schedule:"]
    found_anything = False
    for delta in range(7):
        day = today + timedelta(days=delta)
        key = day.strftime("%Y-%m-%d")
        day_name = day.strftime("%A %b %d")
        items = schedule.get(key, [])
        if items:
            found_anything = True
            lines.append(f"\n{day_name}:")
            for i, item in enumerate(items, 1):
                time_str = item.get("time", "")
                task = item.get("task", "")
                line = f"  {i}. [{time_str}] {task}" if time_str else f"  {i}. {task}"
                lines.append(line)
    if not found_anything:
        lines.append("Nothing scheduled for the next 7 days.")
    return "\n".join(lines)


def load_reminders() -> list:
    return load_json(REMINDERS_FILE, [])


def save_reminders(reminders: list):
    save_json(REMINDERS_FILE, reminders)


def get_reminders_text() -> str:
    reminders = load_reminders()
    active = [r for r in reminders if not r.get("sent", False)]
    if not active:
        return "No active reminders."
    lines = ["⏰ Active Reminders:"]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. [{r.get('datetime', '?')}] {r.get('message', '')}")
    return "\n".join(lines)


def load_price_watches() -> list:
    return load_json(PRICE_WATCHES_FILE, [])


def save_price_watches(watches: list):
    save_json(PRICE_WATCHES_FILE, watches)


def get_price_watches_text() -> str:
    watches = load_price_watches()
    if not watches:
        return "No active price watches."
    lines = ["👀 Active Price Watches:"]
    for i, w in enumerate(watches, 1):
        lines.append(f"{i}. {w.get('description', '?')} — last: {w.get('last_checked', 'never')}")
    return "\n".join(lines)


# ── Tavily ───────────────────────────────────────────────────────────────────
def web_search(query: str, search_depth: str = "advanced", max_results: int = 5) -> str:
    try:
        logger.info(f"Tavily search: {query}")
        response = tavily_client.search(query=query, search_depth=search_depth,
                                         max_results=max_results, include_answer=True)
        text = ""
        if response.get("answer"):
            text += f"AI Summary: {response['answer']}\n\n"
        text += "Sources:\n"
        for i, r in enumerate(response.get("results", []), 1):
            content = r.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            text += f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {content}\n\n"
        return text
    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return f"Search failed: {str(e)[:200]}"


SEARCH_DECISION_PROMPT = """You are a routing assistant. Decide if the user's message needs a live web search.
Reply with EXACTLY one JSON object.
If search needed: {"needs_search": true, "search_queries": ["query"], "search_type": "general"}
If not: {"needs_search": false}
Search needed for: flights, events, hotels, news, weather, prices, restaurants, real-world data.
Not needed for: scheduling, reminders, thoughts, conversation, personal data."""


def decide_if_search_needed(user_message: str) -> dict:
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=300,
            system=SEARCH_DECISION_PROMPT,
            messages=[{"role": "user", "content": user_message}])
        text = response.content[0].text.strip()
        if "{" in text:
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
        return {"needs_search": False}
    except Exception as e:
        logger.error(f"Search decision error: {e}")
        return {"needs_search": False}


def build_system_prompt(user_id: int, search_context: str = "") -> str:
    today_schedule = get_today_schedule_text()
    today = get_now().strftime("%A, %B %d, %Y")
    current_time = get_now().strftime("%H:%M")
    reminders_text = get_reminders_text()
    price_watches = get_price_watches_text()
    search_section = ""
    if search_context:
        search_section = f"\n── LIVE SEARCH RESULTS ──\n{search_context}\nUse these results. Cite prices/links/dates.\n"

    return f"""You are Shanti — a warm, intelligent personal assistant. Deeply personal, context-aware, proactive.

Current date: {today}
Current time: {current_time}
Timezone: {USER_TIMEZONE}

── CAPABILITIES ──

1. SCHEDULING — Extract structured data from plans/appointments:
     ```SCHEDULE_ADD
     {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "description", "notes": "optional"}}
     ```
   Tomorrow = {(get_now() + timedelta(days=1)).strftime("%Y-%m-%d")}

2. REMINDERS — When user says "remind me"/"notify me"/"alert me":
     ```REMINDER_ADD
     {{"datetime": "YYYY-MM-DD HH:MM", "message": "what to remind"}}
     ```
   In 30min = {(get_now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")}
   In 2hrs = {(get_now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")}
   ALWAYS include REMINDER_ADD when user wants reminder.

3. THOUGHT STRUCTURING — Brain dumps → key themes, action items, ideas, emotional check-in.

4. VOICE/AUDIO — Short → brain dump. Long → summary, key points, action items, quotes.

5. WEB SEARCH — Results provided when relevant. NEVER say you can't search.

6. PRICE WATCHING:
     ```PRICE_WATCH
     {{"description": "what", "search_query": "query", "frequency": "daily|weekly"}}
     ```

7. EDITING:
     ```SCHEDULE_REMOVE
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword"}}
     ```
     ```SCHEDULE_EDIT
     {{"date": "YYYY-MM-DD", "task_keyword": "kw", "new_time": "HH:MM", "new_task": "desc"}}
     ```
     ```REMINDER_REMOVE
     {{"keyword": "keyword"}}
     ```

── TODAY ──
{today_schedule}

── REMINDERS ──
{reminders_text}

── PRICE WATCHES ──
{price_watches}
{search_section}
── RULES ──
- NEVER say "As an AI language model..."
- NEVER say you can't search the web
- NEVER give generic responses
- Scheduling → SCHEDULE_ADD block
- Reminder → REMINDER_ADD block
- All blocks = valid JSON
- Warm, witty, concise. Emoji sparingly."""


def ask_claude(user_id: int, user_message: str):
    search_decision = decide_if_search_needed(user_message)
    search_context = ""
    if search_decision.get("needs_search"):
        queries = search_decision.get("search_queries", [])
        logger.info(f"Searching: {queries}")
        results = []
        for q in queries[:3]:
            results.append(f"Search: {q}\n{web_search(q)}")
        search_context = "\n---\n".join(results)

    system_prompt = build_system_prompt(user_id, search_context)
    append_message(user_id, "user", user_message)
    messages = get_history(user_id)

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=4096,
            system=system_prompt, messages=messages)
        text = response.content[0].text
        append_message(user_id, "assistant", text)
        process_schedule_commands(text)
        process_price_watch_commands(text)
        reminder_data = process_reminder_commands(text, user_id)
        return clean_response(text), reminder_data
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return f"⚠️ Error: {str(e)[:200]}", None


def process_schedule_commands(text: str):
    schedule = load_schedule()
    changed = False
    if "```SCHEDULE_ADD" in text:
        try:
            block = text.split("```SCHEDULE_ADD")[1].split("```")[0].strip()
            data = json.loads(block)
            dk = data["date"]
            entry = {"time": data.get("time", ""), "task": data.get("task", ""), "notes": data.get("notes", "")}
            schedule.setdefault(dk, []).append(entry)
            schedule[dk].sort(key=lambda x: x.get("time", "99:99"))
            changed = True
        except Exception as e:
            logger.error(f"SCHEDULE_ADD error: {e}")
    if "```SCHEDULE_REMOVE" in text:
        try:
            block = text.split("```SCHEDULE_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            dk, kw = data["date"], data.get("task_keyword", "").lower()
            if dk in schedule:
                b = len(schedule[dk])
                schedule[dk] = [i for i in schedule[dk] if kw not in i.get("task", "").lower()]
                changed = len(schedule[dk]) < b
        except Exception as e:
            logger.error(f"SCHEDULE_REMOVE error: {e}")
    if "```SCHEDULE_EDIT" in text:
        try:
            block = text.split("```SCHEDULE_EDIT")[1].split("```")[0].strip()
            data = json.loads(block)
            dk, kw = data["date"], data.get("task_keyword", "").lower()
            if dk in schedule:
                for item in schedule[dk]:
                    if kw in item.get("task", "").lower():
                        if "new_time" in data: item["time"] = data["new_time"]
                        if "new_task" in data: item["task"] = data["new_task"]
                        changed = True
                        break
                schedule[dk].sort(key=lambda x: x.get("time", "99:99"))
        except Exception as e:
            logger.error(f"SCHEDULE_EDIT error: {e}")
    if changed:
        save_schedule(schedule)


def process_reminder_commands(text: str, user_id: int) -> dict | None:
    reminder_data = None
    if "```REMINDER_ADD" in text:
        try:
            block = text.split("```REMINDER_ADD")[1].split("```")[0].strip()
            data = json.loads(block)
            dt_str = data.get("datetime", "")
            msg = data.get("message", "Reminder!")
            rdt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            reminders = load_reminders()
            reminders.append({"datetime": dt_str, "message": msg, "user_id": user_id,
                             "sent": False, "created": get_now().isoformat()})
            save_reminders(reminders)
            reminder_data = {"datetime": rdt, "message": msg, "user_id": user_id}
        except Exception as e:
            logger.error(f"REMINDER_ADD error: {e}")
    if "```REMINDER_REMOVE" in text:
        try:
            block = text.split("```REMINDER_REMOVE")[1].split("```")[0].strip()
            kw = json.loads(block).get("keyword", "").lower()
            reminders = load_reminders()
            b = len(reminders)
            reminders = [r for r in reminders if kw not in r.get("message", "").lower() or r.get("sent")]
            if len(reminders) < b:
                save_reminders(reminders)
        except Exception as e:
            logger.error(f"REMINDER_REMOVE error: {e}")
    return reminder_data


def process_price_watch_commands(text: str):
    if "```PRICE_WATCH" not in text:
        return
    try:
        block = text.split("```PRICE_WATCH")[1].split("```")[0].strip()
        data = json.loads(block)
        watches = load_price_watches()
        watches.append({"description": data.get("description", ""), "search_query": data.get("search_query", ""),
                        "frequency": data.get("frequency", "daily"), "created": get_now().isoformat(),
                        "last_checked": "never", "last_result": ""})
        save_price_watches(watches)
    except Exception as e:
        logger.error(f"PRICE_WATCH error: {e}")


def clean_response(text: str) -> str:
    clean = text
    for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT",
                "REMINDER_ADD", "REMINDER_REMOVE", "PRICE_WATCH"]:
        while f"```{tag}" in clean:
            try:
                before = clean.split(f"```{tag}")[0]
                after = clean.split(f"```{tag}")[1].split("```", 1)
                clean = before + (after[1] if len(after) > 1 else "")
            except IndexError:
                break
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


# ── Audio detection ──────────────────────────────────────────────────────────
AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".webm", ".mpga", ".mpeg", ".aac", ".wma"}


def is_audio_document(doc) -> bool:
    if not doc:
        return False
    mime = (doc.mime_type or "").lower()
    if mime.startswith("audio/") or mime == "video/mp4":
        return True
    name = (doc.file_name or "").lower()
    return any(name.endswith(e) for e in AUDIO_EXTS)


def get_ext(mime_type: str = "", file_name: str = "") -> str:
    if file_name:
        for e in AUDIO_EXTS:
            if file_name.lower().endswith(e):
                return e
    m = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
         "audio/m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg", "audio/wav": ".wav",
         "audio/flac": ".flac", "audio/webm": ".webm", "video/mp4": ".mp4"}
    return m.get((mime_type or "").lower(), ".ogg")


# ── Download with retry and fallback ─────────────────────────────────────────
async def download_file(bot, file_id: str, dest_path: str) -> bool:
    """
    Try multiple methods to download a file from Telegram.
    Method 1: Standard bot.get_file + download (works up to 20MB)
    Method 2: Get file_path from get_file, then download via httpx (can work for larger files)
    """
    # Method 1: Standard download
    try:
        logger.info("Trying standard bot API download...")
        tg_file = await bot.get_file(file_id)
        await tg_file.download_to_drive(dest_path)
        size = os.path.getsize(dest_path)
        logger.info(f"Standard download OK: {size / (1024*1024):.1f}MB")
        return True
    except Exception as e1:
        logger.warning(f"Standard download failed: {e1}")

    # Method 2: Direct HTTP download using file_path
    try:
        logger.info("Trying direct HTTP download...")
        # get_file might still return a file_path even for large files on some bot API versions
        tg_file = await bot.get_file(file_id)
        file_path = tg_file.file_path

        if file_path:
            if not file_path.startswith("http"):
                url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            else:
                url = file_path

            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(8192):
                            f.write(chunk)

            size = os.path.getsize(dest_path)
            logger.info(f"Direct HTTP download OK: {size / (1024*1024):.1f}MB")
            return True
    except Exception as e2:
        logger.warning(f"Direct HTTP download failed: {e2}")

    # Method 3: Use a custom API URL format (some setups)
    try:
        logger.info("Trying alternative download method...")
        url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_id}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code == 200:
                    with open(dest_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(8192):
                            f.write(chunk)
                    size = os.path.getsize(dest_path)
                    if size > 0:
                        logger.info(f"Alternative download OK: {size / (1024*1024):.1f}MB")
                        return True
    except Exception as e3:
        logger.warning(f"Alternative download failed: {e3}")

    return False


# ── Audio chunking & transcription ───────────────────────────────────────────
def duration_text(ms: int) -> str:
    s = ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def split_audio(file_path: str) -> list[str]:
    logger.info(f"Loading audio: {file_path}")
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    fmt_map = {"mp3": "mp3", "m4a": "mp4", "mp4": "mp4", "wav": "wav", "ogg": "ogg",
               "oga": "ogg", "flac": "flac", "webm": "webm", "aac": "aac", "wma": "wma"}
    fmt = fmt_map.get(ext, "mp3")

    try:
        audio = AudioSegment.from_file(file_path, format=fmt)
    except Exception:
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error(f"Cannot load audio: {e}")
            raise

    dur = len(audio)
    fsize = os.path.getsize(file_path)
    logger.info(f"Audio loaded: {duration_text(dur)}, {fsize / (1024*1024):.1f}MB")

    if fsize <= WHISPER_MAX_BYTES and dur <= CHUNK_DURATION_MS:
        return [file_path]

    n = math.ceil(dur / CHUNK_DURATION_MS)
    logger.info(f"Splitting into {n} chunks")
    paths = []

    for i in range(n):
        start = i * CHUNK_DURATION_MS
        end = min((i + 1) * CHUNK_DURATION_MS, dur)
        chunk = audio[start:end]
        p = os.path.join(str(TEMP_DIR), f"c_{datetime.now().strftime('%H%M%S')}_{i:03d}.mp3")
        chunk.export(p, format="mp3", bitrate="128k")
        cs = os.path.getsize(p)
        logger.info(f"Chunk {i+1}/{n}: {duration_text(end-start)}, {cs/(1024*1024):.1f}MB")

        if cs > WHISPER_MAX_BYTES:
            os.unlink(p)
            sub_dur = CHUNK_DURATION_MS // 3
            for j in range(3):
                ss = start + j * sub_dur
                se = min(start + (j+1) * sub_dur, end)
                if ss >= end:
                    break
                sc = audio[ss:se]
                sp = os.path.join(str(TEMP_DIR), f"c_{datetime.now().strftime('%H%M%S')}_{i:03d}_{j}.mp3")
                sc.export(sp, format="mp3", bitrate="96k")
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
        logger.error(f"Whisper error: {e}")
        return None


async def transcribe_chunked(file_path: str, status_cb=None) -> str:
    chunks = []
    created = False
    try:
        fsize = os.path.getsize(file_path)

        if fsize <= WHISPER_MAX_BYTES:
            result = await transcribe_file(file_path)
            if result:
                return result

        if status_cb:
            await status_cb("✂️ Splitting audio into chunks...")

        chunks = split_audio(file_path)
        created = len(chunks) > 1 or chunks[0] != file_path

        if created and status_cb:
            await status_cb(f"📝 {len(chunks)} chunks ready. Transcribing...")

        parts = []
        for i, cp in enumerate(chunks):
            if status_cb and len(chunks) > 1:
                await status_cb(f"🔄 Transcribing {i+1}/{len(chunks)}...")
            t = await transcribe_file(cp)
            parts.append(t.strip() if t else f"[Chunk {i+1} failed]")

        return "\n\n".join(parts)
    finally:
        if created:
            for cp in chunks:
                if cp != file_path:
                    try:
                        os.unlink(cp)
                    except OSError:
                        pass


# ── Access control ───────────────────────────────────────────────────────────
def is_allowed(uid: int) -> bool:
    return ALLOWED_USER_ID is None or str(uid) == str(ALLOWED_USER_ID)


# ── Reminders ────────────────────────────────────────────────────────────────
async def send_reminder(context: CallbackContext):
    d = context.job.data
    try:
        await context.bot.send_message(chat_id=d["chat_id"], text=f"⏰ REMINDER\n\n{d['message']}")
    except Exception as e:
        logger.error(f"Reminder failed: {e}")
    reminders = load_reminders()
    for r in reminders:
        if r.get("message") == d["message"] and r.get("user_id") == d["user_id"] and not r.get("sent"):
            r["sent"] = True
            break
    save_reminders(reminders)


def schedule_reminder(app, data: dict, chat_id: int):
    delay = max((data["datetime"] - get_now()).total_seconds(), 5)
    app.job_queue.run_once(send_reminder, when=delay,
        data={"chat_id": chat_id, "message": data["message"], "user_id": data["user_id"]},
        name=f"rem_{data['user_id']}_{data['message'][:20]}")


async def check_reminders(context: CallbackContext):
    reminders = load_reminders()
    now = get_now()
    changed = False
    for r in reminders:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except (ValueError, KeyError):
            continue
        if rdt <= now:
            cid = get_chat_id(r.get("user_id"))
            if cid:
                try:
                    await context.bot.send_message(chat_id=cid, text=f"⏰ REMINDER\n\n{r.get('message', '!')}")
                    r["sent"] = True
                    changed = True
                except Exception:
                    pass
    if changed:
        save_reminders(reminders)


async def send_long(update: Update, text: str):
    for i in range(0, max(len(text), 1), 4000):
        await update.message.reply_text(text[i:i+4000])


def fmt_size(b: int) -> str:
    return f"{b/(1024*1024):.1f}MB" if b > 1024*1024 else f"{b/1024:.0f}KB"


# ── Handlers ─────────────────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return await update.message.reply_text("⛔ Private bot.")
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    user_conversations[str(update.effective_user.id)] = []
    await update.message.reply_text(
        "Hey! 👋 I'm Shanti.\n\n"
        "• 📅 Schedule — tell me naturally\n"
        "• ⏰ Reminders — \"remind me in 2h to call mom\"\n"
        "• 🧠 Structure thoughts & voice notes\n"
        "• 🎤 Audio files ANY size (mp3, m4a, wav...)\n"
        "• ✈️ Search flights, events, prices\n"
        "• 👀 Price tracking\n\n"
        "/today /week /reminders /search /watches\n"
        "/checkprices /clear /clearschedule /clearreminders\n\n"
        "Just talk to me! 💬")


async def today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    await update.message.reply_text(get_today_schedule_text())


async def week_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    await update.message.reply_text(get_week_schedule_text())


async def reminders_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    await update.message.reply_text(get_reminders_text())


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    user_conversations[str(update.effective_user.id)] = []
    await update.message.reply_text("🧹 Cleared!")


async def clear_schedule_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_schedule({})
    await update.message.reply_text("🗑️ Schedule cleared.")


async def clear_reminders_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_reminders([])
    for j in context.application.job_queue.jobs():
        if hasattr(j, "name") and j.name and j.name.startswith("rem_"):
            j.schedule_removal()
    await update.message.reply_text("🗑️ Reminders cleared.")


async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    q = " ".join(context.args) if context.args else ""
    if not q:
        return await update.message.reply_text("Usage: /search <query>")
    await update.message.chat.send_action("typing")
    r, _ = ask_claude(update.effective_user.id, f"Search for \"{q}\":\n\n{web_search(q)}")
    await send_long(update, r)


async def watches_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    await update.message.reply_text(get_price_watches_text())


async def checkprices_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    watches = load_price_watches()
    if not watches:
        return await update.message.reply_text("No watches.")
    await update.message.reply_text(f"🔍 Checking {len(watches)}...")
    await update.message.chat.send_action("typing")
    results = []
    for w in watches:
        res = web_search(w.get("search_query", w.get("description", "")))
        w["last_checked"] = get_now().isoformat()
        w["last_result"] = res[:500]
        results.append(f"{w['description']}:\n{res}")
    save_price_watches(watches)
    r, _ = ask_claude(update.effective_user.id, f"Price results:\n\n" + "\n---\n".join(results))
    await send_long(update, r)


async def clearwatches_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    save_price_watches([])
    await update.message.reply_text("🗑️ Watches cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id): return
    uid = update.effective_user.id
    save_chat_id(uid, update.effective_chat.id)
    await update.message.chat.send_action("typing")
    response, rd = ask_claude(uid, update.message.text)
    if rd:
        schedule_reminder(context.application, rd, update.effective_chat.id)
    await send_long(update, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    uid = update.effective_user.id
    cid = update.effective_chat.id
    save_chat_id(uid, cid)

    voice = update.message.voice
    audio = update.message.audio
    doc = update.message.document

    file_id = mime = fname = None
    fsize = 0

    if voice:
        file_id, mime, fname, fsize = voice.file_id, voice.mime_type or "audio/ogg", "voice.ogg", voice.file_size or 0
    elif audio:
        file_id, mime, fname, fsize = audio.file_id, audio.mime_type or "", audio.file_name or "audio", audio.file_size or 0
    elif doc and is_audio_document(doc):
        file_id, mime, fname, fsize = doc.file_id, doc.mime_type or "", doc.file_name or "file", doc.file_size or 0
    else:
        return

    logger.info(f"Audio from {uid}: {fname} ({mime}, {fmt_size(fsize)})")

    await update.message.reply_text(f"🎤 Got audio ({fmt_size(fsize)}), downloading...")
    await update.message.chat.send_action("typing")

    ext = get_ext(mime, fname)
    tmp_path = os.path.join(str(TEMP_DIR), f"a_{uid}_{datetime.now().strftime('%H%M%S')}{ext}")

    try:
        # Try to download — our function handles multiple methods
        success = await download_file(context.bot, file_id, tmp_path)

        if not success:
            await update.message.reply_text(
                f"⚠️ Couldn't download ({fmt_size(fsize)}).\n\n"
                f"Telegram has a 20MB bot download limit.\n"
                f"Please compress or split the file:\n\n"
                f"🔧 Quick fix options:\n"
                f"1. Use an app to compress to 64kbps MP3\n"
                f"2. Split into parts under 20MB\n"
                f"3. Record as Telegram voice notes instead\n"
                f"4. Use voice memo app and send in parts"
            )
            return

        actual = os.path.getsize(tmp_path)
        logger.info(f"Downloaded: {actual/(1024*1024):.1f}MB")

        if actual < 100:
            await update.message.reply_text("⚠️ Downloaded file is empty. Try sending again.")
            return

        async def status(msg):
            try:
                await update.message.reply_text(msg)
                await update.message.chat.send_action("typing")
            except Exception:
                pass

        transcript = await transcribe_chunked(tmp_path, status_cb=status)

        if not transcript:
            await update.message.reply_text("⚠️ Transcription failed. Supported: MP3, M4A, WAV, OGG, FLAC, WEBM")
            return

        logger.info(f"Transcribed: {len(transcript)} chars")

        if len(transcript) > 5000:
            vm = (f"[LONG AUDIO — {len(transcript)} chars]\n{transcript}\n[END]\n\n"
                  f"Provide: 1) Summary 2) Action items 3) Key details 4) Scheduling items")
        else:
            vm = f"[VOICE NOTE]\n{transcript}\n[END]\n\nStructure and respond."

        await update.message.chat.send_action("typing")
        response, rd = ask_claude(uid, vm)

        if rd:
            schedule_reminder(context.application, rd, cid)

        if len(transcript) > 3000:
            await update.message.reply_text("📝 TRANSCRIPTION:")
            for i in range(0, len(transcript), 4000):
                await update.message.reply_text(transcript[i:i+4000])
            await update.message.reply_text("---\n📋 SUMMARY:")
            await send_long(update, response)
        else:
            await send_long(update, f"📝 Transcription:\n{transcript}\n\n---\n\n{response}")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    if update.message.document and is_audio_document(update.message.document):
        await handle_voice(update, context)


async def post_init(application: Application):
    for f in TEMP_DIR.glob("*"):
        try:
            os.unlink(f)
        except OSError:
            pass

    reminders = load_reminders()
    now = get_now()
    count = 0
    for r in reminders:
        if r.get("sent"):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except (ValueError, KeyError):
            continue
        cid = get_chat_id(r.get("user_id"))
        if not cid:
            continue
        delay = max((rdt - now).total_seconds(), 5)
        application.job_queue.run_once(send_reminder, when=delay,
            data={"chat_id": cid, "message": r.get("message", "!"), "user_id": r.get("user_id")},
            name=f"rem_{r.get('user_id')}_{r.get('message', '')[:20]}")
        count += 1
    if count:
        logger.info(f"Rescheduled {count} reminders")
    application.job_queue.run_repeating(check_reminders, interval=60, first=10, name="rem_check")
    logger.info("Bot ready.")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("today", today_cmd))
    app.add_handler(CommandHandler("week", week_cmd))
    app.add_handler(CommandHandler("reminders", reminders_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("clearschedule", clear_schedule_cmd))
    app.add_handler(CommandHandler("clearreminders", clear_reminders_cmd))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("watches", watches_cmd))
    app.add_handler(CommandHandler("checkprices", checkprices_cmd))
    app.add_handler(CommandHandler("clearwatches", clearwatches_cmd))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
