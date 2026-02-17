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
# Telegram Bot API limit for getFile is 20MB
TELEGRAM_GETFILE_LIMIT = 20 * 1024 * 1024
# We'll handle files up to 2GB by downloading via direct URL
MAX_FILE_SIZE = 2000 * 1024 * 1024  # 2GB absolute max

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


# ── Chat ID tracking ────────────────────────────────────────────────────────
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


# ── Reminder helpers ─────────────────────────────────────────────────────────
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
        dt = r.get("datetime", "unknown time")
        msg = r.get("message", "")
        lines.append(f"{i}. [{dt}] {msg}")
    return "\n".join(lines)


# ── Price watch helpers ──────────────────────────────────────────────────────
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
        lines.append(f"{i}. {w.get('description', 'unknown')} — last checked: {w.get('last_checked', 'never')}")
    return "\n".join(lines)


# ── Tavily web search ────────────────────────────────────────────────────────
def web_search(query: str, search_depth: str = "advanced", max_results: int = 5) -> str:
    try:
        logger.info(f"Tavily search: {query}")
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
        )
        results_text = ""
        if response.get("answer"):
            results_text += f"AI Summary: {response['answer']}\n\n"
        results_text += "Sources:\n"
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            results_text += f"{i}. {title}\n   {url}\n   {content}\n\n"
        return results_text
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Search failed: {str(e)[:200]}"


# ── Search decision ──────────────────────────────────────────────────────────
SEARCH_DECISION_PROMPT = """You are a routing assistant. Decide if the user's message needs a live web search.

Reply with EXACTLY one JSON object, nothing else.

If search IS needed:
{"needs_search": true, "search_queries": ["query 1", "query 2"], "search_type": "general|flights|events|prices"}

If search is NOT needed:
{"needs_search": false}

Messages that NEED search: flight prices, event listings, hotel prices, news, weather, restaurant recommendations, anything requiring current real-world data.

Messages that do NOT need search: scheduling, reminders, thought structuring, personal conversation, summarizing plans, anything about the user's own data."""


def decide_if_search_needed(user_message: str) -> dict:
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SEARCH_DECISION_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text.strip()
        if "{" in text:
            json_str = text[text.index("{"):text.rindex("}") + 1]
            return json.loads(json_str)
        return {"needs_search": False}
    except Exception as e:
        logger.error(f"Search decision error: {e}")
        return {"needs_search": False}


# ── System prompt ────────────────────────────────────────────────────────────
def build_system_prompt(user_id: int, search_context: str = "") -> str:
    today_schedule = get_today_schedule_text()
    today = get_now().strftime("%A, %B %d, %Y")
    current_time = get_now().strftime("%H:%M")
    reminders_text = get_reminders_text()
    price_watches = get_price_watches_text()

    search_section = ""
    if search_context:
        search_section = f"""
── LIVE SEARCH RESULTS (just fetched from the web) ──
{search_context}

IMPORTANT: Use these search results to answer the user's question. Cite specific prices, links, and dates.
"""

    return f"""You are Shanti — a warm, intelligent personal assistant. You are deeply personal, context-aware, and proactive.

Current date: {today}
Current time: {current_time}
Timezone: {USER_TIMEZONE}

── YOUR CORE CAPABILITIES ──

1. **SCHEDULING & PLANNING**
   - When the user mentions plans/appointments/tasks, EXTRACT structured data.
   - Put at END of your message:
     ```SCHEDULE_ADD
     {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "description", "notes": "optional"}}
     ```
   - Calculate dates: "tomorrow" = {(get_now() + timedelta(days=1)).strftime("%Y-%m-%d")}

2. **REMINDERS (PUSH NOTIFICATIONS)**
   - When user says "remind me" / "notify me" / "alert me":
     ```REMINDER_ADD
     {{"datetime": "YYYY-MM-DD HH:MM", "message": "what to remind about"}}
     ```
   - "in 30 min" → {(get_now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")}
   - "tomorrow 9am" → {(get_now() + timedelta(days=1)).strftime("%Y-%m-%d")} 09:00
   - "in 2 hours" → {(get_now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")}
   - ALWAYS include REMINDER_ADD block when user wants a reminder.

3. **DAILY SUMMARY** — schedule + reminders cleanly.

4. **THOUGHT STRUCTURING** — brain dumps → key themes, action items, ideas, emotional check-in.

5. **VOICE NOTES & LONG AUDIO**
   - Short voice → structure as brain dump.
   - Long recordings → structured summary, key points, action items, notable quotes.

6. **WEB SEARCH** — live results provided when relevant. NEVER say you can't search.

7. **PRICE WATCHING**
     ```PRICE_WATCH
     {{"description": "what", "search_query": "query", "frequency": "daily|weekly"}}
     ```

8. **REMOVING/EDITING**
     ```SCHEDULE_REMOVE
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword"}}
     ```
     ```SCHEDULE_EDIT
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword", "new_time": "HH:MM", "new_task": "description"}}
     ```
     ```REMINDER_REMOVE
     {{"keyword": "keyword"}}
     ```

── TODAY'S SCHEDULE ──
{today_schedule}

── ACTIVE REMINDERS ──
{reminders_text}

── PRICE WATCHES ──
{price_watches}
{search_section}
── PERSONALITY ──
Warm, supportive, slightly witty. Remember context. Proactive. Concise. Emoji sparingly.

── RULES ──
- NEVER say "As an AI language model..."
- NEVER say you can't search the web
- NEVER give generic responses
- Scheduling intent → SCHEDULE_ADD block
- Reminder intent → REMINDER_ADD block
- All blocks must be valid JSON
- Today is {today}, time is {current_time}
"""


# ── Claude API call ──────────────────────────────────────────────────────────
def ask_claude(user_id: int, user_message: str):
    search_decision = decide_if_search_needed(user_message)
    search_context = ""

    if search_decision.get("needs_search"):
        queries = search_decision.get("search_queries", [])
        logger.info(f"Search needed. Queries: {queries}")
        all_results = []
        for query in queries[:3]:
            result = web_search(query)
            all_results.append(f"Search: {query}\n{result}")
        search_context = "\n---\n".join(all_results)

    system_prompt = build_system_prompt(user_id, search_context)
    append_message(user_id, "user", user_message)
    messages = get_history(user_id)

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
        assistant_text = response.content[0].text
        append_message(user_id, "assistant", assistant_text)

        process_schedule_commands(assistant_text)
        process_price_watch_commands(assistant_text)
        reminder_data = process_reminder_commands(assistant_text, user_id)

        clean_text = clean_response(assistant_text)
        return clean_text, reminder_data

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"⚠️ Something went wrong: {str(e)[:200]}", None


# ── Command processing ───────────────────────────────────────────────────────
def process_schedule_commands(text: str):
    schedule = load_schedule()
    changed = False

    if "```SCHEDULE_ADD" in text:
        try:
            block = text.split("```SCHEDULE_ADD")[1].split("```")[0].strip()
            data = json.loads(block)
            date_key = data["date"]
            entry = {"time": data.get("time", ""), "task": data.get("task", ""), "notes": data.get("notes", "")}
            if date_key not in schedule:
                schedule[date_key] = []
            schedule[date_key].append(entry)
            schedule[date_key].sort(key=lambda x: x.get("time", "99:99"))
            changed = True
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"SCHEDULE_ADD error: {e}")

    if "```SCHEDULE_REMOVE" in text:
        try:
            block = text.split("```SCHEDULE_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            date_key = data["date"]
            keyword = data.get("task_keyword", "").lower()
            if date_key in schedule:
                before = len(schedule[date_key])
                schedule[date_key] = [i for i in schedule[date_key] if keyword not in i.get("task", "").lower()]
                if len(schedule[date_key]) < before:
                    changed = True
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"SCHEDULE_REMOVE error: {e}")

    if "```SCHEDULE_EDIT" in text:
        try:
            block = text.split("```SCHEDULE_EDIT")[1].split("```")[0].strip()
            data = json.loads(block)
            date_key = data["date"]
            keyword = data.get("task_keyword", "").lower()
            if date_key in schedule:
                for item in schedule[date_key]:
                    if keyword in item.get("task", "").lower():
                        if "new_time" in data:
                            item["time"] = data["new_time"]
                        if "new_task" in data:
                            item["task"] = data["new_task"]
                        changed = True
                        break
                schedule[date_key].sort(key=lambda x: x.get("time", "99:99"))
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"SCHEDULE_EDIT error: {e}")

    if changed:
        save_schedule(schedule)


def process_reminder_commands(text: str, user_id: int) -> dict | None:
    reminder_data = None

    if "```REMINDER_ADD" in text:
        try:
            block = text.split("```REMINDER_ADD")[1].split("```")[0].strip()
            data = json.loads(block)
            reminder_dt_str = data.get("datetime", "")
            message = data.get("message", "Reminder!")
            reminder_dt = datetime.strptime(reminder_dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            reminders = load_reminders()
            reminders.append({
                "datetime": reminder_dt_str, "message": message,
                "user_id": user_id, "sent": False, "created": get_now().isoformat(),
            })
            save_reminders(reminders)
            reminder_data = {"datetime": reminder_dt, "message": message, "user_id": user_id}
            logger.info(f"Added reminder: '{message}' at {reminder_dt_str}")
        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            logger.error(f"REMINDER_ADD error: {e}")

    if "```REMINDER_REMOVE" in text:
        try:
            block = text.split("```REMINDER_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            keyword = data.get("keyword", "").lower()
            reminders = load_reminders()
            before = len(reminders)
            reminders = [r for r in reminders if keyword not in r.get("message", "").lower() or r.get("sent", False)]
            if len(reminders) < before:
                save_reminders(reminders)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"REMINDER_REMOVE error: {e}")

    return reminder_data


def process_price_watch_commands(text: str):
    if "```PRICE_WATCH" not in text:
        return
    try:
        block = text.split("```PRICE_WATCH")[1].split("```")[0].strip()
        data = json.loads(block)
        watches = load_price_watches()
        watches.append({
            "description": data.get("description", ""), "search_query": data.get("search_query", ""),
            "frequency": data.get("frequency", "daily"), "created": get_now().isoformat(),
            "last_checked": "never", "last_result": "",
        })
        save_price_watches(watches)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"PRICE_WATCH error: {e}")


def clean_response(text: str) -> str:
    clean = text
    for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT",
                "REMINDER_ADD", "REMINDER_REMOVE", "PRICE_WATCH"]:
        while f"```{tag}" in clean:
            try:
                before = clean.split(f"```{tag}")[0]
                after = clean.split(f"```{tag}")[1].split("```", 1)
                remaining = after[1] if len(after) > 1 else ""
                clean = before + remaining
            except IndexError:
                break
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


# ── Audio file detection ─────────────────────────────────────────────────────
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga",
    ".flac", ".webm", ".mpga", ".mpeg", ".aac", ".wma",
}
SUPPORTED_AUDIO_MIMES = {"audio/", "video/mp4", "application/octet-stream"}


def is_audio_document(document) -> bool:
    if not document:
        return False
    mime = (document.mime_type or "").lower()
    if any(mime.startswith(m) for m in SUPPORTED_AUDIO_MIMES):
        return True
    name = (document.file_name or "").lower()
    if any(name.endswith(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS):
        return True
    return False


def get_file_extension(mime_type: str = "", file_name: str = "") -> str:
    if file_name:
        name_lower = file_name.lower()
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            if name_lower.endswith(ext):
                return ext
    mime_map = {
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
        "audio/mp4a-latm": ".m4a", "audio/x-m4a": ".m4a", "audio/m4a": ".m4a",
        "audio/aac": ".aac", "audio/ogg": ".ogg", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/webm": ".webm",
        "video/mp4": ".mp4", "audio/mpeg3": ".mp3", "audio/x-mpeg-3": ".mp3",
    }
    mime_lower = (mime_type or "").lower()
    return mime_map.get(mime_lower, ".ogg")


# ── Large file download via direct URL ───────────────────────────────────────
async def download_telegram_file(bot, file_id: str, dest_path: str, file_size: int = 0) -> bool:
    """
    Download a file from Telegram. For files under 20MB uses bot API.
    For larger files, uses the direct file URL workaround.
    """
    try:
        if file_size <= TELEGRAM_GETFILE_LIMIT:
            # Standard method for small files
            file = await bot.get_file(file_id)
            await file.download_to_drive(dest_path)
            logger.info(f"Downloaded via bot API: {os.path.getsize(dest_path) / (1024*1024):.1f}MB")
            return True
        else:
            # For large files: get file path then download directly
            # This works with the standard Bot API but requires the file_path
            try:
                file = await bot.get_file(file_id)
                file_url = file.file_path

                # If file_path is a full URL, use it directly
                if not file_url.startswith("http"):
                    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_url}"

                logger.info(f"Downloading large file via direct URL: {file_size / (1024*1024):.1f}MB")

                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    async with client.stream("GET", file_url) as response:
                        response.raise_for_status()
                        with open(dest_path, "wb") as f:
                            downloaded = 0
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                                downloaded += len(chunk)

                actual_size = os.path.getsize(dest_path)
                logger.info(f"Downloaded via direct URL: {actual_size / (1024*1024):.1f}MB")
                return True

            except Exception as e:
                logger.warning(f"Direct URL download failed: {e}, trying standard method...")
                # Fallback to standard method
                try:
                    file = await bot.get_file(file_id)
                    await file.download_to_drive(dest_path)
                    logger.info(f"Fallback download succeeded: {os.path.getsize(dest_path) / (1024*1024):.1f}MB")
                    return True
                except Exception as e2:
                    logger.error(f"All download methods failed: {e2}")
                    return False

    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


# ── Audio chunking and transcription ─────────────────────────────────────────
def get_audio_duration_text(duration_ms: int) -> str:
    total_seconds = duration_ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def split_audio_to_chunks(file_path: str, chunk_duration_ms: int = CHUNK_DURATION_MS) -> list[str]:
    logger.info(f"Loading audio file for chunking: {file_path}")

    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    format_map = {
        "mp3": "mp3", "m4a": "mp4", "mp4": "mp4", "wav": "wav",
        "ogg": "ogg", "oga": "ogg", "flac": "flac", "webm": "webm",
        "aac": "aac", "wma": "wma", "mpga": "mp3", "mpeg": "mp3",
    }
    audio_format = format_map.get(ext, "mp3")

    try:
        audio = AudioSegment.from_file(file_path, format=audio_format)
    except Exception:
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error(f"Could not load audio: {e}")
            raise

    duration_ms = len(audio)
    file_size = os.path.getsize(file_path)
    logger.info(f"Audio: {get_audio_duration_text(duration_ms)}, {file_size / (1024*1024):.1f}MB")

    if file_size <= WHISPER_MAX_BYTES and duration_ms <= chunk_duration_ms:
        return [file_path]

    num_chunks = math.ceil(duration_ms / chunk_duration_ms)
    logger.info(f"Splitting into {num_chunks} chunks")

    chunk_paths = []
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
        chunk = audio[start_ms:end_ms]

        chunk_path = os.path.join(str(TEMP_DIR), f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="128k")

        chunk_size = os.path.getsize(chunk_path)
        logger.info(f"Chunk {i+1}/{num_chunks}: {get_audio_duration_text(end_ms - start_ms)}, {chunk_size / (1024*1024):.1f}MB")

        if chunk_size > WHISPER_MAX_BYTES:
            os.unlink(chunk_path)
            sub_duration = chunk_duration_ms // 3
            for j in range(3):
                sub_start = start_ms + (j * sub_duration)
                sub_end = min(start_ms + ((j + 1) * sub_duration), end_ms)
                if sub_start >= end_ms:
                    break
                sub_chunk = audio[sub_start:sub_end]
                sub_path = os.path.join(str(TEMP_DIR), f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}_{j}.mp3")
                sub_chunk.export(sub_path, format="mp3", bitrate="96k")
                chunk_paths.append(sub_path)
        else:
            chunk_paths.append(chunk_path)

    return chunk_paths


async def transcribe_audio_file(file_path: str) -> str:
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        return transcript
    except Exception as e:
        logger.error(f"Whisper error for {file_path}: {e}")
        return None


async def transcribe_with_chunking(file_path: str, status_callback=None) -> str:
    chunk_paths = []
    created_chunks = False

    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"Transcribing: {file_size / (1024*1024):.1f}MB")

        if file_size <= WHISPER_MAX_BYTES:
            result = await transcribe_audio_file(file_path)
            if result:
                return result

        if status_callback:
            await status_callback("✂️ Splitting audio into chunks for transcription...")

        chunk_paths = split_audio_to_chunks(file_path)
        created_chunks = len(chunk_paths) > 1 or chunk_paths[0] != file_path

        if created_chunks and status_callback:
            await status_callback(f"📝 Split into {len(chunk_paths)} chunks. Starting transcription...")

        all_transcripts = []
        for i, chunk_path in enumerate(chunk_paths):
            if status_callback and len(chunk_paths) > 1:
                await status_callback(f"🔄 Transcribing chunk {i+1}/{len(chunk_paths)}...")

            transcript = await transcribe_audio_file(chunk_path)
            if transcript:
                all_transcripts.append(transcript.strip())
                logger.info(f"Chunk {i+1} done: {len(transcript)} chars")
            else:
                all_transcripts.append(f"[Chunk {i+1} failed to transcribe]")

        full_transcript = "\n\n".join(all_transcripts)
        logger.info(f"Full transcription: {len(full_transcript)} chars")
        return full_transcript

    finally:
        if created_chunks:
            for chunk_path in chunk_paths:
                if chunk_path != file_path:
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        pass


# ── Access control ───────────────────────────────────────────────────────────
def is_allowed(user_id: int) -> bool:
    if ALLOWED_USER_ID is None:
        return True
    return str(user_id) == str(ALLOWED_USER_ID)


# ── Reminder callbacks ──────────────────────────────────────────────────────
async def send_reminder(context: CallbackContext):
    job_data = context.job.data
    try:
        await context.bot.send_message(chat_id=job_data["chat_id"], text=f"⏰ REMINDER\n\n{job_data['message']}")
        logger.info(f"Sent reminder: {job_data['message']}")
    except Exception as e:
        logger.error(f"Reminder send failed: {e}")

    reminders = load_reminders()
    for r in reminders:
        if r.get("message") == job_data["message"] and r.get("user_id") == job_data["user_id"] and not r.get("sent"):
            r["sent"] = True
            break
    save_reminders(reminders)


def schedule_reminder_job(application, reminder_data: dict, chat_id: int):
    delay = (reminder_data["datetime"] - get_now()).total_seconds()
    if delay <= 0:
        delay = 5
    application.job_queue.run_once(
        send_reminder, when=delay,
        data={"chat_id": chat_id, "message": reminder_data["message"], "user_id": reminder_data["user_id"]},
        name=f"reminder_{reminder_data['user_id']}_{reminder_data['message'][:20]}",
    )
    logger.info(f"Scheduled reminder in {delay:.0f}s")


async def check_saved_reminders(context: CallbackContext):
    reminders = load_reminders()
    now = get_now()
    changed = False
    for r in reminders:
        if r.get("sent", False):
            continue
        try:
            reminder_dt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except (ValueError, KeyError):
            continue
        if reminder_dt <= now:
            chat_id = get_chat_id(r.get("user_id"))
            if chat_id:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=f"⏰ REMINDER\n\n{r.get('message', 'Reminder!')}")
                    r["sent"] = True
                    changed = True
                except Exception as e:
                    logger.error(f"Backup reminder failed: {e}")
    if changed:
        save_reminders(reminders)


# ── Helpers ──────────────────────────────────────────────────────────────────
async def send_long_message(update: Update, text: str):
    if len(text) > 4000:
        for i in range(0, len(text), 4000):
            await update.message.reply_text(text[i:i + 4000])
    else:
        await update.message.reply_text(text)


def format_file_size(size_bytes: int) -> str:
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024*1024):.1f}MB"
    return f"{size_bytes / 1024:.0f}KB"


# ── Telegram handlers ───────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Private bot.")
        return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    user_conversations[str(update.effective_user.id)] = []
    await update.message.reply_text(
        "Hey! 👋 I'm Shanti, your personal assistant.\n\n"
        "• 📅 Schedule — just tell me naturally\n"
        "• ⏰ Reminders — \"remind me in 2 hours to call mom\"\n"
        "• 📋 /today or /week for summaries\n"
        "• 🧠 Structure your thoughts & voice notes\n"
        "• 🎤 Audio files of ANY size (mp3, m4a, wav...)\n"
        "• 📻 Long recordings (meetings, lectures, podcasts)\n"
        "• ✈️ Search flights, events, prices\n"
        "• 👀 /watches for price tracking\n\n"
        "Commands: /today /week /reminders /search /watches\n"
        "/checkprices /clear /clearschedule /clearreminders\n\n"
        "Just start talking! 💬"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    await update.message.reply_text(get_today_schedule_text())


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    await update.message.reply_text(get_week_schedule_text())


async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    await update.message.reply_text(get_reminders_text())


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_conversations[str(update.effective_user.id)] = []
    await update.message.reply_text("🧹 Conversation cleared!")


async def clear_schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_schedule({})
    await update.message.reply_text("🗑️ Schedule cleared.")


async def clear_reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_reminders([])
    for job in context.application.job_queue.jobs():
        if hasattr(job, "name") and job.name and job.name.startswith("reminder_"):
            job.schedule_removal()
    await update.message.reply_text("🗑️ Reminders cleared.")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_chat_id(update.effective_user.id, update.effective_chat.id)
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /search <query>")
        return
    await update.message.chat.send_action("typing")
    results = web_search(query)
    response, _ = ask_claude(update.effective_user.id, f"Search results for \"{query}\":\n\n{results}")
    await send_long_message(update, response)


async def watches_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(get_price_watches_text())


async def check_prices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    watches = load_price_watches()
    if not watches:
        await update.message.reply_text("No active price watches.")
        return
    await update.message.reply_text(f"🔍 Checking {len(watches)} watch(es)...")
    await update.message.chat.send_action("typing")
    all_results = []
    for w in watches:
        result = web_search(w.get("search_query", w.get("description", "")))
        w["last_checked"] = get_now().isoformat()
        w["last_result"] = result[:500]
        all_results.append(f"{w['description']}:\n{result}")
    save_price_watches(watches)
    response, _ = ask_claude(update.effective_user.id, f"Price results:\n\n" + "\n---\n".join(all_results))
    await send_long_message(update, response)


async def clear_watches_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_price_watches([])
    await update.message.reply_text("🗑️ Price watches cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    save_chat_id(user_id, update.effective_chat.id)
    logger.info(f"Text from {user_id}: {update.message.text[:100]}...")
    await update.message.chat.send_action("typing")
    response, reminder_data = ask_claude(user_id, update.message.text)
    if reminder_data:
        schedule_reminder_job(context.application, reminder_data, update.effective_chat.id)
    await send_long_message(update, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    save_chat_id(user_id, chat_id)

    voice = update.message.voice
    audio = update.message.audio
    document = update.message.document

    file_id = None
    mime_type = ""
    file_name = ""
    file_size = 0

    if voice:
        file_id = voice.file_id
        mime_type = voice.mime_type or "audio/ogg"
        file_name = "voice.ogg"
        file_size = voice.file_size or 0
    elif audio:
        file_id = audio.file_id
        mime_type = audio.mime_type or ""
        file_name = audio.file_name or "audio"
        file_size = audio.file_size or 0
    elif document and is_audio_document(document):
        file_id = document.file_id
        mime_type = document.mime_type or ""
        file_name = document.file_name or "file"
        file_size = document.file_size or 0
    else:
        return

    logger.info(f"Audio from {user_id}: '{file_name}' ({mime_type}, {format_file_size(file_size)})")

    # Absolute max check
    if file_size > MAX_FILE_SIZE:
        await update.message.reply_text(f"⚠️ File too large ({format_file_size(file_size)}). Maximum is 2GB.")
        return

    size_str = format_file_size(file_size)
    if file_size > TELEGRAM_GETFILE_LIMIT:
        await update.message.reply_text(
            f"🎤 Large file detected ({size_str}). Downloading... this may take a moment."
        )
    else:
        await update.message.reply_text(f"🎤 Got your audio ({size_str}), processing...")

    await update.message.chat.send_action("typing")

    ext = get_file_extension(mime_type, file_name)
    tmp_path = os.path.join(str(TEMP_DIR), f"audio_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")

    try:
        # Download using our enhanced method that handles large files
        success = await download_telegram_file(context.bot, file_id, tmp_path, file_size)

        if not success:
            await update.message.reply_text(
                f"⚠️ Couldn't download this file ({size_str}).\n\n"
                f"Telegram Bot API has a ~20MB limit for file downloads.\n"
                f"For files over 20MB, try:\n"
                f"1. Compress to lower bitrate (64kbps MP3)\n"
                f"2. Split into smaller parts\n"
                f"3. Send as multiple voice notes"
            )
            return

        actual_size = os.path.getsize(tmp_path)
        logger.info(f"File downloaded: {actual_size / (1024*1024):.1f}MB")

        async def status_update(msg: str):
            try:
                await update.message.reply_text(msg)
                await update.message.chat.send_action("typing")
            except Exception:
                pass

        transcript = await transcribe_with_chunking(tmp_path, status_callback=status_update)

        if transcript is None:
            await update.message.reply_text(
                "⚠️ Couldn't transcribe this audio.\n"
                "Supported: MP3, M4A, MP4, WAV, OGG, FLAC, WEBM"
            )
            return

        logger.info(f"Transcription: {len(transcript)} chars from {user_id}")

        if len(transcript) > 5000:
            voice_message = (
                f"[LONG AUDIO TRANSCRIPTION — {len(transcript)} characters]\n"
                f"{transcript}\n[END TRANSCRIPTION]\n\n"
                f"Provide: 1) Concise summary 2) Action items 3) Notable details 4) Any scheduling items"
            )
        else:
            voice_message = f"[VOICE NOTE]\n{transcript}\n[END]\n\nStructure and respond to this."

        await update.message.chat.send_action("typing")
        response, reminder_data = ask_claude(user_id, voice_message)

        if reminder_data:
            schedule_reminder_job(context.application, reminder_data, chat_id)

        if len(transcript) > 3000:
            await update.message.reply_text("📝 TRANSCRIPTION:")
            for i in range(0, len(transcript), 4000):
                await update.message.reply_text(transcript[i:i + 4000])
            await update.message.reply_text("---\n\n📋 SUMMARY & ANALYSIS:")
            await send_long_message(update, response)
        else:
            await send_long_message(update, f"📝 Transcription:\n{transcript}\n\n---\n\n{response}")

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


# ── Startup ──────────────────────────────────────────────────────────────────
async def post_init(application: Application):
    for f in TEMP_DIR.glob("*"):
        try:
            os.unlink(f)
        except OSError:
            pass

    reminders = load_reminders()
    now = get_now()
    rescheduled = 0
    for r in reminders:
        if r.get("sent", False):
            continue
        try:
            rdt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except (ValueError, KeyError):
            continue
        chat_id = get_chat_id(r.get("user_id"))
        if not chat_id:
            continue
        delay = max((rdt - now).total_seconds(), 5)
        application.job_queue.run_once(
            send_reminder, when=delay,
            data={"chat_id": chat_id, "message": r.get("message", "Reminder!"), "user_id": r.get("user_id")},
            name=f"reminder_{r.get('user_id')}_{r.get('message', '')[:20]}",
        )
        rescheduled += 1
    if rescheduled:
        logger.info(f"Rescheduled {rescheduled} reminders")

    application.job_queue.run_repeating(check_saved_reminders, interval=60, first=10, name="reminder_checker")
    logger.info("Bot initialized. Reminder checker started.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("week", week_command))
    app.add_handler(CommandHandler("reminders", reminders_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("clearschedule", clear_schedule_command))
    app.add_handler(CommandHandler("clearreminders", clear_reminders_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("watches", watches_command))
    app.add_handler(CommandHandler("checkprices", check_prices_command))
    app.add_handler(CommandHandler("clearwatches", clear_watches_command))

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Shanti bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
