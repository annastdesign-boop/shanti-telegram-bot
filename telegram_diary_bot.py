import os
import json
import logging
import tempfile
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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

# ── Whisper limits ───────────────────────────────────────────────────────────
# OpenAI Whisper max file size is 25MB. We chunk at 24MB to be safe.
WHISPER_MAX_BYTES = 24 * 1024 * 1024  # 24 MB
# Each chunk duration in milliseconds (10 minutes per chunk)
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes

# Telegram Bot API file download limit is 20MB.
# For larger files we need a local Bot API server OR the file URL trick.
# We'll handle this with a custom download approach.
TELEGRAM_FILE_LIMIT = 20 * 1024 * 1024  # 20 MB

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

IMPORTANT: Use these search results to answer the user's question. Cite specific prices, links, and dates. If results aren't relevant enough, say so.
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
   - Calculate dates from today: "tomorrow" = {(get_now() + timedelta(days=1)).strftime("%Y-%m-%d")}, etc.

2. **REMINDERS (PUSH NOTIFICATIONS)**
   - When the user says "remind me", "send me a reminder", "notify me", "alert me" — create a reminder.
   - Put at END of your message:
     ```REMINDER_ADD
     {{"datetime": "YYYY-MM-DD HH:MM", "message": "what to remind about"}}
     ```
   - Calculate the datetime properly. Examples:
     - "remind me in 30 minutes" → {(get_now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")}
     - "remind me tomorrow at 9am" → {(get_now() + timedelta(days=1)).strftime("%Y-%m-%d")} 09:00
     - "remind me in 2 hours" → {(get_now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")}
   - ALWAYS include the REMINDER_ADD block when the user wants to be reminded.

3. **DAILY SUMMARY**
   - When asked for summary — provide schedule + active reminders cleanly.

4. **THOUGHT STRUCTURING**
   - Brain dumps → organize into: key themes, action items, ideas to revisit, emotional check-in.

5. **VOICE NOTES & LONG AUDIO**
   - Transcribed voice/audio → treat as brain dumps unless specific requests.
   - Long recordings (meetings, lectures, podcasts) will be fully transcribed.
   - For long transcriptions: provide a structured summary with key points, action items, and notable quotes.

6. **WEB SEARCH & LIVE INFO**
   - You have live web search. When results are provided, USE them with specific prices/dates/links.
   - NEVER say you can't search the web.

7. **PRICE WATCHING**
   - Track prices:
     ```PRICE_WATCH
     {{"description": "what to track", "search_query": "search query", "frequency": "daily|weekly"}}
     ```

8. **REMOVING/EDITING**
   - Remove schedule:
     ```SCHEDULE_REMOVE
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword"}}
     ```
   - Edit schedule:
     ```SCHEDULE_EDIT
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword", "new_time": "HH:MM", "new_task": "description"}}
     ```
   - Remove reminder:
     ```REMINDER_REMOVE
     {{"keyword": "keyword to match in reminder message"}}
     ```

── TODAY'S SCHEDULE ──
{today_schedule}

── ACTIVE REMINDERS ──
{reminders_text}

── PRICE WATCHES ──
{price_watches}
{search_section}
── PERSONALITY ──
- Warm, supportive, slightly witty
- Remember conversation context
- Proactive: flag conflicts, remind deadlines
- Concise but thorough
- Ask clarifying questions when needed
- Emoji sparingly but naturally

── RULES ──
- NEVER say "As an AI language model..."
- NEVER say you can't search the web
- NEVER give generic responses
- If scheduling intent → SCHEDULE_ADD block
- If reminder intent → REMINDER_ADD block (ALWAYS)
- All command blocks must be valid JSON
- Today is {today}, current time is {current_time}
"""


# ── Claude API call ──────────────────────────────────────────────────────────
def ask_claude(user_id: int, user_message: str):
    # Step 1: Decide if search needed
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

    # Step 2: Build prompt with search context
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

        # Process all command blocks
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
            entry = {
                "time": data.get("time", ""),
                "task": data.get("task", ""),
                "notes": data.get("notes", ""),
            }
            if date_key not in schedule:
                schedule[date_key] = []
            schedule[date_key].append(entry)
            schedule[date_key].sort(key=lambda x: x.get("time", "99:99"))
            changed = True
            logger.info(f"Added schedule: {entry} on {date_key}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"SCHEDULE_ADD parse error: {e}")

    if "```SCHEDULE_REMOVE" in text:
        try:
            block = text.split("```SCHEDULE_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            date_key = data["date"]
            keyword = data.get("task_keyword", "").lower()
            if date_key in schedule:
                before = len(schedule[date_key])
                schedule[date_key] = [
                    i for i in schedule[date_key]
                    if keyword not in i.get("task", "").lower()
                ]
                if len(schedule[date_key]) < before:
                    changed = True
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"SCHEDULE_REMOVE parse error: {e}")

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
            logger.error(f"SCHEDULE_EDIT parse error: {e}")

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

            reminder_dt = datetime.strptime(reminder_dt_str, "%Y-%m-%d %H:%M")
            reminder_dt = reminder_dt.replace(tzinfo=TZ)

            reminders = load_reminders()
            reminder_entry = {
                "datetime": reminder_dt_str,
                "message": message,
                "user_id": user_id,
                "sent": False,
                "created": get_now().isoformat(),
            }
            reminders.append(reminder_entry)
            save_reminders(reminders)

            reminder_data = {
                "datetime": reminder_dt,
                "message": message,
                "user_id": user_id,
            }

            logger.info(f"Added reminder: '{message}' at {reminder_dt_str}")

        except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
            logger.error(f"REMINDER_ADD parse error: {e}")

    if "```REMINDER_REMOVE" in text:
        try:
            block = text.split("```REMINDER_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            keyword = data.get("keyword", "").lower()

            reminders = load_reminders()
            before = len(reminders)
            reminders = [
                r for r in reminders
                if keyword not in r.get("message", "").lower() or r.get("sent", False)
            ]
            if len(reminders) < before:
                save_reminders(reminders)
                logger.info(f"Removed reminder matching '{keyword}'")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"REMINDER_REMOVE parse error: {e}")

    return reminder_data


def process_price_watch_commands(text: str):
    if "```PRICE_WATCH" not in text:
        return
    try:
        block = text.split("```PRICE_WATCH")[1].split("```")[0].strip()
        data = json.loads(block)
        watches = load_price_watches()
        watches.append({
            "description": data.get("description", ""),
            "search_query": data.get("search_query", ""),
            "frequency": data.get("frequency", "daily"),
            "created": get_now().isoformat(),
            "last_checked": "never",
            "last_result": "",
        })
        save_price_watches(watches)
        logger.info(f"Added price watch: {data.get('description')}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"PRICE_WATCH parse error: {e}")


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
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/mp4a-latm": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/m4a": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/webm": ".webm",
        "video/mp4": ".mp4",
        "audio/mpeg3": ".mp3",
        "audio/x-mpeg-3": ".mp3",
    }
    mime_lower = (mime_type or "").lower()
    if mime_lower in mime_map:
        return mime_map[mime_lower]
    if mime_lower.startswith("audio/"):
        return ".ogg"
    return ".ogg"


# ── Audio chunking and transcription ─────────────────────────────────────────
def get_audio_duration_text(duration_ms: int) -> str:
    """Convert milliseconds to human readable duration."""
    total_seconds = duration_ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def split_audio_to_chunks(file_path: str, chunk_duration_ms: int = CHUNK_DURATION_MS) -> list[str]:
    """
    Split an audio file into chunks that fit within Whisper's 25MB limit.
    Returns list of temporary file paths for each chunk.
    """
    logger.info(f"Loading audio file: {file_path}")

    # Determine format from extension
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    format_map = {
        "mp3": "mp3",
        "m4a": "mp4",
        "mp4": "mp4",
        "wav": "wav",
        "ogg": "ogg",
        "oga": "ogg",
        "flac": "flac",
        "webm": "webm",
        "aac": "aac",
        "wma": "wma",
        "mpga": "mp3",
        "mpeg": "mp3",
    }
    audio_format = format_map.get(ext, "mp3")

    try:
        audio = AudioSegment.from_file(file_path, format=audio_format)
    except Exception:
        # Fallback: let pydub auto-detect
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error(f"Could not load audio file: {e}")
            raise

    duration_ms = len(audio)
    file_size = os.path.getsize(file_path)

    logger.info(f"Audio loaded: {get_audio_duration_text(duration_ms)}, {file_size / (1024*1024):.1f}MB")

    # If file is small enough, no chunking needed
    if file_size <= WHISPER_MAX_BYTES and duration_ms <= chunk_duration_ms:
        logger.info("File is small enough, no chunking needed")
        return [file_path]

    # Calculate number of chunks
    num_chunks = math.ceil(duration_ms / chunk_duration_ms)
    logger.info(f"Splitting into {num_chunks} chunks of {chunk_duration_ms // 60000} minutes each")

    chunk_paths = []
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
        chunk = audio[start_ms:end_ms]

        # Export chunk as mp3 (good compression, Whisper supports it well)
        chunk_path = os.path.join(
            str(TEMP_DIR),
            f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.mp3"
        )
        chunk.export(chunk_path, format="mp3", bitrate="128k")

        chunk_size = os.path.getsize(chunk_path)
        logger.info(
            f"Chunk {i+1}/{num_chunks}: "
            f"{get_audio_duration_text(end_ms - start_ms)}, "
            f"{chunk_size / (1024*1024):.1f}MB"
        )

        # If chunk is still too big, split it further
        if chunk_size > WHISPER_MAX_BYTES:
            logger.warning(f"Chunk {i+1} still too large ({chunk_size / (1024*1024):.1f}MB), splitting further")
            os.unlink(chunk_path)
            sub_duration = chunk_duration_ms // 3
            for j in range(3):
                sub_start = start_ms + (j * sub_duration)
                sub_end = min(start_ms + ((j + 1) * sub_duration), end_ms)
                if sub_start >= end_ms:
                    break
                sub_chunk = audio[sub_start:sub_end]
                sub_path = os.path.join(
                    str(TEMP_DIR),
                    f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}_{j}.mp3"
                )
                sub_chunk.export(sub_path, format="mp3", bitrate="96k")
                chunk_paths.append(sub_path)
                logger.info(f"  Sub-chunk {j+1}: {os.path.getsize(sub_path) / (1024*1024):.1f}MB")
        else:
            chunk_paths.append(chunk_path)

    return chunk_paths


async def transcribe_audio_file(file_path: str) -> str:
    """Transcribe a single audio file using Whisper."""
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
    """
    Transcribe an audio file of any size by chunking if necessary.
    status_callback: async function(message) to send progress updates.
    """
    chunk_paths = []
    created_chunks = False

    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"Transcribing file: {file_size / (1024*1024):.1f}MB")

        # Try direct transcription first for small files
        if file_size <= WHISPER_MAX_BYTES:
            logger.info("File under 24MB, trying direct transcription...")
            result = await transcribe_audio_file(file_path)
            if result:
                return result
            logger.warning("Direct transcription failed, falling back to chunking")

        # Split into chunks
        if status_callback:
            await status_callback("📎 Large file detected. Splitting into chunks for transcription...")

        chunk_paths = split_audio_to_chunks(file_path)
        created_chunks = len(chunk_paths) > 1 or chunk_paths[0] != file_path

        if created_chunks and status_callback:
            await status_callback(f"✂️ Split into {len(chunk_paths)} chunks. Transcribing...")

        # Transcribe each chunk
        all_transcripts = []
        for i, chunk_path in enumerate(chunk_paths):
            if status_callback and len(chunk_paths) > 1:
                await status_callback(f"🔄 Transcribing chunk {i+1}/{len(chunk_paths)}...")

            transcript = await transcribe_audio_file(chunk_path)
            if transcript:
                all_transcripts.append(transcript.strip())
                logger.info(f"Chunk {i+1}/{len(chunk_paths)} transcribed: {len(transcript)} chars")
            else:
                all_transcripts.append(f"[Chunk {i+1} failed to transcribe]")
                logger.error(f"Chunk {i+1} transcription failed")

        # Combine all transcripts
        full_transcript = "\n\n".join(all_transcripts)
        logger.info(f"Full transcription complete: {len(full_transcript)} chars total")
        return full_transcript

    finally:
        # Clean up chunk files (but not the original)
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


# ── Reminder job callbacks ──────────────────────────────────────────────────
async def send_reminder(context: CallbackContext):
    job_data = context.job.data
    chat_id = job_data["chat_id"]
    message = job_data["message"]
    user_id = job_data["user_id"]

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⏰ REMINDER\n\n{message}",
        )
        logger.info(f"Sent reminder to {user_id}: {message}")
    except Exception as e:
        logger.error(f"Failed to send reminder: {e}")

    reminders = load_reminders()
    for r in reminders:
        if r.get("message") == message and r.get("user_id") == user_id and not r.get("sent"):
            r["sent"] = True
            break
    save_reminders(reminders)


def schedule_reminder_job(application, reminder_data: dict, chat_id: int):
    reminder_dt = reminder_data["datetime"]
    now = get_now()

    delay_seconds = (reminder_dt - now).total_seconds()
    if delay_seconds <= 0:
        delay_seconds = 5

    job_data = {
        "chat_id": chat_id,
        "message": reminder_data["message"],
        "user_id": reminder_data["user_id"],
    }

    application.job_queue.run_once(
        send_reminder,
        when=delay_seconds,
        data=job_data,
        name=f"reminder_{reminder_data['user_id']}_{reminder_data['message'][:20]}",
    )

    logger.info(f"Scheduled reminder in {delay_seconds:.0f}s ({reminder_dt})")


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
            user_id = r.get("user_id")
            chat_id = get_chat_id(user_id)
            if chat_id:
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⏰ REMINDER\n\n{r.get('message', 'Reminder!')}",
                    )
                    r["sent"] = True
                    changed = True
                    logger.info(f"Backup sent reminder: {r.get('message')}")
                except Exception as e:
                    logger.error(f"Failed to send reminder: {e}")

    if changed:
        save_reminders(reminders)


# ── Send long messages helper ────────────────────────────────────────────────
async def send_long_message(update: Update, text: str):
    if len(text) > 4000:
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(text)


# ── Telegram handlers ───────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ This bot is private.")
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    save_chat_id(user_id, chat_id)
    user_conversations[str(user_id)] = []

    await update.message.reply_text(
        "Hey! 👋 I'm Shanti, your personal assistant.\n\n"
        "Here's what I can do:\n"
        "• 📅 Schedule things — just tell me naturally\n"
        "• ⏰ Set reminders — \"remind me in 2 hours to call mom\"\n"
        "• 📋 Daily/weekly summaries — /today or /week\n"
        "• 🧠 Structure your random thoughts & voice notes\n"
        "• 🎤 Send voice messages or audio files (mp3, m4a, wav, etc)\n"
        "• 📻 Transcribe long recordings (meetings, lectures, podcasts — any length!)\n"
        "• ✈️ Search flights, events, prices — just ask!\n"
        "• 👀 Watch prices — /watches to see active ones\n\n"
        "Commands:\n"
        "/today — today's schedule\n"
        "/week — this week's schedule\n"
        "/reminders — see active reminders\n"
        "/search <query> — force a web search\n"
        "/watches — see price watches\n"
        "/checkprices — check all watched prices\n"
        "/clear — clear conversation history\n"
        "/clearschedule — wipe schedule\n"
        "/clearreminders — wipe all reminders\n\n"
        "Or just start talking! 💬"
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
    await update.message.reply_text("🧹 Conversation history cleared!")


async def clear_schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_schedule({})
    await update.message.reply_text("🗑️ All schedule data cleared.")


async def clear_reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_reminders([])
    current_jobs = context.application.job_queue.jobs()
    for job in current_jobs:
        if hasattr(job, "name") and job.name and job.name.startswith("reminder_"):
            job.schedule_removal()
    await update.message.reply_text("🗑️ All reminders cleared.")


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

    user_id = update.effective_user.id
    msg = f"I searched for: \"{query}\"\n\nResults:\n\n{results}"
    response, _ = ask_claude(user_id, msg)

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

    user_id = update.effective_user.id
    all_results = []
    for watch in watches:
        query = watch.get("search_query", watch.get("description", ""))
        result = web_search(query)
        watch["last_checked"] = get_now().isoformat()
        watch["last_result"] = result[:500]
        all_results.append(f"{watch['description']}:\n{result}")

    save_price_watches(watches)
    combined = "\n---\n".join(all_results)
    msg = f"Price watch results. Summarize key findings and deals:\n\n{combined}"
    response, _ = ask_claude(user_id, msg)

    await send_long_message(update, response)


async def clear_watches_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_price_watches([])
    await update.message.reply_text("🗑️ All price watches cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    save_chat_id(user_id, chat_id)

    user_text = update.message.text
    logger.info(f"Text from {user_id}: {user_text[:100]}...")

    await update.message.chat.send_action("typing")

    response, reminder_data = ask_claude(user_id, user_text)

    if reminder_data:
        schedule_reminder_job(context.application, reminder_data, chat_id)

    await send_long_message(update, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages, audio messages, and audio files sent as documents."""
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    save_chat_id(user_id, chat_id)

    # Determine source: voice, audio, or document
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
        logger.info(f"Voice message from {user_id} ({file_size / 1024:.0f}KB)")
    elif audio:
        file_id = audio.file_id
        mime_type = audio.mime_type or ""
        file_name = audio.file_name or "audio"
        file_size = audio.file_size or 0
        logger.info(f"Audio '{file_name}' ({mime_type}, {file_size / (1024*1024):.1f}MB) from {user_id}")
    elif document and is_audio_document(document):
        file_id = document.file_id
        mime_type = document.mime_type or ""
        file_name = document.file_name or "file"
        file_size = document.file_size or 0
        logger.info(f"Audio doc '{file_name}' ({mime_type}, {file_size / (1024*1024):.1f}MB) from {user_id}")
    else:
        return

    # Check Telegram's file size limit (20MB for bot API)
    if file_size > TELEGRAM_FILE_LIMIT:
        await update.message.reply_text(
            f"⚠️ This file is {file_size / (1024*1024):.0f}MB. "
            f"Telegram Bot API has a 20MB download limit.\n\n"
            f"To transcribe larger files, please:\n"
            f"1. Split the file into parts under 20MB, or\n"
            f"2. Compress it (lower bitrate MP3), or\n"
            f"3. Send as multiple voice notes\n\n"
            f"Tip: A 2-hour recording as 64kbps MP3 is about 58MB. "
            f"Split into 3 parts of ~40 min each."
        )
        return

    # Size info for user
    size_text = f"{file_size / (1024*1024):.1f}MB" if file_size > 1024*1024 else f"{file_size / 1024:.0f}KB"
    await update.message.reply_text(f"🎤 Got your audio ({size_text}), processing...")
    await update.message.chat.send_action("typing")

    ext = get_file_extension(mime_type, file_name)

    # Download file
    file = await context.bot.get_file(file_id)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=str(TEMP_DIR)) as tmp:
        tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

    logger.info(f"Downloaded to {tmp_path} ({os.path.getsize(tmp_path) / (1024*1024):.1f}MB)")

    try:
        # Create a status callback to send progress updates
        async def status_update(msg: str):
            try:
                await update.message.reply_text(msg)
                await update.message.chat.send_action("typing")
            except Exception:
                pass

        # Transcribe with chunking support
        transcript = await transcribe_with_chunking(tmp_path, status_callback=status_update)

        if transcript is None:
            await update.message.reply_text(
                "⚠️ Couldn't transcribe this audio.\n\n"
                "Supported formats: MP3, M4A, MP4, WAV, OGG, FLAC, WEBM\n"
                "Try converting the file or sending as a voice note."
            )
            return

        logger.info(f"Transcription from {user_id}: {len(transcript)} chars")

        # For very long transcriptions, determine if it's a meeting/lecture vs brain dump
        transcript_length = len(transcript)
        if transcript_length > 5000:
            voice_message = (
                f"[LONG AUDIO TRANSCRIPTION — {transcript_length} characters]\n"
                f"{transcript}\n"
                f"[END TRANSCRIPTION]\n\n"
                f"This is a long recording. Please provide:\n"
                f"1. A concise summary (key points)\n"
                f"2. Action items (if any)\n"
                f"3. Notable quotes or important details\n"
                f"4. Any scheduling items or deadlines mentioned"
            )
        else:
            voice_message = (
                f"[VOICE NOTE TRANSCRIPTION]\n{transcript}\n[END VOICE NOTE]\n\n"
                f"Please structure and respond to this voice note."
            )

        await update.message.chat.send_action("typing")
        response, reminder_data = ask_claude(user_id, voice_message)

        if reminder_data:
            schedule_reminder_job(context.application, reminder_data, chat_id)

        # Send transcription separately if it's very long
        if transcript_length > 3000:
            # Send transcription in chunks
            await update.message.reply_text("📝 TRANSCRIPTION:")
            trans_chunks = [transcript[i:i + 4000] for i in range(0, len(transcript), 4000)]
            for chunk in trans_chunks:
                await update.message.reply_text(chunk)
            await update.message.reply_text("---\n\n📋 SUMMARY & ANALYSIS:")
            await send_long_message(update, response)
        else:
            full_response = f"📝 Transcription:\n{transcript}\n\n---\n\n{response}"
            await send_long_message(update, full_response)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route audio documents to the voice handler."""
    if not is_allowed(update.effective_user.id):
        return
    document = update.message.document
    if document and is_audio_document(document):
        await handle_voice(update, context)


# ── Startup ─────────────────────────────────────────────────────────────────
async def post_init(application: Application):
    # Clean up any leftover temp files
    for f in TEMP_DIR.glob("chunk_*"):
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
            reminder_dt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except (ValueError, KeyError):
            continue

        user_id = r.get("user_id")
        chat_id = get_chat_id(user_id)
        if not chat_id:
            continue

        delay = (reminder_dt - now).total_seconds()
        if delay <= 0:
            delay = 5

        job_data = {
            "chat_id": chat_id,
            "message": r.get("message", "Reminder!"),
            "user_id": user_id,
        }

        application.job_queue.run_once(
            send_reminder,
            when=delay,
            data=job_data,
            name=f"reminder_{user_id}_{r.get('message', '')[:20]}",
        )
        rescheduled += 1

    if rescheduled:
        logger.info(f"Rescheduled {rescheduled} pending reminders from disk")

    application.job_queue.run_repeating(
        check_saved_reminders,
        interval=60,
        first=10,
        name="reminder_checker",
    )
    logger.info("Started periodic reminder checker")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
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

    # Voice and audio messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Documents (catches mp3/m4a/etc sent as files)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Text messages (catch-all, MUST be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Shanti bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
