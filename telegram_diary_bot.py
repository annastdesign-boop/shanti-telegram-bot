import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import openai
from anthropic import Anthropic
from telegram import Update, ForceReply
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
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

# Optional: restrict bot to your user ID only
ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID")  # e.g. "123456789"

# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Persistent storage (simple JSON file) ────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCHEDULE_FILE = DATA_DIR / "schedule.json"
CONVERSATION_FILE = DATA_DIR / "conversations.json"


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


# ── Per-user conversation history ────────────────────────────────────────────
# In-memory store: { user_id_str: [ {role, content}, ... ] }
user_conversations: dict[str, list[dict]] = {}

MAX_HISTORY = 40  # max messages to keep per user in context window


def get_history(user_id: int) -> list[dict]:
    uid = str(user_id)
    if uid not in user_conversations:
        user_conversations[uid] = []
    return user_conversations[uid]


def append_message(user_id: int, role: str, content: str):
    history = get_history(user_id)
    history.append({"role": role, "content": content})
    # trim to last MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        user_conversations[str(user_id)] = history[-MAX_HISTORY:]


# ── Schedule helpers ─────────────────────────────────────────────────────────
def load_schedule() -> dict:
    """{ 'YYYY-MM-DD': [ { 'time': '...', 'task': '...', 'notes': '...' }, ... ] }"""
    return load_json(SCHEDULE_FILE, {})


def save_schedule(schedule: dict):
    save_json(SCHEDULE_FILE, schedule)


def get_today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get_today_schedule_text() -> str:
    schedule = load_schedule()
    today = get_today_key()
    items = schedule.get(today, [])
    if not items:
        return f"No plans scheduled for today ({today})."
    lines = [f"📅 **Schedule for {today}:**"]
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
    today = datetime.now()
    lines = ["📅 **This week's schedule:**"]
    found_anything = False
    for delta in range(7):
        day = today + timedelta(days=delta)
        key = day.strftime("%Y-%m-%d")
        day_name = day.strftime("%A %b %d")
        items = schedule.get(key, [])
        if items:
            found_anything = True
            lines.append(f"\n**{day_name}:**")
            for i, item in enumerate(items, 1):
                time_str = item.get("time", "")
                task = item.get("task", "")
                line = f"  {i}. [{time_str}] {task}" if time_str else f"  {i}. {task}"
                lines.append(line)
    if not found_anything:
        lines.append("Nothing scheduled for the next 7 days.")
    return "\n".join(lines)


# ── System prompt ────────────────────────────────────────────────────────────
def build_system_prompt(user_id: int) -> str:
    today_schedule = get_today_schedule_text()
    today = datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.now().strftime("%H:%M")

    return f"""You are Shanti — a warm, intelligent personal assistant for your user. You are NOT a generic chatbot. You are deeply personal, context-aware, and proactive.

Current date: {today}
Current time: {current_time}

── YOUR CORE CAPABILITIES ──

1. **SCHEDULING & PLANNING**
   - When the user mentions plans, appointments, tasks, or anything time-related, you EXTRACT the structured data and call it out clearly.
   - You respond with a confirmation AND output a JSON block that the system will parse.
   - Format for adding schedule items (put this at the END of your message, after your conversational response):
     ```SCHEDULE_ADD
     {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "description", "notes": "optional"}}
     ```
   - If the user says something like "tomorrow I have a meeting at 3pm about the project" — you confirm it warmly AND include the SCHEDULE_ADD block.
   - If the time is vague (like "morning"), make a reasonable estimate and note it.
   - If the date is vague (like "tomorrow", "next Monday"), calculate the actual date from today's date.

2. **DAILY SUMMARY**
   - When asked for a summary, plan for the day, or "what do I have today" — provide the schedule in a clean, readable format with any advice or reminders.
   
3. **THOUGHT STRUCTURING**
   - When the user dumps random thoughts, brain dumps, ideas, or stream-of-consciousness text/voice notes — you STRUCTURE it.
   - Organize into: key themes, action items, ideas to revisit, and emotional check-in if relevant.
   - Be thoughtful, not robotic. Reflect back what you notice.

4. **VOICE NOTES**
   - Voice messages are transcribed and sent to you. They may be messy, unstructured, with filler words. That's fine.
   - Treat voice messages as brain dumps unless they clearly contain specific requests.
   - Always acknowledge that you received a voice note and structure the content.

5. **REMOVING/EDITING SCHEDULE**
   - If the user wants to remove or change a scheduled item, output:
     ```SCHEDULE_REMOVE
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword to match"}}
     ```
   - Or for editing:
     ```SCHEDULE_EDIT
     {{"date": "YYYY-MM-DD", "task_keyword": "old keyword", "new_time": "HH:MM", "new_task": "new description"}}
     ```

── TODAY'S SCHEDULE ──
{today_schedule}

── PERSONALITY ──
- Warm, supportive, slightly witty
- You remember context from the conversation
- You're proactive: if someone mentions a deadline, remind them; if plans conflict, flag it
- Keep responses concise but thorough — no fluff, no generic AI phrases
- If you don't have enough info, ask a clarifying question
- Use emoji sparingly but naturally
- When the user is clearly just chatting/venting, be a good listener first, structurer second

── IMPORTANT RULES ──
- NEVER say "As an AI language model..." or similar disclaimers
- NEVER give generic responses. Always reference the user's actual context
- If you detect scheduling intent, ALWAYS include the SCHEDULE_ADD block
- The SCHEDULE blocks are parsed by code — they MUST be valid JSON
- Today is {today}. Calculate dates correctly.
"""


# ── Claude API call ──────────────────────────────────────────────────────────
def ask_claude(user_id: int, user_message: str) -> str:
    system_prompt = build_system_prompt(user_id)
    
    # Add user message to history
    append_message(user_id, "user", user_message)
    
    # Build messages for API
    messages = get_history(user_id)
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )
        
        assistant_text = response.content[0].text
        
        # Save assistant response to history
        append_message(user_id, "assistant", assistant_text)
        
        # Process any schedule commands in the response
        process_schedule_commands(assistant_text)
        
        # Clean the response (remove schedule command blocks from what user sees)
        clean_text = clean_response(assistant_text)
        
        return clean_text
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"⚠️ Something went wrong talking to Claude: {str(e)[:200]}"


# ── Schedule command processing ──────────────────────────────────────────────
def process_schedule_commands(text: str):
    schedule = load_schedule()
    changed = False

    # Process SCHEDULE_ADD
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
            # Sort by time
            schedule[date_key].sort(key=lambda x: x.get("time", "99:99"))
            changed = True
            logger.info(f"Added schedule item: {entry} on {date_key}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse SCHEDULE_ADD: {e}")

    # Process SCHEDULE_REMOVE
    if "```SCHEDULE_REMOVE" in text:
        try:
            block = text.split("```SCHEDULE_REMOVE")[1].split("```")[0].strip()
            data = json.loads(block)
            date_key = data["date"]
            keyword = data.get("task_keyword", "").lower()
            if date_key in schedule:
                original_len = len(schedule[date_key])
                schedule[date_key] = [
                    item for item in schedule[date_key]
                    if keyword not in item.get("task", "").lower()
                ]
                if len(schedule[date_key]) < original_len:
                    changed = True
                    logger.info(f"Removed schedule item matching '{keyword}' on {date_key}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse SCHEDULE_REMOVE: {e}")

    # Process SCHEDULE_EDIT
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
                        logger.info(f"Edited schedule item matching '{keyword}' on {date_key}")
                        break
                schedule[date_key].sort(key=lambda x: x.get("time", "99:99"))
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse SCHEDULE_EDIT: {e}")

    if changed:
        save_schedule(schedule)


def clean_response(text: str) -> str:
    """Remove SCHEDULE command blocks from the user-facing response."""
    clean = text
    for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT"]:
        while f"```{tag}" in clean:
            try:
                before = clean.split(f"```{tag}")[0]
                after = clean.split(f"```{tag}")[1].split("```", 1)
                remaining = after[1] if len(after) > 1 else ""
                clean = before + remaining
            except IndexError:
                break
    # Clean up extra whitespace
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


# ── Voice transcription ──────────────────────────────────────────────────────
async def transcribe_voice(file_path: str) -> str:
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        return transcript
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return None


# ── Access control ───────────────────────────────────────────────────────────
def is_allowed(user_id: int) -> bool:
    if ALLOWED_USER_ID is None:
        return True
    return str(user_id) == str(ALLOWED_USER_ID)


# ── Telegram handlers ───────────────────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ This bot is private.")
        return
    
    user_id = update.effective_user.id
    user_conversations[str(user_id)] = []  # Reset conversation
    
    await update.message.reply_text(
        "Hey! 👋 I'm Shanti, your personal assistant.\n\n"
        "Here's what I can do:\n"
        "• 📅 Schedule things — just tell me your plans naturally\n"
        "• 📋 Give you daily/weekly summaries — /today or /week\n"
        "• 🧠 Structure your random thoughts & voice notes\n"
        "• 🎤 Send me voice messages — I'll transcribe and organize them\n\n"
        "Commands:\n"
        "/today — today's schedule\n"
        "/week — this week's schedule\n"
        "/clear — clear conversation history\n"
        "/clearschedule — wipe all schedule data\n\n"
        "Or just start talking! Tell me what's on your mind. 💬"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = get_today_schedule_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = get_week_schedule_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    user_id = update.effective_user.id
    user_conversations[str(user_id)] = []
    await update.message.reply_text("🧹 Conversation history cleared. Fresh start!")


async def clear_schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_schedule({})
    await update.message.reply_text("🗑️ All schedule data cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    
    user_id = update.effective_user.id
    user_text = update.message.text
    
    logger.info(f"Text from {user_id}: {user_text[:100]}...")
    
    # Send typing indicator
    await update.message.chat.send_action("typing")
    
    response = ask_claude(user_id, user_text)
    
    # Split long messages (Telegram limit is 4096 chars)
    if len(response) > 4000:
        chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    
    user_id = update.effective_user.id
    
    await update.message.reply_text("🎤 Got your voice note, transcribing...")
    await update.message.chat.send_action("typing")
    
    # Download voice file
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("Hmm, couldn't get the audio. Try again?")
        return
    
    file = await context.bot.get_file(voice.file_id)
    
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
        await file.download_to_drive(tmp_path)
    
    try:
        transcript = await transcribe_voice(tmp_path)
        
        if transcript is None:
            await update.message.reply_text("⚠️ Couldn't transcribe the audio. Try again or type it out?")
            return
        
        logger.info(f"Voice transcription from {user_id}: {transcript[:100]}...")
        
        # Wrap the transcript so Claude knows it's from a voice note
        voice_message = (
            f"[VOICE NOTE TRANSCRIPTION]\n"
            f"{transcript}\n"
            f"[END VOICE NOTE]\n\n"
            f"Please structure and respond to this voice note."
        )
        
        await update.message.chat.send_action("typing")
        response = ask_claude(user_id, voice_message)
        
        # Prepend the transcription so user can verify
        full_response = f"📝 *Transcription:*\n_{transcript}_\n\n---\n\n{response}"
        
        if len(full_response) > 4000:
            # Send transcription first, then response
            await update.message.reply_text(f"📝 *Transcription:*\n_{transcript}_", parse_mode="Markdown")
            if len(response) > 4000:
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response)
        else:
            try:
                await update.message.reply_text(full_response, parse_mode="Markdown")
            except Exception:
                # Fallback without markdown if parsing fails
                await update.message.reply_text(full_response)
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("week", week_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("clearschedule", clear_schedule_command))
    
    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    
    # Text messages (catch-all, must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    logger.info("Shanti bot is starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
