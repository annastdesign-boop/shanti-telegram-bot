import os
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import openai
from anthropic import Anthropic
from tavily import TavilyClient
from telegram import Update
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
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Optional: restrict bot to your user ID only
ALLOWED_USER_ID = os.environ.get("ALLOWED_USER_ID")

# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ── Persistent storage ──────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCHEDULE_FILE = DATA_DIR / "schedule.json"
PRICE_WATCHES_FILE = DATA_DIR / "price_watches.json"


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


# ── Price watch helpers ──────────────────────────────────────────────────────
def load_price_watches() -> list:
    return load_json(PRICE_WATCHES_FILE, [])


def save_price_watches(watches: list):
    save_json(PRICE_WATCHES_FILE, watches)


def get_price_watches_text() -> str:
    watches = load_price_watches()
    if not watches:
        return "No active price watches."
    lines = ["👀 **Active Price Watches:**"]
    for i, w in enumerate(watches, 1):
        lines.append(f"{i}. {w.get('description', 'unknown')} — last checked: {w.get('last_checked', 'never')}")
    return "\n".join(lines)


# ── Tavily web search ────────────────────────────────────────────────────────
def web_search(query: str, search_depth: str = "advanced", max_results: int = 5) -> str:
    """Perform a web search using Tavily and return formatted results."""
    try:
        logger.info(f"Tavily search: {query}")
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
        )

        results_text = ""

        # Tavily's AI-generated answer
        if response.get("answer"):
            results_text += f"**AI Summary:** {response['answer']}\n\n"

        results_text += "**Sources:**\n"
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "..."
            results_text += f"{i}. **{title}**\n   {url}\n   {content}\n\n"

        return results_text

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Search failed: {str(e)[:200]}"


def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flight prices."""
    query = f"cheapest flights from {origin} to {destination} {date} prices booking"
    return web_search(query, search_depth="advanced", max_results=5)


def search_events(location: str, date_range: str = "") -> str:
    """Search for events in a location."""
    query = f"events in {location} {date_range} concerts festivals things to do"
    return web_search(query, search_depth="advanced", max_results=5)


# ── Two-step Claude: decide if search needed, then respond ───────────────────
SEARCH_DECISION_PROMPT = """You are a routing assistant. Your ONLY job is to decide if the user's message needs a live web search.

Reply with EXACTLY one JSON object, nothing else.

If search IS needed:
{"needs_search": true, "search_queries": ["query 1", "query 2"], "search_type": "general|flights|events|prices"}

If search is NOT needed:
{"needs_search": false}

Examples of messages that NEED search:
- "Find me flights from Berlin to Tokyo in March"
- "What events are happening in Barcelona this weekend?"
- "How much are tickets to Burning Man?"
- "Check the price of flights to Lisbon"
- "What concerts are in NYC next month?"
- "Search for cheap hotels in Amsterdam"
- "What's the weather in Paris?"
- "Latest news about tech layoffs"
- "Find restaurants near me in Kreuzberg"

Examples that do NOT need search:
- "Schedule a meeting tomorrow at 3pm"
- "What do I have planned today?"
- "I'm feeling stressed about work"
- "Remind me to buy groceries"
- "Organize my thoughts about the project"
- General conversation, scheduling, thought structuring
"""


def decide_if_search_needed(user_message: str) -> dict:
    """Ask Claude to decide if web search is needed."""
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SEARCH_DECISION_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text.strip()

        # Try to extract JSON from response
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
    today = datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.now().strftime("%H:%M")
    price_watches = get_price_watches_text()

    search_section = ""
    if search_context:
        search_section = f"""
── LIVE SEARCH RESULTS (just fetched from the web) ──
{search_context}

IMPORTANT: Use these search results to answer the user's question. Cite specific prices, links, and dates from the results. If the results are not relevant enough, say so and suggest a better search query.
"""

    return f"""You are Shanti — a warm, intelligent personal assistant. You are NOT a generic chatbot. You are deeply personal, context-aware, and proactive.

Current date: {today}
Current time: {current_time}

── YOUR CORE CAPABILITIES ──

1. **SCHEDULING & PLANNING**
   - When the user mentions plans, appointments, tasks, or anything time-related, EXTRACT the structured data.
   - Format for adding schedule items (put at END of your message):
     ```SCHEDULE_ADD
     {{"date": "YYYY-MM-DD", "time": "HH:MM", "task": "description", "notes": "optional"}}
     ```
   - If time is vague ("morning"), estimate. If date is vague ("tomorrow"), calculate from today.

2. **DAILY SUMMARY**
   - When asked for summary/plan — provide schedule in clean format with advice.

3. **THOUGHT STRUCTURING**
   - When user dumps random thoughts/brain dumps — STRUCTURE into: key themes, action items, ideas to revisit, emotional check-in if relevant.

4. **VOICE NOTES**
   - Voice messages are transcribed. They may be messy. Treat as brain dumps unless clearly specific requests.
   - Always acknowledge and structure the content.

5. **WEB SEARCH & LIVE INFO**
   - You now have access to live web search results when relevant.
   - When search results are provided below, USE THEM to give specific, helpful answers.
   - Include specific prices, dates, links from the results.
   - Be honest about what the search found vs. what you're estimating.

6. **PRICE WATCHING**
   - If the user wants you to track a price (flights, events, etc.), output:
     ```PRICE_WATCH
     {{"description": "what to track", "search_query": "search query to use", "frequency": "daily|weekly"}}
     ```

7. **REMOVING/EDITING SCHEDULE**
   - Remove:
     ```SCHEDULE_REMOVE
     {{"date": "YYYY-MM-DD", "task_keyword": "keyword to match"}}
     ```
   - Edit:
     ```SCHEDULE_EDIT
     {{"date": "YYYY-MM-DD", "task_keyword": "old keyword", "new_time": "HH:MM", "new_task": "new description"}}
     ```

── TODAY'S SCHEDULE ──
{today_schedule}

── PRICE WATCHES ──
{price_watches}
{search_section}
── PERSONALITY ──
- Warm, supportive, slightly witty
- Remember conversation context
- Proactive: flag conflicts, remind deadlines
- Concise but thorough — no fluff
- If not enough info, ask clarifying questions
- Use emoji sparingly but naturally
- When user is chatting/venting, listen first, structure second
- When sharing search results, format them nicely with links

── IMPORTANT RULES ──
- NEVER say "As an AI language model..." or similar
- NEVER say you can't search the web — you CAN via live search
- NEVER give generic responses — reference actual context
- If you detect scheduling intent, ALWAYS include SCHEDULE_ADD block
- Schedule/price blocks must be valid JSON
- Today is {today}. Calculate dates correctly.
"""


# ── Claude API call ──────────────────────────────────────────────────────────
def ask_claude(user_id: int, user_message: str) -> str:
    # Step 1: Decide if search is needed
    search_decision = decide_if_search_needed(user_message)
    search_context = ""

    if search_decision.get("needs_search"):
        queries = search_decision.get("search_queries", [])
        search_type = search_decision.get("search_type", "general")

        logger.info(f"Search needed. Type: {search_type}, Queries: {queries}")

        all_results = []
        for query in queries[:3]:  # Max 3 searches per message
            result = web_search(query)
            all_results.append(f"**Search: {query}**\n{result}")

        search_context = "\n---\n".join(all_results)

    # Step 2: Build system prompt with search context
    system_prompt = build_system_prompt(user_id, search_context)

    # Add user message to history
    append_message(user_id, "user", user_message)

    # Build messages for API
    messages = get_history(user_id)

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=system_prompt,
            messages=messages,
        )

        assistant_text = response.content[0].text

        # Save assistant response to history
        append_message(user_id, "assistant", assistant_text)

        # Process schedule and price watch commands
        process_schedule_commands(assistant_text)
        process_price_watch_commands(assistant_text)

        # Clean response
        clean_text = clean_response(assistant_text)

        return clean_text

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"⚠️ Something went wrong talking to Claude: {str(e)[:200]}"


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
            logger.info(f"Added schedule item: {entry} on {date_key}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse SCHEDULE_ADD: {e}")

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
            "created": datetime.now().isoformat(),
            "last_checked": "never",
            "last_result": "",
        })
        save_price_watches(watches)
        logger.info(f"Added price watch: {data.get('description')}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse PRICE_WATCH: {e}")


def clean_response(text: str) -> str:
    clean = text
    for tag in ["SCHEDULE_ADD", "SCHEDULE_REMOVE", "SCHEDULE_EDIT", "PRICE_WATCH"]:
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
    user_conversations[str(user_id)] = []

    await update.message.reply_text(
        "Hey! 👋 I'm Shanti, your personal assistant.\n\n"
        "Here's what I can do:\n"
        "• 📅 Schedule things — just tell me your plans naturally\n"
        "• 📋 Daily/weekly summaries — /today or /week\n"
        "• 🧠 Structure your random thoughts & voice notes\n"
        "• 🎤 Send voice messages — I'll transcribe and organize\n"
        "• ✈️ Search flights, events, prices — just ask!\n"
        "• 👀 Watch prices for you — /watches to see active ones\n\n"
        "Commands:\n"
        "/today — today's schedule\n"
        "/week — this week's schedule\n"
        "/search <query> — force a web search\n"
        "/watches — see active price watches\n"
        "/checkprices — manually check all watched prices\n"
        "/clear — clear conversation history\n"
        "/clearschedule — wipe all schedule data\n\n"
        "Or just start talking! 💬"
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


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /search <your query>\nExample: /search flights Berlin to Tokyo March 2025")
        return

    await update.message.chat.send_action("typing")
    results = web_search(query)

    # Send results through Claude for nice formatting
    user_id = update.effective_user.id
    message = f"I just searched the web for: \"{query}\"\n\nHere are the results, please summarize them nicely:\n\n{results}"
    response = ask_claude(user_id, message)

    if len(response) > 4000:
        chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(response)


async def watches_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    text = get_price_watches_text()
    await update.message.reply_text(text, parse_mode="Markdown")


async def check_prices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    watches = load_price_watches()
    if not watches:
        await update.message.reply_text("No active price watches. Tell me what to track!")
        return

    await update.message.reply_text(f"🔍 Checking {len(watches)} price watch(es)...")
    await update.message.chat.send_action("typing")

    user_id = update.effective_user.id
    all_results = []

    for i, watch in enumerate(watches):
        query = watch.get("search_query", watch.get("description", ""))
        result = web_search(query)
        watch["last_checked"] = datetime.now().isoformat()
        watch["last_result"] = result[:500]
        all_results.append(f"**{watch['description']}:**\n{result}")

    save_price_watches(watches)

    combined = "\n---\n".join(all_results)
    message = f"Here are the latest results for my price watches. Summarize the key findings, highlight any good deals:\n\n{combined}"
    response = ask_claude(user_id, message)

    if len(response) > 4000:
        chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for chunk in chunks:
            await update.message.reply_text(chunk)
    else:
        await update.message.reply_text(response)


async def clear_watches_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    save_price_watches([])
    await update.message.reply_text("🗑️ All price watches cleared.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    user_id = update.effective_user.id
    user_text = update.message.text

    logger.info(f"Text from {user_id}: {user_text[:100]}...")

    await update.message.chat.send_action("typing")

    response = ask_claude(user_id, user_text)

    if len(response) > 4000:
        chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
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
            await update.message.reply_text("⚠️ Couldn't transcribe. Try again or type it out?")
            return

        logger.info(f"Voice transcription from {user_id}: {transcript[:100]}...")

        voice_message = (
            f"[VOICE NOTE TRANSCRIPTION]\n"
            f"{transcript}\n"
            f"[END VOICE NOTE]\n\n"
            f"Please structure and respond to this voice note."
        )

        await update.message.chat.send_action("typing")
        response = ask_claude(user_id, voice_message)

        full_response = f"📝 *Transcription:*\n_{transcript}_\n\n---\n\n{response}"

        if len(full_response) > 4000:
            await update.message.reply_text(
                f"📝 *Transcription:*\n_{transcript}_",
                parse_mode="Markdown",
            )
            if len(response) > 4000:
                chunks = [response[i:i + 4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response)
        else:
            try:
                await update.message.reply_text(full_response, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(full_response)

    finally:
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
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("watches", watches_command))
    app.add_handler(CommandHandler("checkprices", check_prices_command))
    app.add_handler(CommandHandler("clearwatches", clear_watches_command))

    # Voice messages
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Text messages (catch-all, must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Shanti bot is starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
