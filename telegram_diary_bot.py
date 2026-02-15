#!/usr/bin/env python3
"""
Telegram Diary/Assistant Bot with Claude API
Optimized for minimal API usage and cost
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import sqlite3
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import anthropic

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class DiaryDatabase:
    """SQLite database for storing diary entries and reminders"""
    
    def __init__(self, db_path: str = "diary.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Diary entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general'
            )
        """)
        
        # Reminders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reminder_date DATE NOT NULL,
                reminder_time TIME,
                content TEXT NOT NULL,
                completed BOOLEAN DEFAULT 0,
                notified BOOLEAN DEFAULT 0
            )
        """)
        
        # Conversation context table (to reduce API calls)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context (
                user_id INTEGER PRIMARY KEY,
                last_summary TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_entry(self, user_id: int, content: str, category: str = "general"):
        """Add a diary entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO entries (user_id, content, category) VALUES (?, ?, ?)",
            (user_id, content, category)
        )
        conn.commit()
        conn.close()
    
    def add_reminder(self, user_id: int, reminder_date: str, content: str, reminder_time: str = None):
        """Add a reminder"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (user_id, reminder_date, reminder_time, content) VALUES (?, ?, ?, ?)",
            (user_id, reminder_date, reminder_time, content)
        )
        conn.commit()
        conn.close()
    
    def get_recent_entries(self, user_id: int, days: int = 7) -> List[Dict]:
        """Get recent entries for context (limited to reduce API costs)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT timestamp, content, category FROM entries 
               WHERE user_id = ? AND timestamp > datetime('now', '-' || ? || ' days')
               ORDER BY timestamp DESC LIMIT 10""",
            (user_id, days)
        )
        entries = [{"timestamp": row[0], "content": row[1], "category": row[2]} 
                   for row in cursor.fetchall()]
        conn.close()
        return entries
    
    def get_pending_reminders(self, user_id: int) -> List[Dict]:
        """Get pending reminders"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, reminder_date, reminder_time, content FROM reminders 
               WHERE user_id = ? AND completed = 0 
               ORDER BY reminder_date, reminder_time""",
            (user_id,)
        )
        reminders = [{"id": row[0], "date": row[1], "time": row[2], "content": row[3]} 
                     for row in cursor.fetchall()]
        conn.close()
        return reminders
    
    def get_due_reminders(self) -> List[Dict]:
        """Get reminders that are due for notification"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        today = datetime.now().date().isoformat()
        cursor.execute(
            """SELECT id, user_id, content, reminder_time FROM reminders 
               WHERE reminder_date <= ? AND completed = 0 AND notified = 0""",
            (today,)
        )
        reminders = [{"id": row[0], "user_id": row[1], "content": row[2], "time": row[3]} 
                     for row in cursor.fetchall()]
        conn.close()
        return reminders
    
    def mark_reminder_notified(self, reminder_id: int):
        """Mark reminder as notified"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE reminders SET notified = 1 WHERE id = ?", (reminder_id,))
        conn.commit()
        conn.close()
    
    def complete_reminder(self, reminder_id: int):
        """Mark reminder as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
        conn.commit()
        conn.close()
    
    def get_context_summary(self, user_id: int) -> Optional[str]:
        """Get cached context summary to reduce API calls"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_summary FROM context WHERE user_id = ?",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def update_context_summary(self, user_id: int, summary: str):
        """Update cached context summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO context (user_id, last_summary, last_updated) 
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET 
               last_summary = ?, last_updated = CURRENT_TIMESTAMP""",
            (user_id, summary, summary)
        )
        conn.commit()
        conn.close()


class ClaudeAssistant:
    """Claude API integration optimized for minimal token usage"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-haiku-4-5-20251001"  # Cheapest model
    
    async def analyze_message(self, message: str, context: Optional[str] = None) -> Dict:
        """
        Analyze user message to extract:
        - Diary entries
        - Tasks/reminders
        - Creative ideas
        - Questions requiring response
        
        Uses minimal tokens by being specific in prompt
        """
        system_prompt = """You are a personal diary assistant. Analyze the user's message and extract:
1. Main diary content (what they did/plan to do)
2. Any reminders/tasks with dates (extract date if mentioned)
3. Creative ideas or goals
4. Whether they need advice/brainstorming

Respond in JSON format only:
{
  "diary_entry": "summary of what happened/plans",
  "category": "work|personal|creative|health|other",
  "reminders": [{"content": "task", "date": "YYYY-MM-DD or 'today'/'tomorrow'/'tuesday' etc", "priority": "high|normal"}],
  "ideas": ["idea1", "idea2"],
  "needs_response": true|false,
  "question": "what they're asking if needs_response is true"
}"""
        
        user_message = message
        if context:
            user_message = f"Recent context: {context}\n\nNew message: {message}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,  # Keep low to minimize cost
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            # Parse JSON response
            content = response.content[0].text
            return json.loads(content)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "diary_entry": message,
                "category": "general",
                "reminders": [],
                "ideas": [],
                "needs_response": False,
                "question": None
            }
    
    async def provide_advice(self, question: str, ideas: List[str], context: str) -> str:
        """Provide creative advice and action plans"""
        system_prompt = """You are a helpful personal assistant. Provide:
1. Structured feedback on their ideas
2. Actionable steps to achieve their goals
3. Creative suggestions
Keep response concise (under 200 words)."""
        
        user_message = f"""Context: {context}

Ideas/Goals: {', '.join(ideas)}

Question: {question}

Provide helpful, structured advice."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "I'm having trouble connecting right now. Try again in a moment!"


class DiaryBot:
    """Main Telegram bot class"""
    
    def __init__(self, telegram_token: str, anthropic_key: str):
        self.db = DiaryDatabase()
        self.claude = ClaudeAssistant(anthropic_key)
        self.application = Application.builder().token(telegram_token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("reminders", self.show_reminders))
        self.application.add_handler(CommandHandler("done", self.complete_reminder))
        self.application.add_handler(CommandHandler("summary", self.weekly_summary))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Schedule reminder checks
        job_queue = self.application.job_queue
        job_queue.run_repeating(self.check_reminders, interval=3600, first=10)  # Every hour
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "👋 Welcome to your Personal Diary Assistant!\n\n"
            "I'll help you:\n"
            "✍️ Keep a diary of your thoughts and plans\n"
            "💡 Organize your creative ideas\n"
            "⏰ Remember important tasks\n"
            "🎯 Create action plans for your goals\n\n"
            "Just send me messages about your day, ideas, or tasks!\n\n"
            "Commands:\n"
            "/reminders - View pending reminders\n"
            "/done <id> - Mark reminder as complete\n"
            "/summary - Get weekly summary\n"
            "/help - Show this message"
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await update.message.reply_text(
            "📖 **How to use your diary assistant:**\n\n"
            "**Diary Entries:**\n"
            "Just type what you did today or your plans\n"
            "_Example: Today I finished the project presentation_\n\n"
            "**Reminders:**\n"
            "Mention tasks with dates\n"
            "_Example: I need to register for yoga class on Tuesday_\n\n"
            "**Creative Ideas:**\n"
            "Share your ideas and ask for help\n"
            "_Example: I want to start a blog about cooking. How should I begin?_\n\n"
            "**Commands:**\n"
            "/reminders - See all pending tasks\n"
            "/done <number> - Mark task complete\n"
            "/summary - Get this week's overview",
            parse_mode='Markdown'
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        user_id = update.effective_user.id
        message = update.message.text
        
        # Show typing indicator
        await update.message.chat.send_action("typing")
        
        # Get recent context (minimal to save tokens)
        recent_entries = self.db.get_recent_entries(user_id, days=3)
        context_text = "; ".join([e["content"][:100] for e in recent_entries[:3]]) if recent_entries else None
        
        # Analyze with Claude (optimized prompt)
        analysis = await self.claude.analyze_message(message, context_text)
        
        # Save diary entry
        self.db.add_entry(user_id, analysis["diary_entry"], analysis["category"])
        
        # Process reminders
        reminder_count = 0
        for reminder in analysis["reminders"]:
            reminder_date = self.parse_date(reminder["date"])
            if reminder_date:
                self.db.add_reminder(user_id, reminder_date, reminder["content"])
                reminder_count += 1
        
        # Build response
        response_parts = ["✅ Noted!"]
        
        if reminder_count > 0:
            response_parts.append(f"\n⏰ Set {reminder_count} reminder(s)")
        
        if analysis["ideas"]:
            response_parts.append(f"\n💡 Ideas captured: {len(analysis['ideas'])}")
        
        # Only call Claude again if user needs advice (saves API calls!)
        if analysis["needs_response"] and analysis["question"]:
            advice = await self.claude.provide_advice(
                analysis["question"],
                analysis["ideas"],
                context_text or ""
            )
            response_parts.append(f"\n\n{advice}")
        
        await update.message.reply_text("\n".join(response_parts))
    
    async def show_reminders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show pending reminders"""
        user_id = update.effective_user.id
        reminders = self.db.get_pending_reminders(user_id)
        
        if not reminders:
            await update.message.reply_text("📭 No pending reminders!")
            return
        
        response = "⏰ **Your Reminders:**\n\n"
        for i, reminder in enumerate(reminders, 1):
            date = reminder["date"]
            time = f" at {reminder['time']}" if reminder["time"] else ""
            response += f"{i}. {reminder['content']}\n   📅 {date}{time}\n\n"
        
        response += "\nUse /done <number> to mark as complete"
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def complete_reminder(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mark reminder as complete"""
        user_id = update.effective_user.id
        
        try:
            reminder_num = int(context.args[0])
            reminders = self.db.get_pending_reminders(user_id)
            
            if 1 <= reminder_num <= len(reminders):
                reminder_id = reminders[reminder_num - 1]["id"]
                self.db.complete_reminder(reminder_id)
                await update.message.reply_text("✅ Reminder marked as complete!")
            else:
                await update.message.reply_text("❌ Invalid reminder number")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /done <number>")
    
    async def weekly_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Provide weekly summary without calling API (saves money)"""
        user_id = update.effective_user.id
        entries = self.db.get_recent_entries(user_id, days=7)
        reminders = self.db.get_pending_reminders(user_id)
        
        if not entries:
            await update.message.reply_text("📭 No entries this week!")
            return
        
        response = f"📊 **This Week's Summary**\n\n"
        response += f"✍️ {len(entries)} diary entries\n"
        response += f"⏰ {len(reminders)} pending reminders\n\n"
        
        # Group by category
        categories = {}
        for entry in entries:
            cat = entry["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        response += "**Categories:**\n"
        for cat, count in categories.items():
            response += f"• {cat}: {count}\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def check_reminders(self, context: ContextTypes.DEFAULT_TYPE):
        """Background job to check and send reminders"""
        due_reminders = self.db.get_due_reminders()
        
        for reminder in due_reminders:
            try:
                await context.bot.send_message(
                    chat_id=reminder["user_id"],
                    text=f"⏰ **Reminder!**\n\n{reminder['content']}",
                    parse_mode='Markdown'
                )
                self.db.mark_reminder_notified(reminder["id"])
            except Exception as e:
                logger.error(f"Failed to send reminder: {e}")
    
    def parse_date(self, date_str: str) -> Optional[str]:
        """Parse natural language dates"""
        today = datetime.now().date()
        date_str = date_str.lower().strip()
        
        if date_str == "today":
            return today.isoformat()
        elif date_str == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        elif date_str in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target_day = days.index(date_str)
            current_day = today.weekday()
            days_ahead = (target_day - current_day) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week
            return (today + timedelta(days=days_ahead)).isoformat()
        
        # Try to parse YYYY-MM-DD format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except:
            pass
        
        return None
    
    def run(self):
        """Start the bot"""
        logger.info("Starting Diary Bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    # Load environment variables
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    if not TELEGRAM_TOKEN or not ANTHROPIC_KEY:
        print("ERROR: Please set TELEGRAM_BOT_TOKEN and ANTHROPIC_API_KEY environment variables")
        print("\nOn Linux/Mac:")
        print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("  export ANTHROPIC_API_KEY='your_key_here'")
        print("\nOn Windows:")
        print("  set TELEGRAM_BOT_TOKEN=your_token_here")
        print("  set ANTHROPIC_API_KEY=your_key_here")
        return
    
    bot = DiaryBot(TELEGRAM_TOKEN, ANTHROPIC_KEY)
    bot.run()


if __name__ == "__main__":
    main()
