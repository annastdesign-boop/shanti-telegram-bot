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
from openai import OpenAI
import tempfile

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
        
        # Morning news preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_preferences (
                user_id INTEGER PRIMARY KEY,
                enabled BOOLEAN DEFAULT 0,
                time TIME DEFAULT '08:00',
                topics TEXT DEFAULT 'general,technology,health',
                timezone TEXT DEFAULT 'UTC'
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
    
    def set_morning_news(self, user_id: int, enabled: bool, time: str = "08:00", topics: str = "general"):
        """Set morning news preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO news_preferences (user_id, enabled, time, topics) 
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET 
               enabled = ?, time = ?, topics = ?""",
            (user_id, enabled, time, topics, enabled, time, topics)
        )
        conn.commit()
        conn.close()
    
    def get_news_subscribers(self, current_hour: int) -> List[Dict]:
        """Get users who should receive news at this hour"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Get users whose news time matches current hour
        cursor.execute(
            """SELECT user_id, topics FROM news_preferences 
               WHERE enabled = 1 AND substr(time, 1, 2) = ?""",
            (f"{current_hour:02d}",)
        )
        subscribers = [{"user_id": row[0], "topics": row[1]} for row in cursor.fetchall()]
        conn.close()
        return subscribers
    
    def get_news_preference(self, user_id: int) -> Optional[Dict]:
        """Get user's news preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT enabled, time, topics FROM news_preferences WHERE user_id = ?",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return {"enabled": bool(result[0]), "time": result[1], "topics": result[2]}
        return None


class VoiceTranscriber:
    """OpenAI Whisper API integration for voice-to-text"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    async def transcribe_voice(self, voice_file_bytes: bytes, filename: str = "voice.ogg") -> Optional[str]:
        """
        Transcribe voice message to text using Whisper API
        
        Args:
            voice_file_bytes: Audio file bytes from Telegram
            filename: Original filename (default: voice.ogg)
        
        Returns:
            Transcribed text or None if error
        """
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
                temp_audio.write(voice_file_bytes)
                temp_audio_path = temp_audio.name
            
            # Transcribe using Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None


class ClaudeAssistant:
    """Claude API integration optimized for minimal token usage"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"  # Better quality for conversations!
    
    async def analyze_message(self, message: str, context: Optional[str] = None) -> Dict:
        """
        Analyze message AND provide response in one go - like a real conversation!
        """
        system_prompt = """You are a warm, engaging personal assistant combining: life coach, therapist, project manager, business advisor, Buddhist teacher, and art therapist.

RESPOND LIKE A HELPFUL FRIEND - conversational, warm, insightful, and actionable.

Your response should:
1. Acknowledge what they shared with empathy
2. Provide helpful guidance, encouragement, or advice
3. Be natural and conversational (not robotic)
4. Include Buddhist wisdom or art therapy when relevant
5. Ask thoughtful follow-up questions
6. Be encouraging and supportive

Length: 150-400 words depending on complexity

Also extract structured data for the diary system.

Respond in JSON format:
{
  "response": "Your warm, helpful, conversational response to them - THIS IS MOST IMPORTANT",
  "diary_entry": "brief summary for diary",
  "category": "work|personal|creative|health|other",
  "reminders": [{"content": "task", "date": "YYYY-MM-DD or 'today'/'tomorrow'", "priority": "high|normal"}],
  "ideas": ["idea1", "idea2"],
  "wants_morning_news": false,
  "news_time": "08:00"
}

CRITICAL: The "response" field should ALWAYS have a helpful, encouraging reply. Never leave it empty!

Examples:
User: "I'm thinking about changing careers"
Response: "That's a big decision! Career changes can feel both exciting and scary. What's driving this desire to change? Is it the work itself, the environment, or something else? 

Let me share something from Buddhist wisdom: Lama Zopa Rinpoche teaches that change is impermanence in action - resisting it creates suffering. But that doesn't mean rushing in blindly!

Practical steps:
1. Clarify what you want (not just what you're leaving)
2. Identify transferable skills
3. Maybe explore through a side project first?

What field are you drawn to? 💫"

User: "Finished my presentation today"
Response: "Awesome! 🎉 Completing that presentation is a real win. How did it feel to get it done? 

This is great momentum - what's the next milestone you want to hit while you're on a roll? Sometimes finishing one thing energizes us for the next.

From a dharma perspective, celebrating accomplishments without attachment is a practice - enjoy the satisfaction while staying present. Well done! ✨"
"""
        
        user_message = message
        if context:
            user_message = f"Recent context: {context}\n\nNew message: {message}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,  # Much higher for real conversations!
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            
            # Parse JSON response
            content = response.content[0].text
            return json.loads(content)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {
                "response": "I'm here and listening! Tell me more about what's on your mind. 💭",
                "diary_entry": message,
                "category": "general",
                "reminders": [],
                "ideas": [],
                "wants_morning_news": False,
                "news_time": "08:00"
            }
    
    async def provide_advice(self, question: str, ideas: List[str], context: str) -> str:
        """Provide coaching, advice and action plans with Buddhist wisdom and art therapy"""
        system_prompt = """You are an AI combining deep expertise as:
- Life Coach: Motivation, goal-setting, habit formation
- Therapist: Emotional support, stress management, perspective
- Project Manager: Planning, prioritization, execution strategies  
- Business Advisor: Strategy, growth, problem-solving
- Indo-Tibetan Buddhist Teacher (Mahayana tradition): Wisdom from Dalai Lama, Lama Zopa Rinpoche, Urgyen Rinpoche, and other great masters
- Art Therapist: Creative expression, healing through art, mindfulness in creativity

When appropriate, weave in:
- Buddhist concepts: compassion, impermanence, mindfulness, emptiness, bodhicitta
- Meditation practices and contemplations
- Art therapy techniques: drawing feelings, mandala creation, color therapy, journaling
- Tibetan Buddhist wisdom on suffering, acceptance, loving-kindness
- Practical dharma teachings applicable to modern life

Provide personalized, actionable guidance that:
1. Acknowledges their situation with empathy and compassion
2. Offers spiritual perspective when helpful (without being preachy)
3. Suggests creative/artistic approaches for emotional processing
4. Provides practical next steps
5. Encourages and motivates with wisdom
6. Asks thoughtful follow-up questions when helpful
7. Celebrates progress and wins

Balance practical action with spiritual wisdom. Sometimes the answer is meditation, 
sometimes it's a to-do list, often it's both.

Tone: Warm, compassionate, wise, encouraging, grounded
Length: 150-300 words (concise but complete)
Format: Natural conversation, use emojis sparingly for warmth"""
        
        user_message = f"""Recent context: {context}

Current situation/ideas: {', '.join(ideas) if ideas else 'N/A'}

What they shared: {question}

Provide helpful coaching with wisdom and actionable advice."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,  # Increased for fuller responses with wisdom
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "I'm having trouble connecting right now. Try again in a moment!"
    
    async def search_and_answer(self, query: str, context: str = "") -> str:
        """
        Search the web and provide answer with sources
        Used for: flight prices, current info, research, fact-checking
        """
        system_prompt = """You are a helpful assistant with web search capabilities.
        
When the user asks about:
- Flight prices, hotel rates, current prices
- Current events, news, recent information  
- Research topics requiring up-to-date data
- Factual information that may have changed

You will search the web and provide:
1. Clear, accurate answer
2. Relevant details
3. Sources/links when available
4. Practical next steps

Keep responses concise (200-300 words) but informative."""
        
        search_query = f"""Context: {context}

User query: {query}

Search the web for current information and provide a helpful answer."""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": search_query}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API search error: {e}")
            return "I'm having trouble searching right now. Could you try rephrasing your question?"
    
    async def generate_morning_news(self, topics: str = "general") -> str:
        """Generate personalized morning news briefing"""
        system_prompt = """You are a morning news curator. Create a brief, engaging news summary.

Include:
- Top 3-5 most important stories
- Mix of topics: world events, technology, health, business
- Positive story at the end
- Brief, digestible format
- Encouraging tone to start the day

Format:
🌅 Good Morning! Here's what's happening today:

[Story summaries - 2-3 sentences each]

Keep it under 300 words total. Inspiring and informative!"""
        
        user_message = f"Create a morning news briefing. Focus on: {topics}. Include current date context."
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API news error: {e}")
            return "☀️ Good morning! I'm having trouble fetching news right now. Have a great day!"


class DiaryBot:
    """Main Telegram bot class"""
    
    # Add __weakref__ slot to support weak references in Python 3.13
    __slots__ = ('db', 'claude', 'whisper', 'application', '__weakref__')
    
    def __init__(self, telegram_token: str, anthropic_key: str, openai_key: str):
        self.db = DiaryDatabase()
        self.claude = ClaudeAssistant(anthropic_key)
        self.whisper = VoiceTranscriber(openai_key)
        self.application = Application.builder().token(telegram_token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("reminders", self.show_reminders))
        self.application.add_handler(CommandHandler("done", self.complete_reminder))
        self.application.add_handler(CommandHandler("summary", self.weekly_summary))
        self.application.add_handler(CommandHandler("news", self.news_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))  # Voice messages
        
        # Schedule reminder checks and morning news
        job_queue = self.application.job_queue
        if job_queue:
            job_queue.run_repeating(self.check_reminders, interval=3600, first=10)  # Every hour
            job_queue.run_repeating(self.send_morning_news, interval=3600, first=60)  # Check every hour for news
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "👋 Welcome to your Personal Diary Assistant!\n\n"
            "I'll help you:\n"
            "✍️ Keep a diary of your thoughts and plans\n"
            "💡 Organize your creative ideas\n"
            "⏰ Remember important tasks\n"
            "🎯 Create action plans for your goals\n"
            "🎤 Voice messages supported!\n"
            "📰 Morning news briefings!\n\n"
            "Just send me text or voice messages about your day, ideas, or tasks!\n\n"
            "Commands:\n"
            "/reminders - View pending reminders\n"
            "/done <id> - Mark reminder as complete\n"
            "/summary - Get weekly summary\n"
            "/news - Setup morning news\n"
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
        
        # Get the AI's conversational response (it's already in the analysis!)
        ai_response = analysis.get("response", "")
        
        # Build response parts
        response_parts = []
        
        # Add metadata tags if relevant
        metadata = []
        if reminder_count > 0:
            metadata.append(f"⏰ {reminder_count} reminder(s) set")
        if analysis.get("ideas"):
            metadata.append(f"💡 {len(analysis['ideas'])} idea(s) captured")
        
        if metadata:
            response_parts.append(" | ".join(metadata))
        
        # Add the main conversational response (THIS IS THE KEY!)
        if ai_response:
            response_parts.append(f"\n{ai_response}")
        else:
            # Fallback if somehow response is empty
            response_parts.append("\nI'm here and listening! Tell me more. 💭")
        
        # Handle morning news request
        if analysis.get("wants_morning_news"):
            news_time = analysis.get("news_time", "08:00")
            self.db.set_morning_news(user_id, True, news_time)
            response_parts.append(
                f"\n📰 I'll send you morning news every day at {news_time}!"
            )
        
        await update.message.reply_text("\n".join(response_parts))
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice messages"""
        user_id = update.effective_user.id
        
        # Show typing indicator
        await update.message.chat.send_action("typing")
        
        try:
            # Download voice file
            voice = update.message.voice
            voice_file = await voice.get_file()
            voice_bytes = await voice_file.download_as_bytearray()
            
            # Send processing message
            processing_msg = await update.message.reply_text("🎤 Transcribing your voice message...")
            
            # Transcribe with Whisper
            transcribed_text = await self.whisper.transcribe_voice(bytes(voice_bytes))
            
            if not transcribed_text:
                await processing_msg.edit_text("❌ Sorry, I couldn't transcribe that. Please try again or send a text message.")
                return
            
            # Delete processing message
            await processing_msg.delete()
            
            # Show what was transcribed
            await update.message.reply_text(f"🎤 You said:\n\n_{transcribed_text}_", parse_mode='Markdown')
            
            # Process the transcribed text like a normal message
            recent_entries = self.db.get_recent_entries(user_id, days=3)
            context_text = "; ".join([e["content"][:100] for e in recent_entries[:3]]) if recent_entries else None
            
            # Show typing while processing
            await update.message.chat.send_action("typing")
            
            # Analyze with Claude
            analysis = await self.claude.analyze_message(transcribed_text, context_text)
            
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
            response_parts = []
            
            # Add confirmation with context
            if reminder_count > 0 and analysis["ideas"]:
                response_parts.append("✅ Got it! Reminders set and ideas captured.")
            elif reminder_count > 0:
                response_parts.append("✅ Saved! I'll remind you about that.")
            elif analysis["ideas"]:
                response_parts.append("✅ Noted! Interesting ideas here.")
            else:
                response_parts.append("✅ Logged!")
            
            if reminder_count > 0:
                response_parts.append(f"⏰ {reminder_count} reminder(s) scheduled")
            
            if analysis["ideas"]:
                response_parts.append(f"💡 {len(analysis['ideas'])} idea(s) captured")
            
            # ALWAYS provide coaching/advice when possible
            if analysis["needs_response"] and analysis["question"]:
                advice = await self.claude.provide_advice(
                    analysis["question"],
                    analysis["ideas"],
                    context_text or ""
                )
                response_parts.append(f"\n{advice}")
            
            await update.message.reply_text("\n".join(response_parts))
            
        except Exception as e:
            logger.error(f"Voice handling error: {e}")
            await update.message.reply_text("❌ Sorry, something went wrong processing your voice message. Please try again!")
    
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
    
    async def news_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /news command to setup morning news"""
        user_id = update.effective_user.id
        
        # Check if they provided arguments
        if context.args:
            arg = context.args[0].lower()
            
            if arg == "on":
                time = context.args[1] if len(context.args) > 1 else "08:00"
                topics = " ".join(context.args[2:]) if len(context.args) > 2 else "general"
                self.db.set_morning_news(user_id, True, time, topics)
                await update.message.reply_text(
                    f"📰 Morning news enabled!\n\n"
                    f"⏰ Time: {time}\n"
                    f"📋 Topics: {topics}\n\n"
                    f"You'll receive a news briefing every morning.\n"
                    f"Use /news off to disable."
                )
            
            elif arg == "off":
                self.db.set_morning_news(user_id, False)
                await update.message.reply_text("📰 Morning news disabled.")
            
            elif arg == "now":
                # Send news immediately
                topics = " ".join(context.args[1:]) if len(context.args) > 1 else "general"
                await update.message.chat.send_action("typing")
                news = await self.claude.generate_morning_news(topics)
                await update.message.reply_text(news)
            
            else:
                await update.message.reply_text(
                    "📰 **Morning News Commands:**\n\n"
                    "/news on [time] [topics] - Enable morning news\n"
                    "Example: /news on 07:30 technology health\n\n"
                    "/news off - Disable morning news\n"
                    "/news now [topics] - Get news right now\n"
                    "/news - Check current settings",
                    parse_mode='Markdown'
                )
        else:
            # Show current settings
            pref = self.db.get_news_preference(user_id)
            if pref and pref["enabled"]:
                await update.message.reply_text(
                    f"📰 Morning News: **Enabled**\n\n"
                    f"⏰ Time: {pref['time']}\n"
                    f"📋 Topics: {pref['topics']}\n\n"
                    f"Use /news off to disable\n"
                    f"Use /news now to get news immediately",
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text(
                    "📰 Morning News: **Disabled**\n\n"
                    "Want daily news briefings?\n\n"
                    "Use: /news on [time] [topics]\n"
                    "Example: /news on 08:00 technology health\n\n"
                    "Or try: /news now (get news right away)",
                    parse_mode='Markdown'
                )
    
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
    
    async def send_morning_news(self, context: ContextTypes.DEFAULT_TYPE):
        """Background job to send morning news to subscribers"""
        current_hour = datetime.now().hour
        subscribers = self.db.get_news_subscribers(current_hour)
        
        for sub in subscribers:
            try:
                # Generate personalized news
                news = await self.claude.generate_morning_news(sub["topics"])
                
                # Send to user
                await context.bot.send_message(
                    chat_id=sub["user_id"],
                    text=news
                )
                logger.info(f"Sent morning news to user {sub['user_id']}")
            except Exception as e:
                logger.error(f"Failed to send morning news: {e}")
    
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
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    
    if not TELEGRAM_TOKEN or not ANTHROPIC_KEY or not OPENAI_KEY:
        print("ERROR: Please set TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, and OPENAI_API_KEY environment variables")
        print("\nOn Linux/Mac:")
        print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("  export ANTHROPIC_API_KEY='your_key_here'")
        print("  export OPENAI_API_KEY='your_openai_key_here'")
        print("\nOn Windows:")
        print("  set TELEGRAM_BOT_TOKEN=your_token_here")
        print("  set ANTHROPIC_API_KEY=your_key_here")
        print("  set OPENAI_API_KEY=your_openai_key_here")
        return
    
    bot = DiaryBot(TELEGRAM_TOKEN, ANTHROPIC_KEY, OPENAI_KEY)
    bot.run()


if __name__ == "__main__":
    main()
