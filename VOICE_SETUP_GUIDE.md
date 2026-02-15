# 🎤 Voice Message Support - Setup Guide

## ✨ What's New?

Your bot now supports voice messages! Just hold the mic button in Telegram and speak - the bot will:
1. Transcribe your voice to text using OpenAI Whisper
2. Show you what it heard
3. Process it like a normal diary entry
4. Extract reminders and ideas from your voice

## 📝 Setup Steps

### Step 1: Get OpenAI API Key

1. Go to: https://platform.openai.com/signup
2. Sign up or log in
3. Click your profile → "View API Keys"
4. Click "Create new secret key"
5. Name it: "Telegram Diary Bot"
6. **SAVE THE KEY** - it starts with `sk-proj-` or `sk-`

### Step 2: Add API Key to Railway

1. Go to Railway dashboard
2. Click on your project: `shanti-telegram-bot`
3. Click **"Variables"** tab
4. Click **"+ New Variable"**
5. Variable name: `OPENAI_API_KEY`
6. Value: paste your OpenAI key
7. Click "Add"

### Step 3: Update Files on GitHub

You need to update 2 files:

#### A. Update requirements.txt

1. Go to your GitHub repo
2. Click `requirements.txt`
3. Click pencil icon (Edit)
4. Replace content with:
   ```
   python-telegram-bot[job-queue]==21.0
   anthropic>=0.40.0
   openai>=1.0.0
   ```
5. Commit changes

#### B. Update telegram_diary_bot.py

1. Delete the old `telegram_diary_bot.py` from GitHub
2. Upload the new one I provided (download it from above)
3. Commit changes

### Step 4: Railway Will Auto-Redeploy

Wait 2-3 minutes. Railway will:
- Detect the changes
- Install OpenAI library
- Restart your bot with voice support

---

## 💰 Cost Breakdown

### Whisper API Pricing:
- **$0.006 per minute** of audio

### Example Monthly Costs:

**Light Use (5 voice messages/day, ~30 seconds each):**
- Audio per month: ~1.25 hours
- Whisper cost: ~$0.45/month
- Claude cost: ~$0.06/month
- **Total: ~$0.51/month**

**Medium Use (10 voice messages/day, ~45 seconds each):**
- Audio per month: ~3.75 hours
- Whisper cost: ~$1.35/month
- Claude cost: ~$0.18/month
- **Total: ~$1.53/month**

**Heavy Use (20 voice messages/day, ~1 minute each):**
- Audio per month: ~10 hours
- Whisper cost: ~$3.60/month
- Claude cost: ~$0.30/month
- **Total: ~$3.90/month**

Still super affordable!

---

## 🎤 How to Use

### Send Voice Messages:

1. Open your bot in Telegram
2. **Hold the microphone button** (don't click, hold!)
3. Speak your message:
   - "Today I finished the presentation for work"
   - "I need to call the dentist tomorrow at 3pm"
   - "I have an idea for a new app about meditation"
4. Release the button
5. Bot will:
   - Show "🎤 Transcribing your voice message..."
   - Display what you said
   - Process it normally
   - Save to your diary
   - Extract any reminders

### Example Voice Messages:

**Diary Entry:**
🎤 "Had a really productive day. Finished the design mockups and had a great meeting with the client. They loved the new direction!"

**Setting Reminders:**
🎤 "Reminder to register for that yoga class on Tuesday at 6pm. Also need to call mom this weekend."

**Creative Ideas:**
🎤 "I'm thinking about starting a podcast about sustainable living. Should I focus on zero waste tips or product reviews? What do you think?"

---

## ✅ Verify It's Working

### Test Voice:
1. Send a voice message: "This is a test"
2. Bot should reply:
   ```
   🎤 You said:
   
   This is a test
   
   ✅ Noted!
   ```

### If It Doesn't Work:

**Check Railway logs:**
1. Railway dashboard → Your project
2. Click "Deployments"
3. Click latest deployment
4. Look for errors

**Common issues:**
- ❌ OPENAI_API_KEY not set → Add it in Variables
- ❌ OpenAI key invalid → Check you copied it correctly
- ❌ No credit on OpenAI account → Add $5 to your account

---

## 💳 OpenAI Account Setup

### Add Credit to OpenAI:

1. Go to: https://platform.openai.com/settings/organization/billing
2. Click "Add payment method"
3. Add credit card
4. Set spending limit (e.g., $5/month)
5. Done!

OpenAI requires a small prepaid balance (~$5) to use the API.

---

## 🌟 Features

✅ **Multi-language support** - Whisper supports 50+ languages
✅ **High accuracy** - Better than most speech-to-text services
✅ **Shows transcription** - You can verify what it heard
✅ **Seamless integration** - Works exactly like text messages
✅ **Cost-effective** - Cheaper than most alternatives

---

## 🔧 Troubleshooting

### "Couldn't transcribe that"
- Voice too quiet
- Background noise too loud
- Very short message (< 1 second)
- Solution: Speak clearly, reduce background noise

### "OpenAI API error"
- Check your API key is correct
- Verify you have credit on OpenAI account
- Check OpenAI status: https://status.openai.com

### Bot doesn't respond to voice
- Check Railway logs for errors
- Verify OPENAI_API_KEY is set
- Make sure new code is deployed

---

## 📊 Monitoring Costs

### Check OpenAI Usage:
1. Go to: https://platform.openai.com/usage
2. See how many API calls you're making
3. See total cost

### Set Budget Alerts:
1. Go to: https://platform.openai.com/settings/organization/billing
2. Click "Usage limits"
3. Set monthly budget (e.g., $5)
4. Get email when you hit 80%, 100%

---

## 🎉 You're All Set!

Your bot now supports:
- ✍️ Text messages
- 🎤 Voice messages
- ⏰ Smart reminders
- 💡 Creative advice
- 📊 Weekly summaries

Enjoy your voice-enabled diary assistant! 🚀
