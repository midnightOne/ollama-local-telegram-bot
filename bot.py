import logging
import json
import requests
import time
import os

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
OLLAMA_URL = "http://localhost:11411/generate"
MODEL_NAME = "deepseek-r1:8b"  # Adjust to your desired default model
CONVERSATIONS_FILE = "conversations.json"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

########################################################################
# GLOBAL STORE: Each chat_id -> list of messages [{"role": "...", "content": "..."}]
########################################################################
conversations = {}

def load_conversations():
    """Load conversations from disk into the global dictionary."""
    global conversations
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except (json.JSONDecodeError, IOError):
            logging.warning("Could not load existing conversations.json; starting fresh.")
            conversations = {}
    else:
        conversations = {}

def save_conversations():
    """Save all conversations from memory to disk as JSON."""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)

########################################################################
# BOT COMMANDS AND HANDLERS
########################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears the current chat's conversation and greets the user."""
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []  # Reset conversation for this chat
    save_conversations()

    await update.message.reply_text(
        "Hello! I've cleared our old conversation. Let's start fresh."
    )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Let the user change the active model with /set_model <model_name>."""
    global MODEL_NAME
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return

    MODEL_NAME = args[0]
    await update.message.reply_text(
        f"Model changed to: {MODEL_NAME}. I'll use this model going forward."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main message handler:
    - Retrieve conversation from disk/memory
    - Append user message
    - Build prompt with entire conversation
    - Stream from Ollama, editing one Telegram message for partial tokens
    - Append assistant message
    - Save to disk
    """
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)

    # 1. Load conversation for this chat (list of {"role": ..., "content": ...})
    conversation = conversations.get(chat_id, [])

    # 2. Append user message
    conversation.append({"role": "user", "content": user_message})

    # 3. Build a prompt that includes the entire conversation so far
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:  # "assistant"
            prompt_text += f"Assistant: {turn['content']}\n"
    # Now add the final "Assistant:" to prompt the model to continue
    prompt_text += "Assistant:"

    # Create a placeholder message to update as tokens stream in
    reply_message = await update.message.reply_text("Thinking...")

    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True  # Enable streaming
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                await reply_message.edit_text(
                    f"Error {response.status_code} from Ollama:\n{response.text}"
                )
                return

            generated_text = ""
            last_edit_time = time.time()

            # 4. Stream partial tokens
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in chunk:
                    generated_text += chunk["response"]

                    # Rate-limit message edits to avoid Telegram "FloodWait"
                    if time.time() - last_edit_time > 0.6:
                        try:
                            await reply_message.edit_text(generated_text)
                            last_edit_time = time.time()
                        except:
                            pass

            # Final edit with the complete text
            try:
                await reply_message.edit_text(generated_text)
            except:
                pass

            # 5. Append the assistant's full response to the conversation
            conversation.append({"role": "assistant", "content": generated_text})
            conversations[chat_id] = conversation

            # 6. Save updated conversation to disk
            save_conversations()

    except requests.exceptions.RequestException as e:
        await reply_message.edit_text(f"Error connecting to Ollama:\n{e}")

########################################################################
# MAIN: Load from disk, start the bot
########################################################################

if __name__ == "__main__":
    # Load previous conversations from disk before starting
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
