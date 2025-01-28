import logging
import json
import re
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
MODEL_NAME = "deepseek-r1:8b"  # or whichever model you prefer
CONVERSATIONS_FILE = "conversations.json"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

########################################################################
# CONVERSATION STORAGE ON DISK
########################################################################

conversations = {}

def load_conversations():
    """Load all conversations from disk."""
    global conversations
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except (json.JSONDecodeError, IOError):
            logging.warning("Could not load conversations.json; starting fresh.")
            conversations = {}
    else:
        conversations = {}

def save_conversations():
    """Save current conversations to disk."""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)

########################################################################
# HELPER: EXTRACT <think>...</think> TAGS
########################################################################

def extract_think_tags(text: str):
    """
    Find any <think>...</think> blocks in the text.
    Return a tuple: (main_text, [list_of_think_texts]).
    - main_text has the <think> sections removed.
    - list_of_think_texts has the raw content of each <think>...</think>.
    """
    pattern = r"<think>(.*?)</think>"
    thinking_parts = re.findall(pattern, text, flags=re.DOTALL)
    # Remove them from the main text
    stripped_text = re.sub(pattern, "", text, flags=re.DOTALL)
    # Trim extra whitespace
    stripped_text = stripped_text.strip()
    return stripped_text, thinking_parts

########################################################################
# BOT HANDLERS
########################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears conversation for this chat and greets the user."""
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()

    await update.message.reply_text(
        "Hello! I've cleared our old conversation. Let's start fresh."
    )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch models at runtime with /set_model <model>."""
    global MODEL_NAME
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = args[0]
    await update.message.reply_text(
        f"Model changed to: {MODEL_NAME}. I'll use this going forward."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main entry: stream from Ollama, store conversation, parse out <think> tags, etc."""
    chat_id = str(update.effective_chat.id)
    user_message = update.message.text

    # Load or init conversation for this chat
    conversation = conversations.get(chat_id, [])

    # Append the user's new message
    conversation.append({"role": "user", "content": user_message})

    # Build the prompt from the entire conversation
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    # Create a placeholder message to show streaming progress
    reply_message = await update.message.reply_text("Thinking...")

    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True
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

            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in chunk:
                    generated_text += chunk["response"]

                    # Rate-limit edits to avoid spamming Telegram
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

    except requests.exceptions.RequestException as e:
        await reply_message.edit_text(f"Error connecting to Ollama:\n{e}")
        return

    # -------------------------------------------------------------------
    # NOW: Extract <think>...</think> tags and remove them from final text
    # -------------------------------------------------------------------
    stripped_text, thinking_parts = extract_think_tags(generated_text)

    # Update the conversation with the stripped text (no <think>)
    # so next time the model sees only the user-facing text.
    conversation.append({"role": "assistant", "content": stripped_text})
    conversations[chat_id] = conversation
    save_conversations()

    # If there's any <think> content, send it in separate spoiler messages
    # each <think> is posted as a single message
    for part in thinking_parts:
        # Telegram “spoiler + italic + block quote” syntax in Markdown:
        # > || *some text* ||
        # In Markdown v2, we might need escapes; let's try basic Markdown first.

        spoiler_text = f"> || *{part.strip()}* ||"

        try:
            # We send a new message for each <think> snippet
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=spoiler_text,
                parse_mode="Markdown"
            )
        except:
            pass

    # Finally, if we changed the text displayed (removed <think> parts),
    # we can optionally do one last edit of the main reply to show the stripped text:
    if stripped_text != generated_text:
        try:
            await reply_message.edit_text(stripped_text)
        except:
            pass

########################################################################
# MAIN
########################################################################

if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
