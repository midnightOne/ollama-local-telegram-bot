import logging
import json
import os
import re
import requests
import time

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# --------------------------- CONFIG ---------------------------
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Use /api/generate on port 11434 (as requested)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL_NAME = "deepseek-r1:8b"
CONVERSATIONS_FILE = "conversations.json"

# Toggle chain-of-thought
SHOW_THINKING = True
# --------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

##############################################################################
# DISK STORAGE (Conversation Histories)
##############################################################################

conversations = {}

def load_conversations():
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
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters:
    '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
    so we can safely use spoilers and other formatting.
    """
    # Characters that must be escaped in MarkdownV2
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    for c in escape_chars:
        text = text.replace(c, '\\' + c)
    return text

def extract_think_sections(text):
    """
    Parse out <think>...</think> blocks from accumulated text.
    Returns (thinking_text, public_text):
      - thinking_text: concatenation of fully closed <think> blocks
      - public_text: same text but with those blocks removed
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    thinking_text = "\n\n".join(m.strip() for m in matches)
    public_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return thinking_text, public_text

##############################################################################
# BOT COMMANDS
##############################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()
    await update.message.reply_text("Conversation reset. Let's start fresh!")

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODEL_NAME
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = args[0]
    await update.message.reply_text(f"Model set to: {MODEL_NAME}")

async def toggle_thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global SHOW_THINKING
    SHOW_THINKING = not SHOW_THINKING
    status = "ON" if SHOW_THINKING else "OFF"
    await update.message.reply_text(f"SHOW_THINKING is now {status}.")

##############################################################################
# MAIN MESSAGE HANDLER
##############################################################################

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    If SHOW_THINKING=True -> exactly 3 messages:
      1) "Thinking:"
      2) chain-of-thought in spoilers, updated in real-time
      3) user-facing text, updated in real-time
    If SHOW_THINKING=False -> one streaming message with user-facing text
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    # Retrieve or init conversation
    conversation = conversations.get(chat_id, [])
    conversation.append({"role": "user", "content": user_text})

    # Build prompt from conversation so far
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    raw_accumulated = ""

    if SHOW_THINKING:
        # Message #1: "Thinking:" (static)
        await update.message.reply_text("Thinking:")

        # Message #2: chain-of-thought spoilers
        chain_of_thought_msg = await update.effective_chat.send_message(
            text="(chain-of-thought spoilers here)",
            parse_mode="MarkdownV2"
        )

        # Message #3: public text
        public_msg = await update.effective_chat.send_message(
            text="(final answer will appear here)"
        )
    else:
        # Only one streaming message
        single_msg = await update.message.reply_text("(Generating...)")

    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True
    }

    last_update_time = time.time()

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                error_text = f"Error {response.status_code} from Ollama:\n{response.text}"
                if SHOW_THINKING:
                    await chain_of_thought_msg.edit_text(error_text)
                else:
                    await single_msg.edit_text(error_text)
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in chunk:
                    raw_accumulated += chunk["response"]

                    # Separate chain-of-thought vs. public
                    thinking_text, public_text = extract_think_sections(raw_accumulated)

                    # Limit how often we edit messages to avoid spam/rate limits
                    if time.time() - last_update_time > 0.7:
                        if SHOW_THINKING:
                            # Format chain-of-thought in MarkdownV2 spoilers
                            # e.g. ||some chain-of-thought|| + escaped
                            safe_thinking = escape_markdown_v2(thinking_text) if thinking_text else ""
                            spoiler_text = f"||{safe_thinking}||" if safe_thinking else "|| (nothing yet) ||"

                            try:
                                await chain_of_thought_msg.edit_text(spoiler_text, parse_mode="MarkdownV2")
                            except:
                                pass

                            # Update public text (no special parse mode needed if not using Markdown)
                            try:
                                await public_msg.edit_text(public_text if public_text else "(no public text yet)")
                            except:
                                pass
                        else:
                            # If thinking is off, just show user-facing text
                            try:
                                await single_msg.edit_text(public_text if public_text else "(still thinking...)")
                            except:
                                pass

                        last_update_time = time.time()

            # Final update after streaming ends
            thinking_text, public_text = extract_think_sections(raw_accumulated)

            if SHOW_THINKING:
                # Final chain-of-thought update
                safe_thinking = escape_markdown_v2(thinking_text) if thinking_text else ""
                spoiler_text = f"||{safe_thinking}||" if safe_thinking else "|| (nothing) ||"
                try:
                    await chain_of_thought_msg.edit_text(spoiler_text, parse_mode="MarkdownV2")
                except:
                    pass

                # Final public text
                try:
                    await public_msg.edit_text(public_text if public_text else "(no text)")
                except:
                    pass
            else:
                # Single final text
                try:
                    await single_msg.edit_text(public_text if public_text else "(no text)")
                except:
                    pass

    except requests.exceptions.RequestException as e:
        err_msg = f"Error connecting to Ollama:\n{e}"
        if SHOW_THINKING:
            await update.effective_chat.send_message(err_msg)
        else:
            await single_msg.edit_text(err_msg)
        return

    # Save user-facing text in conversation
    conversation.append({"role": "assistant", "content": public_text})
    conversations[chat_id] = conversation
    save_conversations()

##############################################################################
# MAIN
##############################################################################

if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(CommandHandler("toggle_thinking", toggle_thinking_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
