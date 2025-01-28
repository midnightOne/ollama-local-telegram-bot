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

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"   # per your request
MODEL_NAME = "deepseek-r1:8b"
CONVERSATIONS_FILE = "conversations.json"

SHOW_THINKING = True  # Toggle chain-of-thought display

# Telegram imposes ~4096 char limit for messages in most cases.
MAX_TELEGRAM_MESSAGE_LENGTH = 4096

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# -------------------------------------------------------------------
# DISK STORAGE FOR CONVERSATIONS
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# MARKDOWN V2 ESCAPING AND CHUNKING
# -------------------------------------------------------------------
def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters:
    '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    for c in escape_chars:
        text = text.replace(c, f"\\{c}")
    return text

def chunk_text(text: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH):
    """
    Splits a (possibly very long) string into a list of chunks
    each up to 'max_length' characters.
    """
    lines = []
    start = 0
    while start < len(text):
        end = start + max_length
        lines.append(text[start:end])
        start = end
    return lines


# -------------------------------------------------------------------
# HELPER: EXTRACT <think> TAGS
# -------------------------------------------------------------------
def extract_think_sections(text: str):
    """
    Finds <think>...</think> blocks in 'text' and removes them.
    Returns (thinking_text, public_text):
      - thinking_text: all fully matched <think> blocks combined
      - public_text: 'text' with those <think> blocks removed
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    thinking_text = "\n\n".join(m.strip() for m in matches)
    public_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return thinking_text, public_text


# -------------------------------------------------------------------
# BOT COMMANDS
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# MAIN MESSAGE HANDLER
# -------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    If SHOW_THINKING=True -> 3 messages total:
      1) "Thinking:" (static)
      2) chain-of-thought in spoilers (MarkdownV2), updated partial
      3) user-facing text, updated partial
      Then chunk final texts if > 4096 chars.

    If SHOW_THINKING=False -> one streamed message with user-facing text,
      chunk final if > 4096.
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    # Retrieve or init conversation
    conversation = conversations.get(chat_id, [])
    conversation.append({"role": "user", "content": user_text})

    # Build prompt from entire conversation
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    # We'll accumulate partial tokens in this string
    raw_accumulated = ""

    # Prepare messages depending on SHOW_THINKING
    if SHOW_THINKING:
        # 1) "Thinking:"
        await update.message.reply_text("Thinking:")

        # 2) chain-of-thought in spoilers
        chain_of_thought_msg = await update.effective_chat.send_message(
            text="(chain-of-thought spoilers here)",
            parse_mode="MarkdownV2"
        )

        # 3) public text
        public_msg = await update.effective_chat.send_message(
            text="(final answer will appear here)"
        )
    else:
        # Single message for user-facing text
        single_msg = await update.message.reply_text("(Generating...)")

    # We'll stream from Ollama
    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True
    }
    last_update_time = time.time()

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                err_text = f"Error {response.status_code} from Ollama:\n{response.text}"
                if SHOW_THINKING:
                    await chain_of_thought_msg.edit_text(err_text)
                else:
                    await single_msg.edit_text(err_text)
                return

            # Partial token streaming
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in chunk:
                    raw_accumulated += chunk["response"]

                    # Extract chain-of-thought vs public
                    thinking_text, public_text = extract_think_sections(raw_accumulated)

                    # Throttle updates to avoid spamming
                    if time.time() - last_update_time > 0.7:
                        if SHOW_THINKING:
                            # Update chain-of-thought in spoilers
                            spoiler_raw = escape_markdown_v2(thinking_text) or "(no thinking yet)"
                            spoiler_text = f"||{spoiler_raw}||"
                            try:
                                await chain_of_thought_msg.edit_text(spoiler_text, parse_mode="MarkdownV2")
                            except:
                                pass

                            # Update public text
                            try:
                                await public_msg.edit_text(public_text if public_text else "(no public text yet)")
                            except:
                                pass
                        else:
                            # Show only public text
                            try:
                                await single_msg.edit_text(public_text if public_text else "(still thinking...)")
                            except:
                                pass

                        last_update_time = time.time()

            # Final parse after streaming completes
            thinking_text, public_text = extract_think_sections(raw_accumulated)

    except requests.exceptions.RequestException as e:
        err_msg = f"Error connecting to Ollama:\n{e}"
        if SHOW_THINKING:
            await update.effective_chat.send_message(err_msg)
        else:
            await single_msg.edit_text(err_msg)
        return

    # ----------------------------------------------------------------
    # CHUNK & SEND (OR EDIT) FINAL TEXTS IF OVER TELEGRAM LIMIT
    # ----------------------------------------------------------------
    if SHOW_THINKING:
        # Final chain-of-thought
        spoiler_raw = escape_markdown_v2(thinking_text) or "(nothing)"
        spoiler_text = f"||{spoiler_raw}||"

        # Let's try to "edit" the final chain_of_thought_msg if under limit
        # If it's over, we'll send multiple messages
        chunks = chunk_text(spoiler_text)
        if len(chunks) == 1:
            # Fits in one message
            try:
                await chain_of_thought_msg.edit_text(chunks[0], parse_mode="MarkdownV2")
            except:
                pass
        else:
            # It's too big - let's edit the first message with a note, then send the rest
            try:
                await chain_of_thought_msg.edit_text("(chain-of-thought too large, splitting...)")
            except:
                pass
            for c in chunks:
                await update.effective_chat.send_message(c, parse_mode="MarkdownV2")

        # Final public text
        public_chunks = chunk_text(public_text)
        if len(public_chunks) == 1:
            try:
                await public_msg.edit_text(public_chunks[0] if public_chunks[0] else "(no text)")
            except:
                pass
        else:
            # Edit the original to show first chunk or note
            try:
                await public_msg.edit_text("(answer too large, splitting...)")
            except:
                pass
            for c in public_chunks:
                await update.effective_chat.send_message(c if c else "(empty)")
    else:
        # Single final text
        public_chunks = chunk_text(public_text)
        if len(public_chunks) == 1:
            try:
                await single_msg.edit_text(public_chunks[0] if public_chunks[0] else "(no text)")
            except:
                pass
        else:
            # It's bigger than 4096, so we must split into multiple messages
            try:
                await single_msg.edit_text("(answer too large, splitting...)")
            except:
                pass
            for c in public_chunks:
                await update.effective_chat.send_message(c if c else "(empty)")

    # Save final user-facing text to conversation
    conversation.append({"role": "assistant", "content": public_text})
    conversations[chat_id] = conversation
    save_conversations()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(CommandHandler("toggle_thinking", toggle_thinking_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
