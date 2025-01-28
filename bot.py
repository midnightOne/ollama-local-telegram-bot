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

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
# Adjust this if your Ollama server is on a different port or endpoint
OLLAMA_URL = "http://localhost:11411/generate"
MODEL_NAME = "deepseek-r1:8b"
CONVERSATIONS_FILE = "conversations.json"

# Toggle this to show/hide the chain-of-thought in Telegram
SHOW_THINKING = True
# --------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

##############################################################################
# DISK STORAGE FOR CONVERSATIONS
##############################################################################

conversations = {}

def load_conversations():
    """Load conversation data from disk into memory."""
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

##############################################################################
# HELPER: EXTRACT THINKING TEXT
##############################################################################

def extract_thinking_tags(text: str):
    """
    Extracts all <think>...</think> sections, concatenates them
    into one 'thinking' string, and returns (thinking_text, final_text)
    where final_text has all <think> sections removed.
    """
    # Find all matches (DOTALL so it can span multiple lines)
    pattern = r"<think>(.*?)</think>"
    thinking_matches = re.findall(pattern, text, flags=re.DOTALL)

    # Combine them (with some spacing if you like)
    thinking_text = "\n\n".join(m.strip() for m in thinking_matches)

    # Remove them from the main text
    final_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return thinking_text, final_text

##############################################################################
# BOT COMMANDS
##############################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation for this chat and greet the user."""
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()
    await update.message.reply_text(
        "Hello! I've cleared our conversation. Let's start fresh."
    )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Let user switch to a different local model using /set_model <model_name>."""
    global MODEL_NAME
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = args[0]
    await update.message.reply_text(
        f"Model changed to: {MODEL_NAME}."
    )

async def toggle_thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Command to toggle the global SHOW_THINKING flag at runtime.
    Example usage: /toggle_thinking
    """
    global SHOW_THINKING
    SHOW_THINKING = not SHOW_THINKING
    status = "ON" if SHOW_THINKING else "OFF"
    await update.message.reply_text(f"SHOW_THINKING is now {status}.")

##############################################################################
# MAIN MESSAGE HANDLER
##############################################################################

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages, build prompt with conversation, stream from Ollama."""
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)

    # Retrieve or init conversation
    conversation = conversations.get(chat_id, [])

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_message})

    # Build prompt
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:  # assistant
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    # We'll stream the final text from Ollama
    # We'll accumulate the entire text in generated_text
    reply_message = await update.message.reply_text("Thinking...")
    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True
    }

    generated_text = ""
    last_edit_time = time.time()

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                await reply_message.edit_text(
                    f"Error {response.status_code} from Ollama:\n{response.text}"
                )
                return

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

            # Final update
            try:
                await reply_message.edit_text(generated_text)
            except:
                pass

    except requests.exceptions.RequestException as e:
        await reply_message.edit_text(f"Error connecting to Ollama:\n{e}")
        return

    # ------------------------------------------------------------------------
    # Separate "thinking" from "final" text
    # ------------------------------------------------------------------------
    thinking_text, final_text = extract_thinking_tags(generated_text)

    # If SHOW_THINKING is True, send the chain-of-thought message
    if SHOW_THINKING and thinking_text.strip():
        # Mark it clearly: e.g., "Thinking" in code block, or spoiler, etc.
        # We'll just do a plain message here, but you can do f"|| {thinking_text} ||"
        # or "```thinking_text```"
        await update.effective_chat.send_message(
            text=f"*Thinking:*\n\n{thinking_text}",
            parse_mode="Markdown"
        )

    # Now send the final user-facing text as a separate message
    if final_text.strip():
        await update.effective_chat.send_message(
            text=final_text
        )
    else:
        # If there's no final text, at least send something
        await update.effective_chat.send_message(
            text="(No final text returned)"
        )

    # ------------------------------------------------------------------------
    # Update conversation with final text only, not the chain-of-thought
    # ------------------------------------------------------------------------
    conversation.append({"role": "assistant", "content": final_text})
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
