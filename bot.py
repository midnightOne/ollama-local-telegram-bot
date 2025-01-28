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
# Ollama endpoint (adjust if using /api/generate or different port)
OLLAMA_URL = "http://127.0.0.1:11411/generate"
MODEL_NAME = "deepseek-r1:8b"

CONVERSATIONS_FILE = "conversations.json"

# Toggle to show/hide chain-of-thought
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
    """Load conversation data from JSON on disk into the `conversations` dict."""
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
    """Save all conversation data to JSON on disk."""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)

##############################################################################
# HELPER: Extract <think> text vs. public text
##############################################################################

def extract_think_sections(text):
    """
    Given all streamed text so far, find fully closed <think>...</think> sections.
    Return:
      thinking_text (concatenation of everything inside <think>...</think>)
      public_text   (same text but with <think>...</think> removed)

    Any partial <think> block (not yet closed) remains in public_text
    until the closing </think> arrives in later chunks.
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    # Combine them into one big chain-of-thought string
    thinking_text = "\n\n".join(m.strip() for m in matches)

    # Remove those blocks from the public text
    public_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()

    return thinking_text, public_text

##############################################################################
# BOT COMMANDS
##############################################################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conversation for this chat and greet the user."""
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()
    await update.message.reply_text("Conversation reset. Let's start fresh!")

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Let user switch models via /set_model <model_name>."""
    global MODEL_NAME
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = args[0]
    await update.message.reply_text(f"Model set to: {MODEL_NAME}")

async def toggle_thinking_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle whether we show chain-of-thought in Telegram or not."""
    global SHOW_THINKING
    SHOW_THINKING = not SHOW_THINKING
    status = "ON" if SHOW_THINKING else "OFF"
    await update.message.reply_text(f"SHOW_THINKING is now {status}.")

##############################################################################
# MAIN HANDLER: One message at a time from the user
##############################################################################

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Build prompt from conversation so far.
    2. Stream partial tokens from Ollama.
    3. If SHOW_THINKING=True, produce exactly 3 distinct messages:
       (a) "Thinking:" (static, no updates)
       (b) chain-of-thought in spoilers, updated in real-time
       (c) user-facing text (no <think>), updated in real-time
    4. If SHOW_THINKING=False, produce only 1 streaming message with user-facing text.
    5. Store only user-facing text in conversation history.
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    # Retrieve or init conversation
    conversation = conversations.get(chat_id, [])
    # Append user's new message
    conversation.append({"role": "user", "content": user_text})

    # Build the prompt
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    # We'll accumulate partial tokens in here
    raw_accumulated = ""

    if SHOW_THINKING:
        # Create exactly 3 messages
        # 1) "Thinking:" (static)
        thinking_header_msg = await update.message.reply_text("Thinking:")

        # 2) chain-of-thought in spoilers, updated in real time
        chain_of_thought_msg = await update.effective_chat.send_message(
            text="(chain-of-thought spoilers here)",
            parse_mode="Markdown"
        )

        # 3) public text (final answer)
        public_msg = await update.effective_chat.send_message(
            text="(final answer will appear here)"
        )
    else:
        # If thinking is OFF, only 1 streaming message
        single_msg = await update.message.reply_text("(Generating answer...)")

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
                    # Update the chain-of-thought or final message with error
                    await chain_of_thought_msg.edit_text(error_text)
                else:
                    await single_msg.edit_text(error_text)
                return

            # Stream partial tokens
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in chunk:
                    raw_accumulated += chunk["response"]

                    # Extract chain-of-thought vs. public text from everything so far
                    thinking_text, public_text = extract_think_sections(raw_accumulated)

                    # Rate-limit updates so we don't spam Telegram
                    if time.time() - last_update_time > 0.7:
                        if SHOW_THINKING:
                            # Update chain-of-thought in spoilers
                            spoiler_text = f"||{thinking_text}||" if thinking_text else "|| (nothing yet) ||"
                            try:
                                await chain_of_thought_msg.edit_text(
                                    spoiler_text,
                                    parse_mode="Markdown"
                                )
                            except:
                                pass

                            # Update public text
                            try:
                                await public_msg.edit_text(public_text if public_text else "(no public text yet)")
                            except:
                                pass
                        else:
                            # If thinking is off, we only have the single message
                            try:
                                await single_msg.edit_text(public_text if public_text else "(still thinking...)")
                            except:
                                pass

                        last_update_time = time.time()

            # Final update after streaming ends
            thinking_text, public_text = extract_think_sections(raw_accumulated)

            if SHOW_THINKING:
                spoiler_text = f"||{thinking_text}||" if thinking_text else "|| (nothing) ||"
                # Final chain-of-thought
                try:
                    await chain_of_thought_msg.edit_text(spoiler_text, parse_mode="Markdown")
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

    # Save the final user-facing text (no chain-of-thought) into conversation
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
