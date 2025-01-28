import logging
import json
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

# Replace with your own Telegram Bot token
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Ollama server endpoint
# Adjust host/port if needed
OLLAMA_URL = "http://localhost:11411/generate"

# Default model name from `ollama list`
MODEL_NAME = "deepseek-r1:8b"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Respond to /start command."""
    await update.message.reply_text(
        "Hello! I will stream responses token-by-token from Ollama.\n"
        f"Currently using model: {MODEL_NAME}"
    )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command: /set_model <model_name> to switch local models on the fly."""
    global MODEL_NAME
    args = context.args

    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return

    MODEL_NAME = args[0]
    await update.message.reply_text(
        f"Model changed to: {MODEL_NAME}\nNext requests will use this model."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all non-command text messages by streaming tokens from Ollama."""
    user_message = update.message.text

    # Create an initial message to store our "partial" text
    reply_message = await update.message.reply_text(
        "Thinking... (streaming reply)"
    )

    payload = {
        "prompt": user_message,
        "model": MODEL_NAME,
        "stream": True  # Enable streaming from Ollama
    }

    try:
        # We'll make a streaming POST request
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                await reply_message.edit_text(
                    f"Error {response.status_code} from Ollama:\n{response.text}"
                )
                return

            generated_text = ""
            last_edit_time = time.time()

            for line in response.iter_lines(decode_unicode=True):
                # Skip empty lines/heartbeats
                if not line.strip():
                    continue

                # Each line is a separate JSON chunk, e.g. {"response": "...", "done": false}
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # If for some reason we can't parse a line, skip it
                    continue

                # If there's partial text in 'response', append it
                if "response" in chunk:
                    generated_text += chunk["response"]

                    # ----- Optional Rate-Limit Edits -----
                    # Check how long it's been since last edit
                    # so we don't spam Telegram too frequently.
                    if time.time() - last_edit_time > 0.7:  # every ~0.7s
                        try:
                            await reply_message.edit_text(generated_text)
                            last_edit_time = time.time()
                        except:
                            # If we hit an error from Telegram (e.g. rate limit),
                            # we can either break or ignore. We'll ignore for now.
                            pass

            # After the stream finishes, do a final edit with the complete text
            # (in case we have leftover tokens that weren't edited due to timing).
            try:
                await reply_message.edit_text(generated_text)
            except:
                pass

    except requests.exceptions.RequestException as e:
        # Network/connection error
        await reply_message.edit_text(f"Error connecting to Ollama:\n{e}")


if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
