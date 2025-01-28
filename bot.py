import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Replace with your actual Telegram bot token
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
# Ollama API URL (assuming Docker exposes 11411)
OLLAMA_URL = 'http://localhost:11411/generate'

# Default model to use (from your `ollama list`)
MODEL_NAME = "deepseek-r1:8b"   # 4.9 GB model

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Respond to /start command."""
    await update.message.reply_text(
        "Hello! Send me a message, and I'll query the local Ollama model for you.\n"
        f"Currently using model: {MODEL_NAME}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages from the user."""
    user_message = update.message.text

    # Call Ollama locally, specifying the model
    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json={
                "prompt": user_message,
                # Here is the key field to specify your local model
                "model": MODEL_NAME  
            },
            timeout=300  # Increase if needed for bigger models
        )
    except requests.exceptions.RequestException as e:
        await update.message.reply_text(f"Error connecting to Ollama:\n{e}")
        return

    if response.status_code == 200:
        data = response.json()
        generated_text = ""
        # Ollama typically returns a list of chunks in the response
        for chunk in data:
            # Each chunk often includes 'response' or partial text
            if 'response' in chunk:
                generated_text += chunk['response']
        # Send the final aggregated text to Telegram
        await update.message.reply_text(generated_text.strip())
    else:
        await update.message.reply_text(
            f"Error {response.status_code} from Ollama: {response.text}"
        )

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Example optional command: /set_model <model_name>
    Lets you switch to any model you have installed locally.
    """
    global MODEL_NAME
    args = context.args

    if not args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return

    new_model = args[0]
    MODEL_NAME = new_model
    await update.message.reply_text(
        f"Model changed to: {MODEL_NAME}\nNext requests will use this model."
    )

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start long-polling
    application.run_polling()
