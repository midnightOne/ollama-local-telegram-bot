import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your bot token
TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
OLLAMA_URL = 'http://localhost:11411/generate'  # The default Docker port for Ollama

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hello! Send me a message, and I’ll ask Ollama!')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages from the user."""
    user_message = update.message.text

    # Call Ollama locally
    # The prompt structure can be adjusted; below is a simple example
    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        json={"prompt": user_message},
        timeout=300  # Some models can take a while, so increase timeout if needed
    )

    if response.status_code == 200:
        data = response.json()
        # 'data' is typically a JSON object or array; if the text is in 'data["done"]' or 'data["generated_text"]'
        # the structure depends on which version of Ollama you have. Check the output to parse correctly.
        # Below is a simplified approach if the response is just a JSON array of generation steps:
        generated_text = ""
        for chunk in data:
            # Each chunk typically has 'done': bool, 'response': partial text
            if 'response' in chunk:
                generated_text += chunk['response']

        # Send the final response to Telegram
        await update.message.reply_text(generated_text.strip())
    else:
        await update.message.reply_text('Sorry, there was an error connecting to the local LLM.')

if __name__ == '__main__':
    # Initialize Telegram application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot (this uses long-polling)
    application.run_polling()
