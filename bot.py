import logging
import json
import os
import requests
import time
import re

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# ----------------- CONFIG & GLOBALS -----------------
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Ollama endpoint
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL_NAME = "deepseek-r1:8b"
CONVERSATIONS_FILE = "conversations.json"
MAX_CONVERSATION_TURNS = 10

# Telegram’s nominal max message length
TELEGRAM_MAX_LEN = 4096

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# We’ll store conversation history in JSON on disk
conversations = {}

def debug_conversation_structure(chat_id):
    """Helper function to debug conversation structure"""
    if chat_id not in conversations:
        logging.info(f"No conversation found for chat_id {chat_id}")
        return
        
    convo = conversations[chat_id]
    logging.info(f"Conversation length: {len(convo)} turns")
    
    for i, turn in enumerate(convo):
        role = turn.get("role", "MISSING_ROLE")
        if role == "user":
            logging.info(f"Turn {i}: User message")
        elif role == "assistant":
            has_thinking = bool(turn.get("thinking"))
            has_content = bool(turn.get("content"))
            logging.info(f"Turn {i}: Assistant response (thinking: {has_thinking}, content: {has_content})")
        else:
            logging.info(f"Turn {i}: Invalid role: {role}")

def build_prompt(conversation):
    """Build the prompt with clearer instructions about thinking tags"""
    prompt_text = [
        "You are a helpful assistant. Respond naturally to the user's messages.",
        "Previous conversation:"
    ]
    
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text.append(f"User: {turn['content']}")
        else:
            # For assistant turns, reconstruct with thinking tags if present
            thinking = turn.get("thinking", "").strip()
            response = turn.get("content", "").strip()
            
            if thinking and response:
                prompt_text.append(f"Assistant: <think>{thinking}</think>\n{response}")
            elif thinking:
                prompt_text.append(f"Assistant: <think>{thinking}</think>")
            else:
                prompt_text.append(f"Assistant: {response}")
    
    prompt_text.append("Assistant:")
    return "\n".join(prompt_text)

def trim_conversation(conversation, max_turns=MAX_CONVERSATION_TURNS):
    """
    Keep only the last N turns while ensuring we don't break in the middle of a turn.
    A turn consists of a user message and the assistant's response.
    """
    if len(conversation) <= max_turns * 2:
        return conversation
        
    # Ensure we start with a user message when trimming
    start_idx = len(conversation) - (max_turns * 2)
    while start_idx > 0 and conversation[start_idx]["role"] != "user":
        start_idx -= 1
        
    return conversation[start_idx:]

def load_conversations():
    """Load conversations from disk with validation"""
    global conversations
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Validate conversation structure with enhanced checks
                for chat_id, convo in loaded.items():
                    valid_convo = []
                    expected_role = "user"
                    for turn in convo:
                        if isinstance(turn, dict) and "role" in turn:
                            if turn["role"] == "user" and "content" in turn:
                                valid_convo.append(turn)
                                expected_role = "assistant"
                            elif turn["role"] == "assistant" and "content" in turn:
                                # Assistant turns can have both content and thinking
                                valid_turn = {
                                    "role": "assistant",
                                    "content": turn.get("content", ""),
                                    "thinking": turn.get("thinking", "")
                                }
                                valid_convo.append(valid_turn)
                                expected_role = "user"
                    loaded[chat_id] = valid_convo
                conversations = loaded
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load conversations.json: {e}")
            conversations = {}
    else:
        conversations = {}

def save_conversations():
    """Save conversations to disk"""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters except '|',
    so we can use ||spoiler||.
    """
    escape_chars = r'_*[]()~`>#+-={}.!'
    for c in escape_chars:
        text = text.replace(c, "\\" + c)
    return text

def chunk_text(text: str, max_len: int = TELEGRAM_MAX_LEN):
    """
    Splits 'text' into a list of chunks up to 'max_len' characters each.
    """
    chunks = []
    idx = 0
    while idx < len(text):
        end = idx + max_len
        chunks.append(text[idx:end])
        idx = end
    return chunks

# ----------------------------------------------------
# BOT COMMANDS
# ----------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears conversation for this chat."""
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()
    await update.message.reply_text("Conversation reset. Let's start fresh!")

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch models with /set_model <model_name>."""
    global MODEL_NAME
    if not context.args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = context.args[0]
    await update.message.reply_text(f"Model changed to: {MODEL_NAME}")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug conversation structure"""
    chat_id = str(update.effective_chat.id)
    debug_conversation_structure(chat_id)
    await update.message.reply_text("Debug info written to logs")

# ----------------------------------------------------
# MAIN MESSAGE HANDLER
# ----------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main message handler with improved state machine and content separation
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    conversation = conversations.get(chat_id, [])
    conversation.append({"role": "user", "content": user_text})

    prompt_text = build_prompt(conversation)
    thinking_header_msg = await update.message.reply_text("Thinking...")

    chain_of_thought_msg = None
    public_msg = None

    # Improved state machine
    state = "outsideThink"
    outside_buffer = ""
    inside_buffer = ""
    after_buffer = ""
    full_response = ""
    last_update_time = time.time()

    payload = {
        "prompt": prompt_text,
        "model": MODEL_NAME,
        "stream": True
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                err_text = f"Error {response.status_code} from Ollama:\n{response.text}"
                await thinking_header_msg.edit_text(err_text)
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" not in chunk:
                    continue

                new_tokens = chunk["response"]
                full_response += new_tokens

                # Improved token parsing logic
                i = 0
                while i < len(new_tokens):
                    if state == "outsideThink":
                        idx = new_tokens.find("<think>", i)
                        if idx == -1:
                            outside_buffer += new_tokens[i:]
                            i = len(new_tokens)
                        else:
                            outside_buffer += new_tokens[i:idx]
                            state = "insideThink"
                            if chain_of_thought_msg is None:
                                chain_of_thought_msg = await update.effective_chat.send_message(
                                    text=escape_markdown_v2("..."),
                                    parse_mode="MarkdownV2"
                                )
                            i = idx + len("<think>")

                    elif state == "insideThink":
                        end_idx = new_tokens.find("</think>", i)
                        if end_idx == -1:
                            inside_buffer += new_tokens[i:]
                            i = len(new_tokens)
                        else:
                            inside_buffer += new_tokens[i:end_idx]
                            state = "afterThink"
                            i = end_idx + len("</think>")

                            if public_msg is None:
                                public_msg = await update.effective_chat.send_message(
                                    text="..."
                                )

                    elif state == "afterThink":
                        # Look for another <think> tag
                        next_think = new_tokens.find("<think>", i)
                        if next_think == -1:
                            after_buffer += new_tokens[i:]
                            i = len(new_tokens)
                        else:
                            after_buffer += new_tokens[i:next_think]
                            state = "insideThink"
                            i = next_think + len("<think>")

                # Partial updates with same timing
                if time.time() - last_update_time > 0.7:
                    # Update chain-of-thought
                    if chain_of_thought_msg is not None and inside_buffer:
                        escaped = escape_markdown_v2(inside_buffer)
                        spoiler_text = f"||{escaped}||"
                        cots_chunks = chunk_text(spoiler_text, TELEGRAM_MAX_LEN)
                        try:
                            if len(cots_chunks) == 1:
                                await chain_of_thought_msg.edit_text(
                                    cots_chunks[0],
                                    parse_mode="MarkdownV2"
                                )
                        except Exception as e:
                            logging.error(f"Error updating thinking: {e}")

                    # Update public message
                    if public_msg is not None:
                        current_public = outside_buffer + after_buffer
                        if current_public.strip():
                            try:
                                await public_msg.edit_text(current_public)
                            except Exception as e:
                                logging.error(f"Error updating public: {e}")

                    last_update_time = time.time()

    except requests.exceptions.RequestException as e:
        err_msg = f"Error connecting to Ollama:\n{e}"
        await thinking_header_msg.edit_text(err_msg)
        return

    # Clean up the response content
    final_thinking = inside_buffer.strip()
    final_public = (outside_buffer + after_buffer).strip()

    # Fix any lingering tags that might have leaked
    final_public = re.sub(r'<think>.*?</think>', '', final_public, flags=re.DOTALL)
    final_public = final_public.replace('<think>', '').replace('</think>', '')

    # Store the cleaned-up response
    if final_public or final_thinking:
        conversation.append({
            "role": "assistant",
            "thinking": final_thinking,
            "content": final_public
        })
        conversation = trim_conversation(conversation)
        conversations[chat_id] = conversation
        save_conversations()

    # Final message updates
    try:
        await thinking_header_msg.delete()
    except:
        pass

    if chain_of_thought_msg is not None and final_thinking:
        spoiler_text = f"||{escape_markdown_v2(final_thinking)}||"
        cots_chunks = chunk_text(spoiler_text)
        for i, chunk in enumerate(cots_chunks):
            if i == 0:
                await chain_of_thought_msg.edit_text(chunk, parse_mode="MarkdownV2")
            else:
                await update.effective_chat.send_message(chunk, parse_mode="MarkdownV2")

    if final_public:
        pub_chunks = chunk_text(final_public)
        if public_msg is None:
            public_msg = await update.effective_chat.send_message("...")
        
        for i, chunk in enumerate(pub_chunks):
            if i == 0:
                await public_msg.edit_text(chunk)
            else:
                await update.effective_chat.send_message(chunk)

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(CommandHandler("debug", debug_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()