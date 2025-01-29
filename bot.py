import logging
import json
import os
import requests
import time
import re
import aiohttp
import asyncio
from typing import Optional
from datetime import datetime
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "deepseek-r1:8b"
CONVERSATIONS_FILE = "conversations.json"
MAX_CONVERSATION_TURNS = 10
# Telegram’s nominal max message length
TELEGRAM_MAX_LEN = 4096
UPDATE_INTERVAL = 1.2

# Set up main logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Set up raw data logger
raw_logger = logging.getLogger('raw_data')
raw_logger.setLevel(logging.INFO)
raw_handler = logging.FileHandler('raw_conversation_data.log')
raw_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
raw_logger.addHandler(raw_handler)

conversations = {}

class MessageState:
    def __init__(self):
        self.last_thinking = ""
        self.last_public = ""

def build_prompt(conversation):
    """Build the prompt with clearer instructions about thinking tags"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep your thinking concise and focused on the current task. Use <think> tags only for current reasoning, not for recapping previous context."
        }
    ]
    
    for turn in conversation[-6:]:
        if turn["role"] == "user":
            messages.append({
                "role": "user", 
                "content": turn["content"]
            })
        else:
            # For assistant turns, reconstruct with thinking tags if present
            thinking = turn.get("thinking", "").strip()
            response = turn.get("content", "").strip()
            content = f"<think>{thinking}</think>\n{response}" if thinking else response
            messages.append({
                "role": "assistant",
                "content": content
            })
    
    # For Ollama API
    payload = {
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "num_predict": 1000
        }
    }
    
    return payload

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-={}.!'
    for c in escape_chars:
        text = text.replace(c, "\\" + c)
    return text

def chunk_text(text: str, max_len: int = TELEGRAM_MAX_LEN):
    chunks = []
    idx = 0
    while idx < len(text):
        end = idx + max_len
        chunks.append(text[idx:end])
        idx = end
    return chunks

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

async def update_messages(update, chain_of_thought_msg, public_msg, inside_buffer, outside_buffer, after_buffer):
    if chain_of_thought_msg and inside_buffer:
        escaped = escape_markdown_v2(inside_buffer)
        spoiler_text = f"||{escaped}||"
        chunks = chunk_text(spoiler_text, TELEGRAM_MAX_LEN)
        try:
            if len(chunks) == 1:
                await chain_of_thought_msg.edit_text(chunks[0], parse_mode="MarkdownV2")
            else:
                await chain_of_thought_msg.edit_text(chunks[0], parse_mode="MarkdownV2")
                for chunk in chunks[1:]:
                    await update.effective_chat.send_message(chunk, parse_mode="MarkdownV2")
        except Exception as e:
            if "Message is not modified" not in str(e):
                logging.error(f"Error updating thinking: {e}")

    if public_msg:
        current_public = outside_buffer + after_buffer
        if current_public.strip():
            chunks = chunk_text(current_public)
            try:
                if len(chunks) == 1:
                    await public_msg.edit_text(chunks[0])
                else:
                    await public_msg.edit_text(chunks[0])
                    for chunk in chunks[1:]:
                        await update.effective_chat.send_message(chunk)
            except Exception as e:
                if "Message is not modified" not in str(e):
                    logging.error(f"Error updating public: {e}")
                    if "Message to edit not found" in str(e):
                        try:
                            public_msg = await update.effective_chat.send_message(current_public)
                        except Exception as send_error:
                            logging.error(f"Error sending new message: {send_error}")

# ----------------------------------------------------
# MAIN MESSAGE HANDLER
# ----------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    conversation = conversations.get(chat_id, [])
    conversation.append({"role": "user", "content": user_text})

    thinking_header_msg = await update.message.reply_text("Thinking...")

    chain_of_thought_msg = None
    public_msg = None
    complete_response_received = False
    
    state = "outsideThink"
    outside_buffer = ""
    inside_buffer = ""
    after_buffer = ""
    full_response = ""
    last_update_time = time.time()

    payload = build_prompt(conversation[-6:])
    payload["model"] = MODEL_NAME

    # Log raw prompt
    raw_logger.info(f"PAYLOAD [chat_id={chat_id}]:\n{json.dumps(payload, indent=2)}\n")

    async def ensure_messages_sent():
        nonlocal public_msg, chain_of_thought_msg
        
        if inside_buffer.strip():
            if not chain_of_thought_msg:
                try:
                    escaped = escape_markdown_v2(inside_buffer)
                    chain_of_thought_msg = await update.effective_chat.send_message(
                        f"||{escaped}||",
                        parse_mode="MarkdownV2"
                    )
                except Exception as e:
                    logging.error(f"Error sending thinking message: {e}")

        current_public = (outside_buffer + after_buffer).strip()
        if current_public:
            if not public_msg:
                try:
                    public_msg = await update.effective_chat.send_message(current_public)
                except Exception as e:
                    logging.error(f"Error sending public message: {e}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json=payload) as response:
                if response.status != 200:
                    err_text = f"Error {response.status} from Ollama:\n{await response.text()}"
                    await thinking_header_msg.edit_text(err_text)
                    return

                buffer = ""
                async for line in response.content:
                    try:
                        if not line.strip():
                            continue
                            
                        buffer += line.decode('utf-8')
                        if not buffer.endswith('\n'):
                            continue
                            
                        for chunk_line in buffer.splitlines():
                            if not chunk_line.strip():
                                continue
                                
                            try:
                                chunk = json.loads(chunk_line)
                                # raw_logger.info(f"CHUNK:\n{json.dumps(chunk, indent=2)}\n")
                            except json.JSONDecodeError:
                                continue

                            if "message" in chunk:
                                new_tokens = chunk["message"]["content"]
                            elif "response" in chunk:
                                new_tokens = chunk["response"]
                            else:
                                continue

                            full_response += new_tokens

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

                                else:
                                    after_buffer += new_tokens[i:]
                                    i = len(new_tokens)

                            if time.time() - last_update_time > UPDATE_INTERVAL:
                                await ensure_messages_sent()
                                await update_messages(
                                    update,
                                    chain_of_thought_msg,
                                    public_msg,
                                    inside_buffer,
                                    outside_buffer,
                                    after_buffer
                                )
                                last_update_time = time.time()
                                
                        buffer = ""

                    except Exception as e:
                        raw_logger.error(f"Error processing chunk: {str(e)}\nChunk: {chunk_line}")
                        continue

                complete_response_received = True

    except Exception as e:
        logging.error(f"Error in handle_message: {str(e)}")
        await thinking_header_msg.edit_text(f"Error: {str(e)}")
        return

    # Log raw response
    raw_logger.info(f"RESPONSE [chat_id={chat_id}]:\n{full_response}\n")

    try:
        await ensure_messages_sent()
        
        retry_count = 3
        for attempt in range(retry_count):
            try:
                await update_messages(
                    update,
                    chain_of_thought_msg,
                    public_msg,
                    inside_buffer,
                    outside_buffer,
                    after_buffer
                )
                break
            except Exception as e:
                if attempt == retry_count - 1:
                    logging.error(f"Final update failed after {retry_count} attempts: {e}")
                await asyncio.sleep(0.5)

    except Exception as e:
        logging.error(f"Error in final message handling: {e}")

    if complete_response_received:
        final_content = (outside_buffer + after_buffer).strip()
        if final_content or inside_buffer.strip():
            conversation.append({
                "role": "assistant",
                "thinking": inside_buffer.strip(),
                "content": final_content
            })
            conversations[chat_id] = conversation
            save_conversations()

    try:
        await thinking_header_msg.delete()
    except Exception:
        pass

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    conversations[chat_id] = []
    save_conversations()
    await update.message.reply_text("Conversation reset. Let's start fresh!")

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODEL_NAME
    if not context.args:
        await update.message.reply_text("Usage: /set_model <model_name>")
        return
    MODEL_NAME = context.args[0]
    await update.message.reply_text(f"Model changed to: {MODEL_NAME}")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    debug_conversation_structure(chat_id)
    await update.message.reply_text("Debug info written to logs")

def debug_conversation_structure(chat_id):
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

if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    application.add_handler(CommandHandler("debug", debug_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()