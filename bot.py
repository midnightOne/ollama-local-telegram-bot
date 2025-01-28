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

# Telegram’s nominal max message length
TELEGRAM_MAX_LEN = 4096

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# We’ll store conversation history in JSON on disk
conversations = {}

def load_conversations():
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
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False)

# ----------------------------------------------------
# HELPER: ESCAPE MarkdownV2 (but allow "||" for spoilers)
# ----------------------------------------------------
def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters except '|',
    so we can use ||spoiler||. If you want pipe symbols inside your text
    to also be escaped, remove '|' from the logic below.
    """
    # Characters that must be escaped in MarkdownV2:
    # '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '{', '}', '.', '!'
    # We'll omit '|' from this list so the double-pipe spoiler syntax remains intact.
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

# ----------------------------------------------------
# MAIN MESSAGE HANDLER
# ----------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    A more advanced approach:
      1) Post "Thinking:" as Message #1 immediately.
      2) Wait for tokens from Ollama. We maintain a small "state machine":
         - "outsideThink" initially
         - when we detect <think> we switch to "insideThink"
         - when we detect </think> we switch to "afterThink"
      3) As soon as we see <think>, create Message #2 for chain-of-thought spoilers,
         update it in real-time with partial text.
      4) After we see </think>, create Message #3 for normal text,
         update it in real-time with partial text.
      5) We chunk messages that exceed Telegram's limit.
      6) Only at the very end do we finalize everything.
    7) For conversation history, store only the user-facing text
       (no <think>).
    """
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text

    # Retrieve or init conversation
    conversation = conversations.get(chat_id, [])
    conversation.append({"role": "user", "content": user_text})

    # Build the prompt from entire conversation
    prompt_text = "You are a helpful assistant.\n"
    for turn in conversation:
        if turn["role"] == "user":
            prompt_text += f"User: {turn['content']}\n"
        else:
            prompt_text += f"Assistant: {turn['content']}\n"
    prompt_text += "Assistant:"

    # --------------------------
    # 1) Immediately post "Thinking:" message
    # --------------------------
    thinking_header_msg = await update.message.reply_text("Thinking...")

    # We'll create placeholders for the chain-of-thought message (#2) and the public message (#3).
    chain_of_thought_msg = None
    public_msg = None

    # We maintain a small finite-state machine to track whether
    # we are "outsideThink", "insideThink", or "afterThink".
    # - "outsideThink": we haven't encountered <think> yet (all tokens here are public)
    # - "insideThink": we are inside <think>...</think> (all tokens here are chain-of-thought)
    # - "afterThink": we've encountered </think>, so everything else is public text afterwards
    state = "outsideThink"

    # We'll store partial tokens in three buffers
    outside_buffer = ""  # Accumulate text outside <think>
    inside_buffer = ""   # Accumulate text inside <think>
    after_buffer = ""    # Accumulate text after </think>

    # This helps us do partial updates once we have a valid second or third message

    last_update_time = time.time()

    # Prepare to stream from Ollama
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

                # Attempt to parse JSON chunk
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" not in chunk:
                    continue

                new_tokens = chunk["response"]

                # We'll parse 'new_tokens' for <think> or </think> transitions
                # Because these can arrive mid-chunk, we do a mini incremental parse:
                i = 0
                while i < len(new_tokens):
                    if state == "outsideThink":
                        # Look for "<think>"
                        idx = new_tokens.find("<think>", i)
                        if idx == -1:
                            # No <think> in this substring
                            outside_buffer += new_tokens[i:]
                            break
                        else:
                            # Found <think>
                            # Everything before <think> is outside text
                            outside_buffer += new_tokens[i:idx]
                            # Switch state to "insideThink"
                            state = "insideThink"
                            # We'll create the chain-of-thought message #2
                            if chain_of_thought_msg is None:
                                chain_of_thought_msg = await update.effective_chat.send_message(
                                    text="(Chain-of-thought spoilers start...)",
                                    parse_mode="MarkdownV2"
                                )
                            i = idx + len("<think>")

                    elif state == "insideThink":
                        # Look for "</think>"
                        end_idx = new_tokens.find("</think>", i)
                        if end_idx == -1:
                            # Entire substring is still inside <think>
                            inside_buffer += new_tokens[i:]
                            break
                        else:
                            # Found closing tag
                            inside_buffer += new_tokens[i:end_idx]
                            # Switch state to "afterThink"
                            state = "afterThink"
                            i = end_idx + len("</think>")

                            # Now we create the public message #3
                            if public_msg is None:
                                public_msg = await update.effective_chat.send_message(
                                    text="(Public answer streaming...)"
                                )

                    elif state == "afterThink":
                        # All tokens go to 'after_buffer'
                        after_buffer += new_tokens[i:]
                        break

                # End while
                # We do partial updates if enough time has passed
                if time.time() - last_update_time > 0.7:
                    # Update chain-of-thought if we're insideThink
                    # or if we finished insideThink
                    if chain_of_thought_msg is not None:
                        # The entire inside_buffer is chain-of-thought
                        # Escape for MarkdownV2 and wrap in spoilers
                        escaped = escape_markdown_v2(inside_buffer)
                        spoiler_text = f"||{escaped}||"
                        # If it's > 4096, chunk it
                        cots_chunks = chunk_text(spoiler_text, TELEGRAM_MAX_LEN)
                        # For partial updates, we just edit the last chunk in place
                        # (It's advanced to do multiple messages mid-stream,
                        # but let's keep it simpler: we only do partial
                        # updates with the final chunk.)
                        try:
                            if len(cots_chunks) == 1:
                                await chain_of_thought_msg.edit_text(
                                    cots_chunks[0], parse_mode="MarkdownV2"
                                )
                            else:
                                # We'll just show a note that it's too big
                                # and post the final chunk for partial updates
                                # (You can expand if you want)
                                await chain_of_thought_msg.edit_text(
                                    "(Chain-of-thought text too large, showing last chunk)",
                                    parse_mode="MarkdownV2"
                                )
                                await update.effective_chat.send_message(
                                    cots_chunks[-1], parse_mode="MarkdownV2"
                                )
                        except:
                            pass

                    # Update public message if we are in "afterThink" or no <think> found yet
                    # Actually, we show partial for outside_buffer only if we haven't encountered <think>?
                    # But user wants the *third message* to appear only after <think> ends.
                    if public_msg is not None:
                        # We show after_buffer in public_msg
                        pub_chunks = chunk_text(after_buffer)
                        try:
                            if len(pub_chunks) == 0:
                                await public_msg.edit_text("(no public text yet)")
                            elif len(pub_chunks) == 1:
                                await public_msg.edit_text(pub_chunks[0])
                            else:
                                # same logic as chain-of-thought
                                await public_msg.edit_text("(Public text is large, last chunk shown)")
                                await update.effective_chat.send_message(pub_chunks[-1])
                        except:
                            pass

                    last_update_time = time.time()

            # End of streaming
    except requests.exceptions.RequestException as e:
        err_msg = f"Error connecting to Ollama:\n{e}"
        await thinking_header_msg.edit_text(err_msg)
        return

    # -----------------------------------------------------------------
    # FINISHED streaming. Do one final update for chain-of-thought and public text
    # -----------------------------------------------------------------
    # The final outside_buffer is text that occurred before <think>.
    # The final inside_buffer is the chain-of-thought text.
    # The final after_buffer is the public text after </think>.

    # 2) chain-of-thought final update
    if chain_of_thought_msg is not None:
        # Escape and chunk
        spoiler_text = f"||{escape_markdown_v2(inside_buffer)}||" if inside_buffer else "||(empty)||"
        cots_chunks = chunk_text(spoiler_text)
        if len(cots_chunks) == 1:
            try:
                await chain_of_thought_msg.edit_text(cots_chunks[0], parse_mode="MarkdownV2")
            except:
                pass
        else:
            # Large chain-of-thought
            try:
                await chain_of_thought_msg.edit_text("(Final chain-of-thought is large, splitting...)",
                                                     parse_mode="MarkdownV2")
            except:
                pass
            for c in cots_chunks:
                await update.effective_chat.send_message(c, parse_mode="MarkdownV2")

    # 3) public text final update
    # The "public" text is (outside_buffer + after_buffer) if you want everything
    # that wasn't inside <think> to be displayed. Or just after_buffer if you only want
    # the text that came *after* the chain-of-thought.
    # Decide your logic. For simplicity, let's assume user wants final answer = outside_buffer + after_buffer
    final_public = outside_buffer + after_buffer

    if not final_public.strip():
        final_public = "(no public text)"

    if public_msg is None:
        # Means we never encountered a complete <think> block
        # so let's just post the final text as the "public" message
        public_msg = await update.effective_chat.send_message("(Public answer finalizing...)")

    # Split into chunks
    pub_chunks = chunk_text(final_public)
    if len(pub_chunks) == 1:
        try:
            await public_msg.edit_text(pub_chunks[0] if pub_chunks[0] else "(empty)")
        except:
            pass
    else:
        try:
            await public_msg.edit_text("(Final text is large, splitting...)")
        except:
            pass
        for c in pub_chunks:
            await update.effective_chat.send_message(c if c else "(empty)")

    # Finally, remove <think> from the conversation text, store only final_public
    # so chain-of-thought won't feed back into the next prompt
    conversation.append({"role": "assistant", "content": final_public})
    conversations[chat_id] = conversation
    save_conversations()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    load_conversations()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_model", set_model_command))
    # Add more commands if you like

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
