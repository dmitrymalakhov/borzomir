# Borzomir Telegram Bot

Telegram bot that talks to a local LM Studio model through the OpenAI-compatible
API. The bot is designed to run in Docker and use the LM Studio server running
on the host machine.

## Quick Start

1. Start LM Studio, load a model, and make sure the local server is running.
   In Docker, the host LM Studio endpoint is usually available as:

   ```text
   http://host.docker.internal:1234/v1
   ```

2. Create an environment file:

   ```bash
   cp .env.example .env
   ```

3. Put your Telegram bot token into `.env`:

   ```dotenv
   TELEGRAM_BOT_TOKEN=123456789:replace-with-your-token
   ```

4. Download local speech models if you want voice input and voice replies:

   ```bash
   mkdir -p models/whisper models/piper

   curl -L \
     -o models/whisper/ggml-small.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

   curl -L \
     -o models/piper/ru_RU-ruslan-medium.onnx \
     https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx

   curl -L \
     -o models/piper/ru_RU-ruslan-medium.onnx.json \
     https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json
   ```

   Speech recognition and synthesis run locally in the Docker container. The
   bot still uses Telegram Bot API to receive and send Telegram messages.

5. Start the bot:

   ```bash
   docker compose up --build -d
   ```

6. Watch logs:

   ```bash
   docker compose logs -f bot
   ```

## Configuration

`ADMIN_USER_ID` defaults to `92174505`.
When an allowlist is configured, users outside it are silently ignored. The bot
does not answer `/start`, `/help`, regular messages, files, or photos from
unauthorized users.

| Variable | Default | Description |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | required | Telegram bot token from BotFather. |
| `ADMIN_USER_ID` | `92174505` | Telegram user id with admin permissions. |
| `ALLOWED_USER_IDS` | empty | Optional comma-separated allowlist. Empty means everyone can chat. |
| `LM_STUDIO_BASE_URL` | `http://host.docker.internal:1234/v1` | LM Studio OpenAI-compatible API base URL. |
| `LM_STUDIO_MODEL` | `qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive` | Model identifier from LM Studio. |
| `DATABASE_PATH` | `/data/borzomir.sqlite3` | SQLite state database for persistent memory, chat settings, and runtime allowlist. |
| `SYSTEM_PROMPT` | Russian assistant prompt | System prompt sent to the model. |
| `MAX_HISTORY_MESSAGES` | `16` | Per-chat memory size, counted as user/assistant messages. |
| `SUMMARY_ENABLED` | `true` | Enable rolling conversation summaries when short history overflows. |
| `SUMMARY_MAX_TOKENS` | `4096` | Maximum tokens for the local model when updating summaries. |
| `MAX_IMAGE_BYTES` | `20971520` | Maximum Telegram image size to send to LM Studio. |
| `MAX_TEXT_DOCUMENT_BYTES` | `1048576` | Maximum text document size to download and analyze. |
| `LM_STUDIO_TEMPERATURE` | `0.7` | Chat completion temperature. |
| `LM_STUDIO_MAX_TOKENS` | `4096` | Maximum tokens to generate. Reasoning models can spend many tokens before producing the visible answer. |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Timeout for LM Studio requests. |
| `TELEGRAM_POLL_TIMEOUT_SECONDS` | `50` | Telegram long-poll timeout. |
| `VOICE_INPUT_ENABLED` | `true` | Enable local speech-to-text for Telegram voice messages. |
| `VOICE_REPLY_MODE` | `voice-input` | Voice reply mode: `off`, `voice-input`, or `always`. |
| `MAX_VOICE_BYTES` | `20971520` | Maximum Telegram voice file size to download. |
| `MAX_VOICE_SECONDS` | `120` | Maximum Telegram voice duration to transcribe. |
| `MAX_TTS_CHARS` | `2000` | Maximum answer length to synthesize as one voice message. Longer answers fall back to text. |
| `SPEECH_TIMEOUT_SECONDS` | `180` | Timeout for local `ffmpeg`, `whisper.cpp`, and Piper commands. |
| `WHISPER_BIN` | `/usr/local/bin/whisper-cli` | Local `whisper.cpp` CLI path. |
| `WHISPER_MODEL_PATH` | `/models/whisper/ggml-small.bin` | Local Whisper ggml model path. |
| `WHISPER_LANGUAGE` | `ru` | Recognition language hint passed to Whisper. |
| `PIPER_BIN` | `/usr/local/bin/piper` | Local Piper CLI path. |
| `PIPER_MODEL_PATH` | `/models/piper/ru_RU-ruslan-medium.onnx` | Local Piper voice model path. |
| `PIPER_CONFIG_PATH` | `/models/piper/ru_RU-ruslan-medium.onnx.json` | Local Piper voice config path. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Bot Commands

- `/start` - show a short welcome message.
- `/help` - show available commands.
- `/ask <text>` - send a prompt to LM Studio.
- `/remember <text>` - save a persistent note for the current chat.
- `/memory` - show persistent notes for the current chat.
- `/summary` - show the rolling summary of older conversation.
- `/resetsummary` - clear the rolling summary.
- `/forget` - remove persistent notes for the current chat.
- `/reset` - clear short conversation history and rolling summary for the current chat.
- `/params` - show active generation parameters.
- `/model` - show the active LM Studio model for the current chat.
- `/models` - list models available from LM Studio.
- `/admin` - show admin-only commands.
- `/health` - admin only, check LM Studio `/models`.
- `/stats` - admin only, show runtime stats.

Any regular text message is also sent to the local model. Voice messages are
transcribed locally first and then sent to the same local model.

## Memory Model

The bot uses three layers of memory:

- Short history: the latest `MAX_HISTORY_MESSAGES` user/assistant messages.
- Rolling summary: older messages are summarized by the local model and stored
  in SQLite, then added back to future requests as compact context.
- Persistent notes: facts manually saved with `/remember`.

This keeps long conversations continuous without sending the entire transcript
to LM Studio on every request. If summarization fails, the bot logs the error
and retries after later messages instead of dropping the conversation.

## Admin Controls

Admin commands are available only to `ADMIN_USER_ID`.

- `/prompt` - show the active system prompt, including chat memory notes.
- `/setprompt <text>` - set a system prompt for the current chat.
- `/resetprompt` - reset the current chat prompt to `.env`.
- `/temperature <0..2>` - set temperature for the current chat.
- `/maxtokens <number>` - set `max_tokens` for the current chat.
- `/model <number|id>` - switch the current chat to a model returned by `/models`.
- `/resetmodel` - reset the current chat model to `LM_STUDIO_MODEL` from `.env`.
- `/allowed` - show access mode and allowlists.
- `/allow <user_id>` - add a user to the runtime allowlist.
- `/deny <user_id>` - remove a user from the runtime allowlist.
- `/access open` - clear the runtime allowlist. If `ALLOWED_USER_IDS` is empty,
  the bot becomes open to everyone again.

## Images

The bot supports Telegram photo messages when the loaded LM Studio model has
vision capability. It also supports image files sent as Telegram documents.
Send a photo or image document with an optional caption, for example:

```text
Что изображено на фото?
```

If the caption starts with `/ask`, the command part is removed and the rest is
used as the image question.

## Voice Messages

The bot supports Telegram voice messages without external speech APIs:

- Telegram voice OGG/Opus is downloaded through Telegram Bot API.
- `ffmpeg` converts it to 16 kHz mono WAV.
- `whisper.cpp` transcribes the WAV locally.
- The transcript is sent to LM Studio like a normal user message.
- Piper synthesizes the answer locally with the default male Russian voice
  `ru_RU-ruslan-medium`.
- `ffmpeg` converts the Piper WAV output back to OGG/Opus for Telegram
  `sendVoice`.

By default, `VOICE_REPLY_MODE=voice-input`, so the bot answers with voice when
the user sent a voice message. Text messages still get text replies, unless the
user explicitly asks for a voice answer with phrases such as:

```text
Ответь голосом: коротко объясни Docker Compose.
```

`VOICE_REPLY_MODE=off` is a hard disable for voice replies.

## Text Documents

The bot can read text documents sent through Telegram and ask the model about
their contents. Supported formats include `.txt`, `.md`, `.csv`, `.json`,
`.yaml`, source code files, SQL, shell scripts, and logs.

Send a file with a caption such as:

```text
Summarize this file and list risky parts.
```

Large files are rejected or truncated before the request is sent to LM Studio,
so the local model does not run out of context.

## Local Checks

Run unit tests without Docker:

```bash
python3 -m unittest discover -s tests
```

Validate the Compose file:

```bash
docker compose --env-file .env.example config
```
