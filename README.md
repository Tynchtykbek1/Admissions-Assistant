# Admissions RAG Assistant

FastAPI and Telegram RAG assistant for admissions documents. It supports PDF/TXT
uploads, multilingual semantic retrieval, conversation-aware follow-up questions,
SQLite persistence, isolated active documents, and Gemini or OpenAI.

## Architecture

```text
Browser / Telegram
        |
        v
FastAPI /chat
        |
        +--> SQLite conversation + recent messages
        |
        +--> follow-up detector --> optional question rewrite
        |
        +--> active document ID --> document-scoped chunks
        |
        +--> semantic / exact FAQ retrieval
        |
        +--> structured Gemini/OpenAI answer
        |
        +--> persisted assistant turn + safe sources
```

Conversation history helps interpret references such as “А раньше можно?” or
“А кому написать?”. Factual answers still come only from retrieved content in the
conversation's uploaded document. Previous assistant messages are not treated as
authoritative factual sources.

## Conversation memory

`POST /chat` resolves or creates a conversation, loads a bounded recent history,
stores the current user message, optionally rewrites contextual follow-ups into a
standalone retrieval question, retrieves from the active document, generates a
validated answer, and stores the assistant response.

Rewriting is skipped for standalone questions. If rewriting fails, times out, or
returns an invalid result, retrieval safely uses the original question. The current
message is never duplicated in the history sent to the model.

Example:

```text
User: Когда начинается подача?
Assistant: Подача начинается в середине декабря.
User: А раньше можно?

Retrieval query: Можно ли подать документы раньше середины декабря?
```

## Document isolation and uploads

Every upload receives a database document ID and a UUID-based stored filename. The
original filename is retained for display. A conversation has one active document
ID, and retrieval loads chunks only from that document. There is no mutable global
“latest document” list shared by all users.

The browser uses a persistent local conversation ID. Telegram identifies a
conversation using its chat and user IDs. Upload clients can send
`conversation_id`, `external_chat_id`, and `external_user_id` multipart fields to
make the new document active for that conversation.

Uploads:

- accept `.txt` and `.pdf` only;
- enforce `MAX_UPLOAD_SIZE_MB` while reading bounded chunks;
- check PDF signatures and basic TXT validity;
- reject empty or unreadable/scanned documents;
- persist the document and all chunks in one SQLite transaction;
- never return internal filesystem paths.

## API

Primary endpoint:

```http
POST /chat
Content-Type: application/json

{
  "question": "А раньше можно?",
  "conversation_id": "optional",
  "external_chat_id": "optional",
  "external_user_id": "optional",
  "document_id": 12
}
```

The response includes `question`, `standalone_question`, `answer`, `status`,
`sources`, `conversation_id`, document metadata, provider, and retrieval/provider
timings. Supported answer statuses are:

- `success`
- `partial_information`
- `insufficient_document_information`
- `provider_unavailable` for infrastructure/provider failures

`POST /ask-llm` delegates to the same flow for backward compatibility.
`/ask` and `/ask-semantic` remain available for the browser's basic modes.

Other endpoints:

- `GET /health` — cheap process liveness.
- `GET /ready` — database and provider-configuration readiness; no paid LLM call.
- `POST /conversation/reset` — clears messages but keeps the active document.
- `GET /conversation/status` — backend and active-document status.

## Telegram commands

- `/start` — introduction.
- `/help` — usage, follow-ups, and reset guidance.
- `/reset` — clear this chat's persisted message history, retaining its document.
- `/status` — check backend reachability and active document filename.

Questions from the same Telegram chat are serialized so replies cannot overtake one
another. Different chats are processed concurrently. Backend/provider text is
escaped before Telegram HTML rendering.

## Configuration

Copy `.env.example` to `.env` and set credentials. Never commit `.env`.

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_MODEL`, `OPENAI_MODEL` | required | Main answer model |
| `QUESTION_REWRITE_MODEL` | main model | Optional dedicated rewrite model |
| `QUESTION_REWRITE_ENABLED` | `true` | Enable contextual rewriting |
| `CHAT_HISTORY_LIMIT` | `8` | Recent persisted messages used for context |
| `SEMANTIC_TOP_K` | `5` | Maximum retrieved chunks |
| `SEMANTIC_SCORE_THRESHOLD` | `0.20` | Primary semantic threshold |
| `SEMANTIC_FALLBACK_SCORE_THRESHOLD` | `0.18` | Conservative fallback threshold |
| `SEMANTIC_FALLBACK_SAFE_MINIMUM` | `0.15` | Lower bound for fallback threshold |
| `MAX_UPLOAD_SIZE_MB` | `15` | Upload limit |
| `DOCUMENT_CACHE_SIZE` | `8` | Bounded document chunk cache |
| `LOG_LEVEL` | `INFO` | Application log level |
| `DATABASE_PATH` | `admissions.db` | SQLite path |
| `EMBEDDING_MODEL_NAME` | multilingual MiniLM | Embedding model |

The application does not log API keys, Telegram tokens, full histories, complete
documents, authorization headers, or full LLM prompts.

## Docker Compose

```shell
docker compose up --build
docker compose logs -f
docker compose down
```

The backend is at <http://localhost:8000>, the UI at
<http://localhost:8000/ui>, and liveness at <http://localhost:8000/health>.

Named volumes persist:

- SQLite at `/data/admissions.db`;
- uploads at `/app/uploads`;
- Hugging Face model cache.

**Do not use `docker compose down -v` unless permanent deletion of the database,
uploads, and model cache is intentional.**

## Local startup

```shell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Run Telegram separately only when no other deployment is polling with the same
token:

```shell
python telegram_bot.py
```

## SQLite persistence and reset

Startup applies additive `CREATE TABLE IF NOT EXISTS` and column migrations.
Existing document/chunk data is retained. Conversations, active document IDs, and
messages survive restarts.

To clear one conversation without removing its document, use `/reset` or
`POST /conversation/reset`.

For a full local development reset, stop the services, back up the database if
needed, and remove only the explicitly configured development database file. Never
delete an unknown database path. In Docker, removing the `admissions_data` volume is
destructive and should be a deliberate administrative action.

## Tests

Tests use temporary databases, temporary upload directories, fake embeddings, and
mocked Gemini/OpenAI/Telegram calls:

```shell
python -m pytest -q
python -m compileall .
docker compose config --quiet
docker build -t admissions-rag-assistant:local .
```

## Limitations

- Follow-up rewriting is probabilistic when enabled, but fails back to the original
  question.
- SQLite is appropriate for this single-instance project; multiple backend replicas
  would require coordinated database/storage design.
- Scanned PDFs require OCR before upload.
- Telegram does not upload documents itself; an upload must be associated with the
  Telegram conversation identifiers through the API or an administrative workflow.
- The first embedding operation may download the configured model and take longer.
