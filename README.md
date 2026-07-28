# Admissions RAG Assistant

FastAPI and Telegram RAG assistant for admissions documents. It supports PDF/TXT
uploads, multilingual semantic retrieval, conversation-aware follow-up questions,
SQLite persistence, one administrator-configured knowledge document, and Gemini or
OpenAI.

## Architecture

```text
Telegram
        |
        v
FastAPI /chat
        |
        +--> SQLite conversation + recent messages
        |
        +--> follow-up detector --> optional question rewrite
        |
        +--> SYSTEM_DOCUMENT_ID --> document-scoped chunks
        |
        +--> semantic / exact FAQ retrieval
        |
        +--> structured Gemini/OpenAI answer
        |
        +--> persisted assistant turn + safe sources
```

Conversation history helps interpret references such as “А раньше можно?” or
“А кому написать?”. Factual answers still come only from retrieved content in the
shared system document. Previous assistant messages are not treated as
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

## Shared system document and uploads

Every upload receives a database document ID and a UUID-based stored filename. The
original filename is retained for display. In production, `SYSTEM_DOCUMENT_ID`
selects one admissions knowledge document for every Telegram conversation. New
conversations receive it automatically. At backend startup, existing conversations
with no document or an older document are synchronized to it transactionally
without changing their messages. Normal requests repair only the current
conversation when necessary. Conversation histories remain separate even though
retrieval uses the same document.

Clients cannot switch documents while system-document mode is configured. A
conflicting `document_id` receives HTTP 409; sending the configured ID is accepted
for compatibility. There is no mutable global “latest document” fallback in this
mode.

Production currently uses:

```dotenv
SYSTEM_DOCUMENT_ID=1
```

To replace the knowledge document, an administrator uploads and validates the new
PDF/TXT, notes its database document ID, changes `SYSTEM_DOCUMENT_ID` to that ID,
and recreates the containers. Startup and the next request synchronize every
conversation to the replacement document. No database deletion or manual
conversation update is required. Uploading alone does not activate a different
system document.

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
  "external_chat_id": "required",
  "external_user_id": "optional"
}
```

All conversation endpoints are Telegram-only. Requests without
`external_chat_id` return HTTP 400. A saved `conversation_id` is accepted only
when its chat and optional user identifiers match, preventing one Telegram user
from reusing another user's conversation.

`document_id` remains in the request model for backward compatibility. When
`SYSTEM_DOCUMENT_ID` is set, it cannot select any other document.

The response includes `question`, `standalone_question`, `answer`, `status`,
`sources`, `conversation_id`, document metadata, provider, and retrieval/provider
timings. Supported answer statuses are:

- `success`
- `partial_information`
- `insufficient_document_information`
- `provider_unavailable` for infrastructure/provider failures
- `system_document_unavailable` when the configured knowledge document is missing
  or empty

`POST /ask-llm` delegates to the same flow for backward compatibility.
`/ask` and `/ask-semantic` remain available as legacy Telegram-identified API
endpoints.

Other endpoints:

- `GET /health` — cheap process liveness.
- `GET /ready` — database, provider configuration, system document, and chunk
  readiness; no paid LLM call.
- `POST /conversation/reset` — clears messages and preserves/reapplies the system
  document.
- `GET /conversation/status` — backend and system-document status.

## Telegram commands

- `/start` — introduction.
- `/help` — usage, follow-ups, and reset guidance.
- `/reset` — clear this chat's persisted message history, retaining its document.
- `/status` — check backend reachability and active document filename.

Questions from the same Telegram chat are serialized so replies cannot overtake one
another. Different chats are processed concurrently. Backend/provider text is
escaped before Telegram HTML rendering.

### Local Telegram responses

The Telegram bot answers a narrow, explicit set of common questions about its
identity and capabilities without using the backend RAG pipeline. Questions about
human admissions help also return these manager contacts locally:

- Адахан — @TheLuckiestPersonEver
- Максат — @maksatuniguide

An explicit list of obvious unrelated requests, such as weather or joke questions,
receives a local reminder that the bot covers admissions. This is deliberately
conservative matching, not a general topic classifier. These local messages do not
use RAG, call the LLM provider, or get stored as unanswered questions. Normal
admissions questions continue through the backend RAG pipeline.

## Unanswered-question review

The backend records a question for later review when retrieval finds no accepted
chunks, or when retrieved context leads to the controlled
`insufficient_document_information` result. It does not record successful,
partial, provider-unavailable, or system-document-unavailable requests.

Equivalent standalone questions are deduplicated using a normalized SHA-256 hash.
The database retains the first original wording, occurrence count, highest
observed score, relevant FAQ IDs, reason, and timestamps. Telegram identity and
profile data are not stored in this table.

Until a protected administrator panel is available, export open and reviewed
questions with:

```shell
python scripts/export_unanswered_questions.py
python scripts/export_unanswered_questions.py --output unanswered.csv
python scripts/export_unanswered_questions.py --status open --output unanswered-open.csv
```

The exporter reads `DATABASE_PATH`, writes UTF-8 CSV, and never modifies records.

## Configuration

Copy `.env.example` to `.env` and set credentials. Never commit `.env`.

| Variable | Default | Purpose |
|---|---:|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `SYSTEM_DOCUMENT_ID` | unset | Positive database ID of the shared knowledge document; production uses `1` |
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
| `BACKEND_CONNECT_TIMEOUT_SECONDS` | `10` | Telegram-to-backend connection timeout |
| `BACKEND_READ_TIMEOUT_SECONDS` | `90` | Telegram wait for the backend response body |
| `BACKEND_WRITE_TIMEOUT_SECONDS` | `15` | Telegram request-body write timeout |
| `BACKEND_POOL_TIMEOUT_SECONDS` | `10` | Telegram connection-pool timeout |
| `LOG_LEVEL` | `INFO` | Application log level |
| `DATABASE_PATH` | `admissions.db` | SQLite path |
| `EMBEDDING_MODEL_NAME` | multilingual MiniLM | Embedding model |

The application does not log API keys, Telegram tokens, full histories, complete
documents, authorization headers, or full LLM prompts.

Provider responses may occasionally take longer than 45 seconds, so Telegram waits
up to 90 seconds for the backend response body by default. This is an upper bound,
not the expected duration of every request. The existing embedding model is loaded
and given one small local warmup encode during backend startup; no LLM provider is
called. `/ready` succeeds only after that initialization completes successfully,
while `/health` continues to report process availability. This hotfix does not
change the embedding model or retrieval threshold.

## Docker Compose

```shell
docker compose up --build
docker compose logs -f
docker compose down
```

The backend is at <http://localhost:8000> and liveness is at
<http://localhost:8000/health>. There is no public browser UI.

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

Startup applies additive `CREATE TABLE IF NOT EXISTS`, index, and column
migrations. Existing document/chunk data is retained. Conversations, active
document IDs, messages, and unanswered questions survive restarts. SQLite uses
WAL mode and a 30-second busy timeout.

The conversation identity migration adds uniqueness across channel, chat ID, and
optional user ID. If duplicate rows exist, the oldest conversation is retained,
all messages are reassigned to it, and only the now-empty duplicate conversation
rows are removed.

Changing `SYSTEM_DOCUMENT_ID` and recreating the application containers updates
all conversation document references in one transaction. Documents, chunks, and
messages are preserved. Zero, negative, invalid, missing, or empty configured
documents are never replaced by another document silently; readiness returns 503
and users receive a controlled knowledge-base-unavailable response.

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
python -m compileall -x ".venv" .
docker compose config --quiet
docker build -t admissions-rag-assistant:local .
```

## Limitations

- Follow-up rewriting is probabilistic when enabled, but fails back to the original
  question.
- SQLite is appropriate for this single-instance project; multiple backend replicas
  would require coordinated database/storage design.
- Scanned PDFs require OCR before upload.
- Telegram does not upload or select documents. An administrator manages the
  shared document through the upload API and `SYSTEM_DOCUMENT_ID`.
- A protected document/statistics/review administration panel is not implemented.
- The first embedding operation may download the configured model and take longer.
