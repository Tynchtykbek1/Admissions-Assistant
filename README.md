# Admissions RAG Assistant

The project contains a FastAPI admissions-document backend and a Telegram bot. It supports PDF/TXT uploads, multilingual semantic retrieval, SQLite persistence, and Gemini or OpenAI answer generation.

## Run with Docker Compose

Docker Compose builds one application image and runs two services:

- `backend` runs FastAPI on port 8000.
- `telegram-bot` runs Telegram long polling and sends questions to `http://backend:8000` over the private Compose network. `localhost` inside the bot container would refer to the bot container itself, not FastAPI.

Copy `.env.example` to `.env`, then add your provider and Telegram credentials. Compose reads environment variables from `.env`; the file is excluded from the Docker image and Git.

Start locally:

```shell
docker compose up --build
```

Start in the background:

```shell
docker compose up -d --build
```

View all logs:

```shell
docker compose logs -f
```

View only backend logs:

```shell
docker compose logs -f backend
```

View only Telegram bot logs:

```shell
docker compose logs -f telegram-bot
```

Stop the containers while keeping stored data:

```shell
docker compose down
```

Stop the containers and delete their named volumes:

```shell
docker compose down -v
```

**Warning:** `docker compose down -v` permanently deletes the Docker SQLite database, uploaded documents, and downloaded model cache.

The backend is available from the host at <http://localhost:8000>. Its inexpensive health endpoint is <http://localhost:8000/health>. The web interface remains available at <http://localhost:8000/ui>.

SQLite data is stored in the `admissions_data` named volume at `/data/admissions.db` inside the backend container. Uploaded files use a separate `uploaded_documents` volume. The local `admissions.db` and local `uploads/` directory are never copied into the image.

The multilingual sentence-transformers model is downloaded on first use. The first upload or semantic question may therefore take longer. Downloads are retained in the `huggingface_cache` named volume for later container runs.

The Docker image installs PyTorch from the official CPU-only package index before installing `sentence-transformers`. No CUDA-capable GPU or NVIDIA runtime is required.

## Environment variables

See `.env.example` for all supported settings. Important values include the LLM provider and model, provider API keys, Telegram token, embedding model, database path, and semantic retrieval thresholds. Docker Compose sets container-specific `DATABASE_PATH` and `BACKEND_URL` values automatically.

## Run tests locally

```shell
python -m pytest -q
python test_api.py
```

The API smoke test creates and removes a temporary SQLite database; it does not use the persistent application database.
