import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from telegram_bot import ask_backend
from telegram_settings import BACKEND_TIMEOUT_DEFAULTS, load_backend_timeout


TIMEOUT_ENVIRONMENT_VARIABLES = tuple(BACKEND_TIMEOUT_DEFAULTS)


def _clear_timeout_environment(monkeypatch):
    for name in TIMEOUT_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_backend_timeout_defaults(monkeypatch):
    _clear_timeout_environment(monkeypatch)
    timeout = load_backend_timeout()
    assert timeout.connect == 10.0
    assert timeout.read == 90.0
    assert timeout.write == 15.0
    assert timeout.pool == 10.0


def test_backend_timeout_environment_overrides(monkeypatch):
    monkeypatch.setenv("BACKEND_CONNECT_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("BACKEND_READ_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("BACKEND_WRITE_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("BACKEND_POOL_TIMEOUT_SECONDS", "3")
    timeout = load_backend_timeout()
    assert timeout.connect == 1.5
    assert timeout.read == 120.0
    assert timeout.write == 2.5
    assert timeout.pool == 3.0


@pytest.mark.parametrize("value", ["0", "-1", "-0.01"])
def test_backend_timeout_rejects_non_positive_values(monkeypatch, value):
    _clear_timeout_environment(monkeypatch)
    monkeypatch.setenv("BACKEND_READ_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError, match="must be a positive number"):
        load_backend_timeout()


def test_backend_timeout_rejects_non_numeric_values(monkeypatch):
    _clear_timeout_environment(monkeypatch)
    monkeypatch.setenv("BACKEND_READ_TIMEOUT_SECONDS", "not-a-number")
    with pytest.raises(RuntimeError, match="must be a positive number"):
        load_backend_timeout()


def test_chat_post_is_not_retried_after_read_timeout():
    request = httpx.Request("POST", "http://backend/chat")
    client = AsyncMock()
    client.post.side_effect = httpx.ReadTimeout("timed out", request=request)
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = client
    context_manager.__aexit__.return_value = False

    async def run_request():
        with patch("telegram_bot.httpx.AsyncClient", return_value=context_manager):
            with pytest.raises(httpx.ReadTimeout):
                await ask_backend("question", "chat", "user")

    asyncio.run(run_request())
    client.post.assert_awaited_once()
