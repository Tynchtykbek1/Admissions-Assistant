import os
import math

import httpx
from dotenv import load_dotenv


load_dotenv()


BACKEND_TIMEOUT_DEFAULTS = {
    "BACKEND_CONNECT_TIMEOUT_SECONDS": 10.0,
    "BACKEND_READ_TIMEOUT_SECONDS": 90.0,
    "BACKEND_WRITE_TIMEOUT_SECONDS": 15.0,
    "BACKEND_POOL_TIMEOUT_SECONDS": 10.0,
}


def _read_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive number.") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a positive number.")
    return value


def load_backend_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=_read_positive_float(
            "BACKEND_CONNECT_TIMEOUT_SECONDS",
            BACKEND_TIMEOUT_DEFAULTS["BACKEND_CONNECT_TIMEOUT_SECONDS"],
        ),
        read=_read_positive_float(
            "BACKEND_READ_TIMEOUT_SECONDS",
            BACKEND_TIMEOUT_DEFAULTS["BACKEND_READ_TIMEOUT_SECONDS"],
        ),
        write=_read_positive_float(
            "BACKEND_WRITE_TIMEOUT_SECONDS",
            BACKEND_TIMEOUT_DEFAULTS["BACKEND_WRITE_TIMEOUT_SECONDS"],
        ),
        pool=_read_positive_float(
            "BACKEND_POOL_TIMEOUT_SECONDS",
            BACKEND_TIMEOUT_DEFAULTS["BACKEND_POOL_TIMEOUT_SECONDS"],
        ),
    )


BACKEND_TIMEOUT = load_backend_timeout()
