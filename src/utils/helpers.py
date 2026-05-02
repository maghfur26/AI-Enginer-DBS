"""
AE2 — Utils
Helper functions: logging setup, timing decorator, response formatter.
"""

import functools
import logging
import time
from typing import Any, Callable

from src.config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Konfigurasi logging terpusat untuk seluruh modul AE2."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Kurangi verbosity library eksternal
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def timer(func: Callable) -> Callable:
    """Decorator untuk mengukur waktu eksekusi fungsi (dev/debug)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logging.getLogger(func.__module__).debug(
            f"{func.__name__} selesai dalam {elapsed} ms"
        )
        return result
    return wrapper


def format_api_response(
    data: Any,
    message: str = "success",
    status: str = "ok",
) -> dict:
    """
    Standar wrapper response untuk konsistensi format JSON.
    Digunakan opsional — endpoint utama pakai Pydantic schema langsung.
    """
    return {
        "status": status,
        "message": message,
        "data": data,
    }


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Bagi list menjadi chunk-chunk untuk batch processing."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """Load JSON dengan fallback jika parsing gagal."""
    import json
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback
