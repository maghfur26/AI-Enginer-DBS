"""
AE2 — Inference Cache (Sprint 5)
Optimasi performa inference menggunakan Redis cache.
Menghindari prediksi ulang untuk bahan yang sama.
"""

import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis tidak terinstall. Cache dinonaktifkan.")


class InferenceCache:
    """
    Cache layer untuk hasil prediksi bahan.
    - Key: hash dari nama bahan (normalized)
    - Value: JSON hasil prediksi
    - TTL: 1 hari (86400 detik)
    """

    TTL_SECONDS = 86400  # 1 hari

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._client: Optional[Any] = None
        self._enabled = REDIS_AVAILABLE

    async def connect(self) -> None:
        if not self._enabled:
            return
        try:
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._client.ping()
            logger.info("✅ Redis cache terhubung.")
        except Exception as e:
            logger.warning(f"Redis tidak dapat terhubung: {e}. Cache dinonaktifkan.")
            self._client = None
            self._enabled = False

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()

    def _make_key(self, ingredient: str) -> str:
        """Buat cache key dari nama bahan."""
        hashed = hashlib.md5(ingredient.lower().strip().encode()).hexdigest()
        return f"olah:inference:{hashed}"

    async def get(self, ingredient: str) -> Optional[dict]:
        """Ambil hasil prediksi dari cache."""
        if not self._enabled or not self._client:
            return None
        try:
            key = self._make_key(ingredient)
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache HIT: {ingredient}")
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
        return None

    async def set(self, ingredient: str, prediction: dict) -> None:
        """Simpan hasil prediksi ke cache."""
        if not self._enabled or not self._client:
            return
        try:
            key = self._make_key(ingredient)
            await self._client.setex(key, self.TTL_SECONDS, json.dumps(prediction))
            logger.debug(f"Cache SET: {ingredient}")
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")

    async def invalidate(self, ingredient: str) -> None:
        """Hapus cache untuk satu bahan (misal setelah model update)."""
        if not self._enabled or not self._client:
            return
        try:
            key = self._make_key(ingredient)
            await self._client.delete(key)
        except Exception as e:
            logger.warning(f"Cache INVALIDATE error: {e}")

    async def flush_all(self) -> None:
        """Flush seluruh cache OLAH inference (saat model di-update dari AE1)."""
        if not self._enabled or not self._client:
            return
        try:
            keys = await self._client.keys("olah:inference:*")
            if keys:
                await self._client.delete(*keys)
                logger.info(f"Cache flushed: {len(keys)} keys dihapus.")
        except Exception as e:
            logger.warning(f"Cache FLUSH error: {e}")

    async def stats(self) -> dict:
        """Info statistik cache untuk monitoring."""
        if not self._enabled or not self._client:
            return {"enabled": False}
        try:
            keys = await self._client.keys("olah:inference:*")
            info = await self._client.info("memory")
            return {
                "enabled": True,
                "total_cached_ingredients": len(keys),
                "memory_used": info.get("used_memory_human", "unknown"),
                "ttl_seconds": self.TTL_SECONDS,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


# Singleton instance
cache = InferenceCache()
