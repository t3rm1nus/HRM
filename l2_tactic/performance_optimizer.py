
# performance_optimizer.py
"""
L2 Tactical — Performance Optimizer
===================================

Este módulo implementa optimizaciones de rendimiento para L2, cubriendo
las funcionalidades del TODO #15:

1) Cache inteligente de predicciones (TTL + LRU + invalidación por input hash)
2) Batching/micro-batching de requests al modelo de IA (async-friendly)
3) Lazy loading / caching de features costosas (con TTL e invalidación por timestamp)
4) Paralelización segura donde tiene sentido (thread-pool para I/O/C-extensions)
5) Prefetch opcional de features (warmup) y rate limiting de llamadas al modelo

Diseñado para integrarse con:
- ai_model_integration.AIModelWrapper (cualquier objeto con predict / predict_batch)
- signal_generator.L2TacticProcessor (inyectando el wrapper OptimizedModel)
- signal_composer / position_sizer / risk_controls de forma transparente

No requiere dependencias externas. Si está instalado pandas, el batcher lo aprovecha.

Uso rápido
----------
from l2_tactic.performance_optimizer import PerformanceOptimizer, PerfConfig

optimizer = PerformanceOptimizer()              # usa PerfConfig() por defecto
model = AIModelWrapper(cfg)                     # tu wrapper actual
opt_model = optimizer.wrap_model(model, "finrl_ensemble_v1")

# en código async (recomendado)
pred = await opt_model.predict_async(symbol="BTC/USDT", horizon="1h", features=feat_row)

# en código sync
pred = opt_model.predict(symbol="BTC/USDT", horizon="1h", features=feat_row)

# features costosas (lazy + cache)
f = await optimizer.features.get_or_compute_async(
    key=("BTC/USDT", "1h", last_bar_ts), loader=lambda: compute_features(...)
)

Al apagar la app:
await optimizer.aclose()

"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha1
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pandas as pd  # opcional
except Exception:  # pragma: no cover - compat si no hay pandas
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class PerfConfig:
    # --- Cache de predicciones ---
    enable_cache: bool = True
    max_cache_items: int = 10_000
    prediction_ttl_s: int = 300  # 5 min
    hash_precision: int = 8      # precisión al redondear features al hashear

    # --- Batching ---
    enable_batching: bool = True
    batch_window_ms: int = 20           # ventana para micro-batch
    max_batch_size: int = 256           # tamaño máximo por batch

    # --- Paralelización ---
    parallel_workers: int = max(4, (os.cpu_count() or 4))
    thread_name_prefix: str = "l2-perf"

    # --- Features cache ---
    feature_ttl_s: int = 60
    prefetch_top_n: int = 0  # 0 = deshabilitado
    prefetch_interval_s: int = 30

    # --- Rate limiting para modelo ---
    rate_limit_qps: Optional[float] = None  # None = sin límite

    # --- Timeouts ---
    model_call_timeout_s: float = 5.0
    batch_flush_timeout_s: float = 5.0


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _normalize_scalar(v: Any, precision: int = 8) -> Any:
    try:
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return round(v, precision)
        if isinstance(v, (int, str, bool)):
            return v
        if isinstance(v, datetime):
            return int(v.timestamp())
        return float(v)
    except Exception:
        return str(v)


def stable_hash(obj: Any, precision: int = 8) -> str:
    """
    Hash estable de features/inputs.
    - Dict: ordenado por clave.
    - Lista/tupla: elemento a elemento.
    - Escalares: normalizados con precisión.
    """
    def _walk(o: Any) -> str:
        if isinstance(o, dict):
            items = sorted((str(k), _walk(v)) for k, v in o.items())
            inner = "|".join(f"{k}={v}" for k, v in items)
            return f"{{{inner}}}"
        elif isinstance(o, (list, tuple)):
            return f"[{','.join(_walk(x) for x in o)}]"
        else:
            return str(_normalize_scalar(o, precision))
    s = _walk(obj).encode("utf-8")
    return sha1(s).hexdigest()


# ---------------------------------------------------------------------------
# LRU + TTL Cache (thread-safe)
# ---------------------------------------------------------------------------

class _LRUTTL:
    def __init__(self, max_items: int, ttl_s: int) -> None:
        self.max_items = max_items
        self.ttl = ttl_s
        self._lock = threading.RLock()
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires, value = item
            if expires < now:
                # expirado
                try:
                    del self._store[key]
                except KeyError:
                    pass
                return None
            # move to end (recently used)
            self._store.move_to_end(key, last=True)
            return value

    def put(self, key: str, value: Any) -> None:
        expires = time.time() + self.ttl
        with self._lock:
            self._store[key] = (expires, value)
            self._store.move_to_end(key, last=True)
            while len(self._store) > self.max_items:
                self._store.popitem(last=False)

    def invalidate_prefix(self, prefix: str) -> int:
        removed = 0
        with self._lock:
            keys = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys:
                try:
                    del self._store[k]
                    removed += 1
                except KeyError:
                    pass
        return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Rate Limiter (token bucket simplificado)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, qps: float) -> None:
        self.qps = max(0.01, qps)
        self._lock = threading.Lock()
        self._tokens = self.qps
        self._last = time.time()

    def acquire(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.qps, self._tokens + elapsed * self.qps)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            # sleep out of lock
            need = (1.0 - self._tokens) / self.qps
        time.sleep(need)


# ---------------------------------------------------------------------------
# Prediction Cache
# ---------------------------------------------------------------------------

class PredictionCache:
    def __init__(self, cfg: PerfConfig) -> None:
        self.cfg = cfg
        self._cache = _LRUTTL(cfg.max_cache_items, cfg.prediction_ttl_s)

    def key(
        self, *, model_id: str, symbol: str, horizon: str, features: Any
    ) -> str:
        h = stable_hash(features, precision=self.cfg.hash_precision)
        return f"{model_id}|{symbol}|{horizon}|{h}"

    def get(self, key: str) -> Optional[Any]:
        if not self.cfg.enable_cache:
            return None
        return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        if not self.cfg.enable_cache:
            return
        self._cache.put(key, value)

    def invalidate_model(self, model_id: str) -> int:
        return self._cache.invalidate_prefix(f"{model_id}|")

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Feature Store (lazy + TTL + invalidación por timestamp)
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Cachea features costosas por clave arbitraria.
    Clave típica: (symbol, timeframe, last_bar_ts).
    """

    def __init__(self, cfg: PerfConfig) -> None:
        self.cfg = cfg
        self._cache = _LRUTTL(max_items=50_000, ttl_s=cfg.feature_ttl_s)

    def _key(self, key: Tuple[Any, ...]) -> str:
        return stable_hash(key, precision=0)  # clave pequeña, sin redondeos

    def get(self, key: Tuple[Any, ...]) -> Optional[Any]:
        return self._cache.get(self._key(key))

    def put(self, key: Tuple[Any, ...], value: Any) -> None:
        self._cache.put(self._key(key), value)

    def invalidate_prefix(self, key_prefix: Tuple[Any, ...]) -> int:
        return self._cache.invalidate_prefix(stable_hash(key_prefix, precision=0))

    async def get_or_compute_async(
        self,
        key: Tuple[Any, ...],
        loader: Callable[[], Awaitable[Any]] | Callable[[], Any],
    ) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached

        if asyncio.iscoroutinefunction(loader):  # async loader
            val = await loader()  # type: ignore
        else:
            # compute in thread to avoid blocking loop
            val = await asyncio.to_thread(loader)
        self.put(key, val)
        return val

    def get_or_compute(self, key: Tuple[Any, ...], loader: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        val = loader()
        self.put(key, val)
        return val


# ---------------------------------------------------------------------------
# Thread-backed Parallel Executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    def __init__(self, cfg: PerfConfig) -> None:
        self.cfg = cfg
        self._executor = ThreadPoolExecutor(
            max_workers=cfg.parallel_workers, thread_name_prefix=cfg.thread_name_prefix
        )

    async def map_async(self, fn: Callable[[Any], Any], iterable: Iterable[Any]) -> List[Any]:
        loop = asyncio.get_running_loop()
        return await asyncio.gather(
            *[loop.run_in_executor(self._executor, fn, x) for x in iterable]
        )

    def map(self, fn: Callable[[Any], Any], iterable: Iterable[Any]) -> List[Any]:
        # Síncrono: bloquear hasta completar
        return list(self._executor.map(fn, iterable))

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Async Micro-Batcher
# ---------------------------------------------------------------------------

class _BatchGroup:
    def __init__(
        self,
        model: Any,
        model_id: str,
        cfg: PerfConfig,
        cache: PredictionCache,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self.model = model
        self.model_id = model_id
        self.cfg = cfg
        self.cache = cache
        self.rate_limiter = rate_limiter
        self._pending: List[Tuple[asyncio.Future, Dict[str, Any]]] = []
        self._flush_scheduled: Optional[asyncio.TimerHandle] = None
        self._lock = asyncio.Lock()

    async def submit(self, req: Dict[str, Any]) -> Any:
        """
        req = {
          "symbol": str,
          "horizon": str,
          "features": dict/row/Series/np.array,
          "cache_key": str,
        }
        """
        # 1) cache hit inmediato
        if self.cfg.enable_cache and (pred := self.cache.get(req["cache_key"])) is not None:
            return pred

        # 2) agrupar para micro-batch
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._pending.append((fut, req))
            # programa flush si no existe
            if not self._flush_scheduled:
                loop = asyncio.get_running_loop()
                delay = self.cfg.batch_window_ms / 1000.0
                self._flush_scheduled = loop.call_later(delay, lambda: asyncio.create_task(self._flush()))

            # si superamos max_batch_size, flush inmediato
            if len(self._pending) >= self.cfg.max_batch_size:
                if self._flush_scheduled:
                    self._flush_scheduled.cancel()
                    self._flush_scheduled = None
                asyncio.create_task(self._flush())

        return await asyncio.wait_for(fut, timeout=self.cfg.batch_flush_timeout_s)

    async def _flush(self) -> None:
        async with self._lock:
            pending = self._pending
            self._pending = []
            self._flush_scheduled = None

        if not pending:
            return

        # rate limit
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        # preparar batch
        reqs = [r for _, r in pending]
        feats = [r["features"] for r in reqs]
        cache_keys = [r["cache_key"] for r in reqs]

        # llamar al modelo (batch si existe, si no, fallback item a item)
        try:
            if hasattr(self.model, "predict_batch"):
                preds = await _call_model_async(self.model.predict_batch, feats, self.cfg.model_call_timeout_s)
            else:
                # fallback: map paralelo con threads (permite C-extensions sin GIL)
                preds = await asyncio.gather(
                    *[ _call_model_async(self.model.predict, f, self.cfg.model_call_timeout_s) for f in feats ]
                )
        except Exception as e:
            logger.exception("[Batcher] Error calling model", exc_info=True)
            # fallar todas las futures
            for fut, _ in pending:
                if not fut.done():
                    fut.set_exception(e)
            return

        # guardar en cache + resolver futures
        for (fut, _), key, pred in zip(pending, cache_keys, preds):
            try:
                self.cache.put(key, pred)
            except Exception:
                logger.exception("[Batcher] Error caching prediction")
            if not fut.done():
                fut.set_result(pred)


async def _call_model_async(callable_fn: Callable[..., Any], arg: Any, timeout: float) -> Any:
    if asyncio.iscoroutinefunction(callable_fn):
        return await asyncio.wait_for(callable_fn(arg), timeout=timeout)
    # ejecutar llamada potencialmente pesada en thread para no bloquear
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, callable_fn, arg), timeout=timeout)


# ---------------------------------------------------------------------------
# Optimized Model Wrapper (caching + batching + rate limiting)
# ---------------------------------------------------------------------------

class OptimizedModel:
    def __init__(
        self,
        model: Any,
        model_id: str,
        cfg: PerfConfig,
        cache: PredictionCache,
        batcher: Optional[_BatchGroup],
    ) -> None:
        self._model = model
        self._id = model_id
        self._cfg = cfg
        self._cache = cache
        self._batcher = batcher

    # --------- Async ---------
    async def predict_async(
        self,
        *,
        symbol: str,
        horizon: str,
        features: Any,
    ) -> Any:
        key = self._cache.key(model_id=self._id, symbol=symbol, horizon=horizon, features=features)

        # cache hit directo
        if (pred := self._cache.get(key)) is not None:
            return pred

        req = {"symbol": symbol, "horizon": horizon, "features": features, "cache_key": key}

        if self._batcher and self._cfg.enable_batching:
            return await self._batcher.submit(req)

        # sin batcher: llamada directa async-safe
        res = await _call_model_async(self._model.predict, features, self._cfg.model_call_timeout_s)
        self._cache.put(key, res)
        return res

    # --------- Sync (fallback) ---------
    def predict(self, *, symbol: str, horizon: str, features: Any) -> Any:
        key = self._cache.key(model_id=self._id, symbol=symbol, horizon=horizon, features=features)

        if (pred := self._cache.get(key)) is not None:
            return pred

        # llamada directa
        if hasattr(self._model, "predict"):
            res = self._model.predict(features)
        elif hasattr(self._model, "predict_batch"):
            res = self._model.predict_batch([features])[0]
        else:
            raise AttributeError("Model has no predict/predict_batch")
        self._cache.put(key, res)
        return res

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_id(self) -> str:
        return self._id


# ---------------------------------------------------------------------------
# Performance Orchestrator
# ---------------------------------------------------------------------------

class PerformanceOptimizer:
    """
    Orquestador que expone:
      - .cache: PredictionCache
      - .features: FeatureStore
      - .executor: ParallelExecutor
      - .wrap_model(): OptimizedModel
      - .prefetch_features(): warmup opcional
    """

    def __init__(self, cfg: Optional[PerfConfig] = None) -> None:
        self.cfg = cfg or PerfConfig()
        self.cache = PredictionCache(self.cfg)
        self.features = FeatureStore(self.cfg)
        self.executor = ParallelExecutor(self.cfg)
        self._batch_groups: Dict[str, _BatchGroup] = {}
        self._rate_limiter: Optional[RateLimiter] = (
            RateLimiter(self.cfg.rate_limit_qps) if self.cfg.rate_limit_qps else None
        )
        self._prefetch_task: Optional[asyncio.Task] = None

    # --------------- Model Wrapping ---------------
    def wrap_model(self, model: Any, model_id: str) -> OptimizedModel:
        if self.cfg.enable_batching:
            bg = self._batch_groups.get(model_id)
            if not bg:
                bg = _BatchGroup(
                    model=model,
                    model_id=model_id,
                    cfg=self.cfg,
                    cache=self.cache,
                    rate_limiter=self._rate_limiter,
                )
                self._batch_groups[model_id] = bg
        else:
            bg = None
        return OptimizedModel(model, model_id, self.cfg, self.cache, bg)

    # --------------- Prefetch ---------------------
    async def prefetch_features(
        self,
        symbols: List[str],
        timeframe: str,
        last_bar_ts: Union[int, float, datetime],
        loader_fn: Callable[[str], Any] | Callable[[str], Awaitable[Any]],
    ) -> None:
        """
        Warming de features para N símbolos más importantes.
        loader_fn(symbol) -> features
        """
        if self.cfg.prefetch_top_n <= 0:
            return

        top = symbols[: self.cfg.prefetch_top_n]

        async def _load(sym: str) -> None:
            key = (sym, timeframe, last_bar_ts)
            try:
                await self.features.get_or_compute_async(key, lambda: loader_fn(sym))
            except Exception:
                logger.exception("[prefetch] error precargando features")

        await asyncio.gather(*[_load(s) for s in top])

    def start_periodic_prefetch(
        self,
        symbols_provider: Callable[[], List[str]],
        timeframe_provider: Callable[[], str],
        last_bar_ts_provider: Callable[[], Union[int, float, datetime]],
        loader_fn: Callable[[str], Any] | Callable[[str], Awaitable[Any]],
    ) -> None:
        """
        Lanza una tarea de prefetch periódico en background (idempotente).
        """
        if self.cfg.prefetch_top_n <= 0:
            return
        if self._prefetch_task and not self._prefetch_task.done():
            return  # ya corriendo

        async def _loop():
            while True:
                try:
                    symbols = symbols_provider()
                    timeframe = timeframe_provider()
                    ts = last_bar_ts_provider()
                    await self.prefetch_features(symbols, timeframe, ts, loader_fn)
                except asyncio.CancelledError:
                    break
                except Exception:
                    logger.exception("[prefetch] fallo en ciclo")
                await asyncio.sleep(self.cfg.prefetch_interval_s)

        try:
            loop = asyncio.get_running_loop()
            self._prefetch_task = loop.create_task(_loop())
        except RuntimeError:
            # sin loop => ignorar
            self._prefetch_task = None

    # --------------- Shutdown ---------------------
    async def aclose(self) -> None:
        if self._prefetch_task:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except Exception:
                pass
        self.executor.shutdown()
        self.cache.clear()

    # --------------- Helpers métricos -------------
    def cache_stats(self) -> Dict[str, Any]:
        # Exporta métricas básicas (placeholder)
        return {
            "cache_enabled": self.cfg.enable_cache,
            "batching_enabled": self.cfg.enable_batching,
            "feature_ttl_s": self.cfg.feature_ttl_s,
            "prediction_ttl_s": self.cfg.prediction_ttl_s,
            "max_cache_items": self.cfg.max_cache_items,
        }


# ---------------------------------------------------------------------------
# Integración ligera con L2: helpers
# ---------------------------------------------------------------------------

async def get_or_predict(
    opt_model: OptimizedModel,
    *,
    symbol: str,
    horizon: str,
    features: Any,
) -> Any:
    """
    Helper para usar desde L2TacticProcessor:
    - intenta cache/batch si fue envuelto por PerformanceOptimizer
    """
    return await opt_model.predict_async(symbol=symbol, horizon=horizon, features=features)


def get_or_predict_sync(
    opt_model: OptimizedModel,
    *,
    symbol: str,
    horizon: str,
    features: Any,
) -> Any:
    return opt_model.predict(symbol=symbol, horizon=horizon, features=features)
