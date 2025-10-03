"""
HRM Memory Management System

Implements model caching/pooling, LRU eviction, and cleanup of unused data structures
to optimize memory usage and prevent memory leaks in the trading system.
"""

import weakref
import gc
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
import psutil
import os

from core.logging import logger
from core.exceptions import safe_execute

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    memory_percent: float = 0.0
    model_cache_size: int = 0
    cache_memory_usage_mb: float = 0.0
    last_gc_time: float = 0.0
    gc_collections: Dict[int, int] = field(default_factory=dict)

class ModelCacheEntry:
    """Cache entry for model instances with metadata."""
    def __init__(self, model: Any, config: Dict[str, Any], size_bytes: int = 0):
        self.model = weakref.ref(model, self._cleanup)
        self.config = config
        self.size_bytes = size_bytes
        self.created_time = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.is_active = True

    def _cleanup(self, ref):
        """Called when model is garbage collected."""
        self.is_active = False

    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.last_accessed > ttl_seconds

    def is_valid(self) -> bool:
        """Check if model reference is still valid."""
        return self.is_active and self.model() is not None

class LRUModelPool:
    """
    LRU (Least Recently Used) model pool with automatic eviction.

    Maintains a pool of trained models for reuse, evicting least recently used
    models when memory limits are exceeded.
    """

    def __init__(self, max_size: int = 10, max_memory_mb: float = 500.0):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="ModelCacheCleanup"
        )
        self.cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve model from cache, moving to most recently used."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_valid():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.touch()
                    model = entry.model()
                    logger.debug(f"Cache hit for model: {key}")
                    return model
                else:
                    # Invalid entry, remove it
                    del self.cache[key]
                    logger.debug(f"Removed invalid cache entry: {key}")

            logger.debug(f"Cache miss for model: {key}")
            return None

    def put(self, key: str, model: Any, config: Optional[Dict[str, Any]] = None,
            size_bytes: int = 0) -> bool:
        """Add model to cache, evicting if necessary."""
        with self._lock:
            if not key or not model:
                return False

            # Check if already exists
            if key in self.cache:
                self.cache.move_to_end(key)
                entry = self.cache[key]
                entry.touch()
                entry.size_bytes = size_bytes or entry.size_bytes
                return True

            # Estimate memory usage if not provided
            if size_bytes == 0:
                size_bytes = self._estimate_model_size(model)

            # Check memory limits before adding
            current_memory_mb = self._get_current_memory_usage() / (1024 * 1024)

            if current_memory_mb + (size_bytes / (1024 * 1024)) > self.max_memory_mb:
                # Need to evict some entries
                if not self._evict_to_fit(size_bytes):
                    logger.warning("Cannot add model to cache - would exceed memory limits")
                    return False

            # Add new entry
            entry = ModelCacheEntry(model, config or {}, size_bytes)
            self.cache[key] = entry
            self.cache.move_to_end(key)  # Mark as most recently used

            logger.debug(f"Added model to cache: {key} (size: {size_bytes} bytes)")

            # Evict if over max size
            self._evict_lru_if_needed()

            return True

    def remove(self, key: str) -> bool:
        """Remove model from cache."""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Removed model from cache: {key}")
                return True
            return False

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared {count} models from cache")

    def cleanup_expired(self, ttl_seconds: float = 3600) -> int:
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired(ttl_seconds) or not entry.is_valid():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            valid_entries = sum(1 for entry in self.cache.values() if entry.is_valid())
            total_accesses = sum(entry.access_count for entry in self.cache.values())

            return {
                'cache_size': len(self.cache),
                'valid_entries': valid_entries,
                'total_memory_bytes': total_size,
                'total_memory_mb': total_size / (1024 * 1024),
                'hit_ratio_estimate': self._calculate_hit_ratio(),
                'total_accesses': total_accesses,
                'entries': [{
                    'key': key,
                    'size_bytes': entry.size_bytes,
                    'access_count': entry.access_count,
                    'last_accessed': entry.last_accessed,
                    'is_valid': entry.is_valid()
                } for key, entry in self.cache.items()]
            }

    def _evict_lru_if_needed(self):
        """Evict least recently used entries to stay within size limits."""
        while len(self.cache) > self.max_size:
            oldest_key, oldest_entry = next(iter(self.cache.items()))
            self.cache.popitem(last=False)  # Remove oldest (left side)
            logger.info(f"Evicted LRU model from cache: {oldest_key}")

    def _evict_to_fit(self, required_bytes: int) -> bool:
        """Evict entries until there's enough space for required_bytes."""
        required_mb = required_bytes / (1024 * 1024)
        current_mb = self._get_current_memory_usage() / (1024 * 1024)

        if required_mb > self.max_memory_mb:
            logger.error(f"Required memory {required_mb:.2f}MB exceeds max cache memory")
            return False

        # Evict from least recently used until we have space
        while self.cache and (current_mb + required_mb > self.max_memory_mb):
            oldest_key, oldest_entry = next(iter(self.cache.items()))
            current_mb -= oldest_entry.size_bytes / (1024 * 1024)
            self.cache.popitem(last=False)
            logger.info(f"Evicted model from cache to fit new entry: {oldest_key}")

        return True

    def _estimate_model_size(self, model: Any) -> int:
        """Estimate memory usage of a model (rough approximation)."""
        try:
            # Use sys.getsizeof for rough estimate
            import sys

            # For most ML models, this is very rough
            # In production, you'd use more sophisticated memory profiling
            base_size = sys.getsizeof(model)

            # Add some overhead for model weights/biases
            if hasattr(model, 'parameters'):
                # PyTorch-style models
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                base_size += param_size
            elif hasattr(model, 'coef_'):
                # Scikit-learn models
                import numpy as np
                base_size += model.coef_.nbytes if hasattr(model.coef_, 'nbytes') else 0

            return max(base_size, 1024 * 1024)  # Minimum 1MB estimate

        except Exception:
            return 1024 * 1024  # Default 1MB estimate

    def _get_current_memory_usage(self) -> int:
        """Get current cache memory usage."""
        return sum(entry.size_bytes for entry in self.cache.values())

    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (rough estimate)."""
        # This is a simplification - in practice you'd track hits/misses
        if not self.cache:
            return 0.0

        total_accesses = sum(entry.access_count for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0

        # Assume frequently accessed items have better hit ratios
        weighted_hits = sum(
            min(entry.access_count / max(1, len(self.cache)), 1.0)
            for entry in self.cache.values()
        )
        return weighted_hits / len(self.cache)

    def _cleanup_worker(self):
        """Background worker for periodic cleanup."""
        while True:
            time.sleep(300)  # Cleanup every 5 minutes
            try:
                self.cleanup_expired()
                gc.collect()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

class HRMMemoryManager:
    """
    Central HRM memory management system.

    Manages model caching, data structure cleanup, and memory monitoring
    across the entire trading system.
    """

    def __init__(self, model_cache_size: int = 10, model_memory_limit_mb: float = 500.0,
                 enable_gc_tracking: bool = True):
        self.model_pool = LRUModelPool(model_cache_size, model_memory_limit_mb)
        self.enable_gc_tracking = enable_gc_tracking

        # WeakKeyDictionary to track data structures to cleanup
        self.temp_data_structures = weakref.WeakKeyDictionary()
        self.cleanup_callbacks = []

        # Monitoring
        self.start_time = time.time()
        self.gc_stats = {}

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_worker,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()

    def cache_model(self, model_key: str, model: Any, config: Optional[Dict[str, Any]] = None) -> bool:
        """Cache a model instance."""
        return self.model_pool.put(model_key, model, config)

    def get_cached_model(self, model_key: str) -> Optional[Any]:
        """Retrieve cached model."""
        return self.model_pool.get(model_key)

    def remove_cached_model(self, model_key: str) -> bool:
        """Remove model from cache."""
        return self.model_pool.remove(model_key)

    def register_temp_data(self, data: Any, cleanup_func: Optional[Callable] = None):
        """
        Register temporary data structure for cleanup.

        Args:
            data: Data structure to track
            cleanup_func: Optional cleanup function to call when garbage collected
        """
        if cleanup_func:
            def cleanup(ref):
                safe_execute(cleanup_func, data)

            # Add cleanup via weakref callback
            weakref.ref(data, cleanup)
        else:
            self.temp_data_structures[data] = True

    def manual_cleanup(self, force_gc: bool = False):
        """Perform manual memory cleanup."""
        logger.info("Performing manual memory cleanup")

        # Clear expired cache entries
        expired_count = self.model_pool.cleanup_expired()

        # Clear temp data structures (garbage collected automatically)
        temp_count = len(list(self.temp_data_structures.keys()))

        # Force garbage collection if requested
        if force_gc:
            collected = gc.collect(2)  # Full collection
            logger.info(f"Manual GC collected {collected} objects")

        # Call registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            safe_execute(callback)

        # Update GC stats
        self._update_gc_stats()

        logger.info(f"Memory cleanup completed - expired cache: {expired_count}, temp objects: {temp_count}")

    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            cache_stats = self.model_pool.get_stats()

            stats = MemoryStats()
            stats.total_memory_mb = memory_info.rss / (1024 * 1024)
            stats.used_memory_mb = stats.total_memory_mb
            stats.model_cache_size = cache_stats['valid_entries']
            stats.cache_memory_usage_mb = cache_stats['total_memory_mb']
            stats.last_gc_time = time.time()  # Would need to track actual GC timing
            stats.gc_collections = self.gc_stats.copy()

            # Get system memory info
            system_memory = psutil.virtual_memory()
            stats.available_memory_mb = system_memory.available / (1024 * 1024)
            stats.memory_percent = system_memory.percent

        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")
            stats = MemoryStats()

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        return self.model_pool.get_stats()

    def set_memory_limits(self, max_models: int, max_memory_mb: float):
        """Update memory limits for model caching."""
        self.model_pool.max_size = max_models
        self.model_pool.max_memory_mb = max_memory_mb
        logger.info(f"Updated memory limits - max models: {max_models}, max memory: {max_memory_mb}MB")

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def force_gc_cycle(self) -> Dict[int, int]:
        """Force a garbage collection cycle and return collection stats."""
        prev_stats = gc.get_stats()
        collected = gc.collect(2)  # Full collection

        # Calculate differences
        stats_diff = {}
        for i, gen_stats in enumerate(gc.get_stats()):
            prev_gen = prev_stats[i]
            stats_diff[i] = gen_stats['collected'] - prev_gen['collected']

        self._update_gc_stats()
        logger.info(f"GC cycle completed - collected: {collected}, by generation: {stats_diff}")

        return stats_diff

    def _update_gc_stats(self):
        """Update GC statistics."""
        if self.enable_gc_tracking:
            self.gc_stats = gc.get_stats()

    def _memory_monitor_worker(self):
        """Background memory monitoring worker."""
        while True:
            time.sleep(60)  # Monitor every minute

            try:
                stats = self.get_memory_stats()

                # Log warnings for high memory usage
                if stats.memory_percent > 85:
                    logger.warning(".2f")
                elif stats.memory_percent > 95:
                    logger.error(".2f")

                # Log cache usage warnings
                if stats.cache_memory_usage_mb > self.model_pool.max_memory_mb * 0.8:
                    logger.warning(".2f")

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

# Global memory manager instance
_memory_manager = None

def get_memory_manager(model_cache_size: int = 10, model_memory_limit_mb: float = 500.0) -> HRMMemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = HRMMemoryManager(model_cache_size, model_memory_limit_mb)
    return _memory_manager

def cache_model(key: str, model: Any, config: Optional[Dict[str, Any]] = None) -> bool:
    """Cache a model globally."""
    return get_memory_manager().cache_model(key, model, config)

def get_cached_model(key: str) -> Optional[Any]:
    """Get cached model globally."""
    return get_memory_manager().get_cached_model(key)

# Example usage:
#
# from core.memory_manager import get_memory_manager, cache_model, get_cached_model
#
# # Get memory manager
# mem_mgr = get_memory_manager()
#
# # Cache trained model
# trainer = L1ModelTrainer()
# model = trainer.train_model()
# cache_model('l1_lr_model', model, {'features': ['rsi', 'macd']})
#
# # Retrieve from cache later
# cached_model = get_cached_model('l1_lr_model')
# if cached_model:
#     prediction = cached_model.predict(new_data)
#
# # Check memory stats
# stats = mem_mgr.get_memory_stats()
# print(f"Total memory: {stats.total_memory_mb:.1f}MB")
#
# # Manual cleanup
# mem_mgr.manual_cleanup(force_gc=True)
