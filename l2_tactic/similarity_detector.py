# l2_tactic/similarity_detector.py
"""
Similarity detector for trading signals.

This module provides functionality to detect and handle similar signals, patterns,
and market conditions in the trading system. It includes:

- Signal similarity scoring algorithms (cosine, euclidean, feature-based)
- Duplicate signal filtering
- Pattern recognition for market conditions
- Configurable similarity thresholds
- Integration with signal processing pipeline
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import TacticalSignal
from .l2_utils import safe_float
from core.logging import logger


class SimilarityAlgorithm(Enum):
    """Available similarity algorithms"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    FEATURE_WEIGHTED = "feature_weighted"


class SimilarityThreshold(Enum):
    """Predefined similarity thresholds"""
    STRICT = 0.9  # Very similar signals
    HIGH = 0.8    # Highly similar
    MEDIUM = 0.7  # Moderately similar
    LOW = 0.6     # Somewhat similar
    MINIMAL = 0.5 # Minimally similar


class TriggerType(Enum):
    """Types of similarity triggers"""
    DUPLICATE_DETECTED = "duplicate_detected"
    CONSENSUS_FORMED = "consensus_formed"
    CONFLICT_DETECTED = "conflict_detected"
    MARKET_REGIME_CHANGE = "market_regime_change"
    HIGH_CONFIDENCE_CLUSTER = "high_confidence_cluster"
    SIMILARITY_GROUP_CREATED = "similarity_group_created"
    SIGNAL_BOOSTED = "signal_boosted"
    SIGNAL_PENALIZED = "signal_penalized"
    PATTERN_RECOGNIZED = "pattern_recognized"


@dataclass
class SimilarityConfig:
    """Configuration for similarity detection"""
    algorithm: SimilarityAlgorithm = SimilarityAlgorithm.FEATURE_WEIGHTED
    threshold: float = SimilarityThreshold.MEDIUM.value
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'strength': 0.25,
        'confidence': 0.25,
        'rsi': 0.15,
        'macd': 0.15,
        'vol_zscore': 0.10,
        'momentum_5': 0.05,
        'adx': 0.05
    })
    time_window_minutes: int = 30  # Consider signals within this time window
    max_similar_signals: int = 3   # Maximum similar signals to keep
    enable_duplicate_filtering: bool = True
    enable_pattern_recognition: bool = True
    enable_clustering: bool = False
    cluster_threshold: float = 0.7


@dataclass
class SimilarityResult:
    """Result of similarity comparison between two signals"""
    signal_a: TacticalSignal
    signal_b: TacticalSignal
    similarity_score: float
    algorithm_used: SimilarityAlgorithm
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    is_similar: bool = False
    similarity_reason: str = ""


@dataclass
class SimilarityGroup:
    """Group of similar signals"""
    group_id: str
    signals: List[TacticalSignal] = field(default_factory=list)
    centroid_signal: Optional[TacticalSignal] = None
    avg_similarity: float = 0.0
    group_confidence: float = 0.0
    group_strength: float = 0.0
    pattern_type: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TriggerEvent:
    """Event data for similarity triggers"""
    trigger_type: TriggerType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "similarity_detector"
    priority: str = "normal"  # low, normal, high, critical


class SynchronizedTriggerSystem:
    """
    Synchronized trigger system for similarity detector events.

    This system allows multiple components to register for similarity-related events
    and handles them in a thread-safe, synchronized manner.
    """

    def __init__(self):
        self._triggers: Dict[TriggerType, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="similarity-triggers")
        self._event_history: List[TriggerEvent] = []
        self._max_history = 1000
        self.logger = logger

        self.logger.info("🔥 Synchronized Trigger System initialized")

    def register_trigger(self, trigger_type: TriggerType, callback: Callable[[TriggerEvent], None],
                        priority: int = 0) -> None:
        """
        Register a callback for a specific trigger type.

        Args:
            trigger_type: Type of trigger to listen for
            callback: Function to call when trigger fires
            priority: Priority order (higher numbers = higher priority)
        """
        with self._lock:
            # Insert callback in priority order (higher priority first)
            callbacks = self._triggers[trigger_type]
            insert_pos = 0
            for i, (existing_callback, existing_priority) in enumerate(callbacks):
                if priority > existing_priority:
                    insert_pos = i
                    break
                insert_pos = i + 1

            callbacks.insert(insert_pos, (callback, priority))
            self.logger.debug(f"📝 Registered trigger: {trigger_type.value} (priority: {priority})")

    def unregister_trigger(self, trigger_type: TriggerType, callback: Callable) -> None:
        """
        Unregister a callback for a specific trigger type.

        Args:
            trigger_type: Type of trigger to stop listening for
            callback: Function to remove
        """
        with self._lock:
            callbacks = self._triggers[trigger_type]
            self._triggers[trigger_type] = [
                (cb, pri) for cb, pri in callbacks if cb != callback
            ]
            self.logger.debug(f"🗑️ Unregistered trigger: {trigger_type.value}")

    def fire_trigger(self, trigger_type: TriggerType, data: Dict[str, Any] = None,
                    priority: str = "normal", synchronous: bool = False) -> None:
        """
        Fire a trigger event.

        Args:
            trigger_type: Type of trigger to fire
            data: Event data to pass to callbacks
            priority: Event priority level
            synchronous: If True, execute callbacks synchronously
        """
        event = TriggerEvent(
            trigger_type=trigger_type,
            data=data or {},
            priority=priority
        )

        # Store event in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Log the trigger
        self.logger.info(f"🔥 Trigger fired: {trigger_type.value} (priority: {priority})")

        # Get callbacks for this trigger type
        with self._lock:
            callbacks = self._triggers[trigger_type].copy()

        if not callbacks:
            self.logger.debug(f"⚠️ No callbacks registered for trigger: {trigger_type.value}")
            return

        # Execute callbacks
        if synchronous:
            # Execute synchronously (blocking)
            for callback, _ in callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"❌ Error in synchronous trigger callback: {e}")
        else:
            # Execute asynchronously (non-blocking)
            for callback, _ in callbacks:
                self._executor.submit(self._execute_callback_safe, callback, event)

    def _execute_callback_safe(self, callback: Callable, event: TriggerEvent) -> None:
        """Execute callback with error handling"""
        try:
            callback(event)
        except Exception as e:
            self.logger.error(f"❌ Error in asynchronous trigger callback: {e}")

    def get_trigger_history(self, trigger_type: TriggerType = None,
                           limit: int = 100) -> List[TriggerEvent]:
        """
        Get trigger event history.

        Args:
            trigger_type: Filter by trigger type (None for all)
            limit: Maximum number of events to return

        Returns:
            List of recent trigger events
        """
        with self._lock:
            history = self._event_history.copy()

        if trigger_type:
            history = [event for event in history if event.trigger_type == trigger_type]

        return history[-limit:] if limit > 0 else history

    def clear_trigger_history(self) -> None:
        """Clear the trigger event history"""
        with self._lock:
            self._event_history.clear()
            self.logger.debug("🧹 Trigger history cleared")

    def get_registered_triggers(self) -> Dict[str, int]:
        """
        Get information about registered triggers.

        Returns:
            Dictionary mapping trigger types to callback counts
        """
        with self._lock:
            return {
                trigger_type.value: len(callbacks)
                for trigger_type, callbacks in self._triggers.items()
            }

    def shutdown(self) -> None:
        """Shutdown the trigger system"""
        self.logger.info("🔥 Shutting down Synchronized Trigger System")
        self._executor.shutdown(wait=True)


class SignalSimilarityDetector:
    """
    Main class for detecting signal similarities and managing signal groups.

    This class provides methods to:
    - Compare signals for similarity
    - Group similar signals
    - Filter duplicates
    - Recognize patterns
    - Integrate with signal processing pipeline
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.signal_history: List[TacticalSignal] = []
        self.similarity_groups: List[SimilarityGroup] = []
        self.duplicate_cache: Set[str] = set()
        self.trigger_system = SynchronizedTriggerSystem()
        self.logger = logger

        self.logger.info("🔍 Signal Similarity Detector initialized")
        self.logger.info(f"   Algorithm: {self.config.algorithm.value}")
        self.logger.info(f"   Threshold: {self.config.threshold}")
        self.logger.info(f"   Time window: {self.config.time_window_minutes} minutes")

    def calculate_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> SimilarityResult:
        """
        Calculate similarity between two signals using the configured algorithm.

        Args:
            signal_a: First signal to compare
            signal_b: Second signal to compare

        Returns:
            SimilarityResult with score and analysis
        """
        try:
            # Basic validation
            if not self._validate_signals(signal_a, signal_b):
                return SimilarityResult(
                    signal_a=signal_a,
                    signal_b=signal_b,
                    similarity_score=0.0,
                    algorithm_used=self.config.algorithm,
                    is_similar=False,
                    similarity_reason="Invalid signals"
                )

            # Different symbols are not similar
            if signal_a.symbol != signal_b.symbol:
                return SimilarityResult(
                    signal_a=signal_a,
                    signal_b=signal_b,
                    similarity_score=0.0,
                    algorithm_used=self.config.algorithm,
                    is_similar=False,
                    similarity_reason="Different symbols"
                )

            # Calculate similarity based on algorithm
            if self.config.algorithm == SimilarityAlgorithm.COSINE:
                score, contributions = self._cosine_similarity(signal_a, signal_b)
            elif self.config.algorithm == SimilarityAlgorithm.EUCLIDEAN:
                score, contributions = self._euclidean_similarity(signal_a, signal_b)
            elif self.config.algorithm == SimilarityAlgorithm.MANHATTAN:
                score, contributions = self._manhattan_similarity(signal_a, signal_b)
            elif self.config.algorithm == SimilarityAlgorithm.JACCARD:
                score, contributions = self._jaccard_similarity(signal_a, signal_b)
            elif self.config.algorithm == SimilarityAlgorithm.FEATURE_WEIGHTED:
                score, contributions = self._feature_weighted_similarity(signal_a, signal_b)
            else:
                raise ValueError(f"Unsupported similarity algorithm: {self.config.algorithm}")

            # Determine if signals are similar
            is_similar = score >= self.config.threshold

            # Generate reason
            reason = self._generate_similarity_reason(score, is_similar, contributions)

            result = SimilarityResult(
                signal_a=signal_a,
                signal_b=signal_b,
                similarity_score=score,
                algorithm_used=self.config.algorithm,
                feature_contributions=contributions,
                is_similar=is_similar,
                similarity_reason=reason
            )

            self.logger.debug(f"🔍 Similarity calculated: {signal_a.symbol} {signal_a.side} vs {signal_b.side} = {score:.3f} ({'similar' if is_similar else 'different'})")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error calculating similarity: {e}")
            return SimilarityResult(
                signal_a=signal_a,
                signal_b=signal_b,
                similarity_score=0.0,
                algorithm_used=self.config.algorithm,
                is_similar=False,
                similarity_reason=f"Error: {str(e)}"
            )

    def _cosine_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> Tuple[float, Dict[str, float]]:
        """Calculate cosine similarity between signal feature vectors"""
        try:
            # Extract feature vectors
            vec_a = self._extract_feature_vector(signal_a)
            vec_b = self._extract_feature_vector(signal_b)

            # Calculate cosine similarity
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 0.0, {}

            similarity = dot_product / (norm_a * norm_b)

            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))

            return similarity, {"cosine": similarity}

        except Exception as e:
            self.logger.error(f"Error in cosine similarity: {e}")
            return 0.0, {}

    def _euclidean_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> Tuple[float, Dict[str, float]]:
        """Calculate similarity based on Euclidean distance (converted to similarity score)"""
        try:
            vec_a = self._extract_feature_vector(signal_a)
            vec_b = self._extract_feature_vector(signal_b)

            # Calculate Euclidean distance
            distance = np.linalg.norm(vec_a - vec_b)

            # Convert distance to similarity (higher distance = lower similarity)
            # Assuming max reasonable distance is around 10 (based on normalized features)
            max_distance = 10.0
            similarity = max(0.0, 1.0 - (distance / max_distance))

            return similarity, {"euclidean_distance": distance, "euclidean_similarity": similarity}

        except Exception as e:
            self.logger.error(f"Error in euclidean similarity: {e}")
            return 0.0, {}

    def _manhattan_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> Tuple[float, Dict[str, float]]:
        """Calculate similarity based on Manhattan distance"""
        try:
            vec_a = self._extract_feature_vector(signal_a)
            vec_b = self._extract_feature_vector(signal_b)

            # Calculate Manhattan distance
            distance = np.sum(np.abs(vec_a - vec_b))

            # Convert to similarity
            max_distance = 20.0  # Assuming reasonable max Manhattan distance
            similarity = max(0.0, 1.0 - (distance / max_distance))

            return similarity, {"manhattan_distance": distance, "manhattan_similarity": similarity}

        except Exception as e:
            self.logger.error(f"Error in manhattan similarity: {e}")
            return 0.0, {}

    def _jaccard_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> Tuple[float, Dict[str, float]]:
        """Calculate Jaccard similarity based on feature presence"""
        try:
            # Convert features to sets of significant indicators
            set_a = self._extract_feature_set(signal_a)
            set_b = self._extract_feature_set(signal_b)

            if not set_a and not set_b:
                return 1.0, {"jaccard": 1.0}

            intersection = len(set_a.intersection(set_b))
            union = len(set_a.union(set_b))

            if union == 0:
                return 0.0, {"jaccard": 0.0}

            similarity = intersection / union
            return similarity, {"jaccard": similarity, "intersection": intersection, "union": union}

        except Exception as e:
            self.logger.error(f"Error in jaccard similarity: {e}")
            return 0.0, {}

    def _feature_weighted_similarity(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted similarity based on feature importance.

        This is the most sophisticated algorithm that considers:
        - Signal strength and confidence
        - Technical indicators (RSI, MACD, volume, momentum)
        - Direction alignment
        """
        try:
            contributions = {}
            total_weight = 0.0
            weighted_sum = 0.0

            # Direction similarity (perfect match required for high similarity)
            direction_sim = 1.0 if signal_a.side == signal_b.side else 0.0
            if direction_sim == 0.0:
                # Opposite directions get very low similarity
                return 0.1, {"direction_mismatch": 0.0}

            contributions["direction"] = direction_sim

            # Strength similarity
            strength_a = safe_float(signal_a.strength)
            strength_b = safe_float(signal_b.strength)
            strength_sim = 1.0 - abs(strength_a - strength_b)
            contributions["strength"] = strength_sim

            # Confidence similarity
            conf_a = safe_float(signal_a.confidence)
            conf_b = safe_float(signal_b.confidence)
            conf_sim = 1.0 - abs(conf_a - conf_b)
            contributions["confidence"] = conf_sim

            # Technical indicator similarities
            features_a = signal_a.features or {}
            features_b = signal_b.features or {}

            for feature_name, weight in self.config.feature_weights.items():
                if feature_name in ['strength', 'confidence']:
                    continue  # Already handled above

                val_a = safe_float(features_a.get(feature_name, 0.0))
                val_b = safe_float(features_b.get(feature_name, 0.0))

                # Normalize values if needed (assuming they're already in reasonable ranges)
                if feature_name in ['rsi']:
                    # RSI is 0-100, normalize to 0-1
                    val_a = val_a / 100.0
                    val_b = val_b / 100.0
                elif feature_name in ['macd']:
                    # MACD can be normalized by dividing by a typical range
                    val_a = val_a / 100.0
                    val_b = val_b / 100.0

                feature_sim = 1.0 - min(1.0, abs(val_a - val_b))
                contributions[feature_name] = feature_sim

                weighted_sum += feature_sim * weight
                total_weight += weight

            # Add strength and confidence with their weights
            weighted_sum += strength_sim * self.config.feature_weights.get('strength', 0.25)
            total_weight += self.config.feature_weights.get('strength', 0.25)

            weighted_sum += conf_sim * self.config.feature_weights.get('confidence', 0.25)
            total_weight += self.config.feature_weights.get('confidence', 0.25)

            # Calculate final similarity
            if total_weight == 0:
                similarity = 0.5  # Neutral fallback
            else:
                similarity = weighted_sum / total_weight

            # Boost similarity if signals are very close in time and have high confidence
            time_diff = abs((signal_a.timestamp - signal_b.timestamp).total_seconds() / 60.0)  # minutes
            if time_diff < 5 and conf_a > 0.8 and conf_b > 0.8:
                similarity = min(1.0, similarity * 1.2)

            return similarity, contributions

        except Exception as e:
            self.logger.error(f"Error in feature weighted similarity: {e}")
            return 0.0, {}

    def _extract_feature_vector(self, signal: TacticalSignal) -> np.ndarray:
        """Extract numerical feature vector from signal for similarity calculations"""
        try:
            features = []

            # Basic signal features
            features.append(safe_float(signal.strength))
            features.append(safe_float(signal.confidence))

            # Direction encoding (buy=1, sell=-1, hold=0)
            if signal.side.lower() == 'buy':
                features.append(1.0)
            elif signal.side.lower() == 'sell':
                features.append(-1.0)
            else:
                features.append(0.0)

            # Technical indicators
            signal_features = signal.features or {}
            feature_names = ['rsi', 'macd', 'macd_signal', 'vol_zscore', 'momentum_5', 'adx']

            for feature_name in feature_names:
                value = safe_float(signal_features.get(feature_name, 0.0))
                # Normalize some features
                if feature_name == 'rsi':
                    value = value / 100.0  # 0-1 range
                elif feature_name in ['macd', 'macd_signal']:
                    value = value / 50.0  # Normalize by typical range
                features.append(value)

            return np.array(features)

        except Exception as e:
            self.logger.error(f"Error extracting feature vector: {e}")
            return np.array([0.0, 0.0, 0.0])  # Minimal fallback

    def _extract_feature_set(self, signal: TacticalSignal) -> Set[str]:
        """Extract set of significant features for Jaccard similarity"""
        feature_set = set()

        # Add direction
        feature_set.add(f"side_{signal.side}")

        # Add strength/confidence categories
        strength = safe_float(signal.strength)
        if strength > 0.8:
            feature_set.add("high_strength")
        elif strength > 0.6:
            feature_set.add("medium_strength")
        else:
            feature_set.add("low_strength")

        confidence = safe_float(signal.confidence)
        if confidence > 0.8:
            feature_set.add("high_confidence")
        elif confidence > 0.6:
            feature_set.add("medium_confidence")
        else:
            feature_set.add("low_confidence")

        # Add technical indicator signals
        features = signal.features or {}
        if safe_float(features.get('rsi', 50)) < 30:
            feature_set.add("oversold_rsi")
        elif safe_float(features.get('rsi', 50)) > 70:
            feature_set.add("overbought_rsi")

        if safe_float(features.get('macd', 0)) > 0:
            feature_set.add("bullish_macd")
        else:
            feature_set.add("bearish_macd")

        if safe_float(features.get('vol_zscore', 0)) > 1:
            feature_set.add("high_volume")
        elif safe_float(features.get('vol_zscore', 0)) < -1:
            feature_set.add("low_volume")

        return feature_set

    def _validate_signals(self, signal_a: TacticalSignal, signal_b: TacticalSignal) -> bool:
        """Validate that signals are suitable for similarity comparison"""
        if not signal_a or not signal_b:
            return False

        required_attrs = ['symbol', 'side', 'strength', 'confidence']
        for signal in [signal_a, signal_b]:
            for attr in required_attrs:
                if not hasattr(signal, attr):
                    return False

        return True

    def _generate_similarity_reason(self, score: float, is_similar: bool,
                                  contributions: Dict[str, float]) -> str:
        """Generate human-readable reason for similarity score"""
        if not is_similar:
            if score < 0.3:
                return "Very different signals"
            elif score < 0.5:
                return "Somewhat different"
            else:
                return "Borderline similarity"

        # Find top contributing factors
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factors = [f"{k}({v:.2f})" for k, v in sorted_contributions[:3]]

        return f"Similar due to: {', '.join(top_factors)}"

    def find_similar_signals(self, target_signal: TacticalSignal,
                           candidate_signals: List[TacticalSignal]) -> List[SimilarityResult]:
        """
        Find all signals similar to the target signal from a list of candidates.

        Args:
            target_signal: Signal to find similarities for
            candidate_signals: List of signals to compare against

        Returns:
            List of SimilarityResult objects for similar signals
        """
        similar_results = []

        for candidate in candidate_signals:
            if candidate is target_signal:
                continue  # Skip self-comparison

            result = self.calculate_similarity(target_signal, candidate)
            if result.is_similar:
                similar_results.append(result)

        # Sort by similarity score (highest first)
        similar_results.sort(key=lambda x: x.similarity_score, reverse=True)

        return similar_results

    def filter_duplicate_signals(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        """
        Filter out duplicate or highly similar signals.

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list with duplicates removed
        """
        if not self.config.enable_duplicate_filtering:
            return signals

        try:
            filtered_signals = []
            processed_signals = set()

            # Group signals by symbol
            signals_by_symbol = defaultdict(list)
            for signal in signals:
                signals_by_symbol[signal.symbol].append(signal)

            for symbol, symbol_signals in signals_by_symbol.items():
                # Sort by confidence and strength (highest first)
                symbol_signals.sort(key=lambda s: (s.confidence, s.strength), reverse=True)

                kept_signals = []

                for signal in symbol_signals:
                    # Create unique identifier for duplicate detection
                    signal_id = f"{signal.symbol}_{signal.side}_{signal.timestamp.isoformat()}_{signal.confidence:.3f}"

                    if signal_id in self.duplicate_cache:
                        self.logger.debug(f"🚫 Duplicate signal filtered: {signal.symbol} {signal.side}")
                        # Fire duplicate detected trigger
                        self.trigger_system.fire_trigger(
                            TriggerType.DUPLICATE_DETECTED,
                            data={
                                'duplicate_signal': signal,
                                'reason': 'cache_hit',
                                'signal_id': signal_id
                            }
                        )
                        continue

                    # Check similarity with already kept signals
                    is_duplicate = False
                    for kept_signal in kept_signals:
                        similarity = self.calculate_similarity(signal, kept_signal)
                        if similarity.similarity_score >= self.config.threshold:
                            self.logger.debug(f"🚫 Similar signal filtered: {signal.symbol} {signal.side} (similar to existing)")
                            is_duplicate = True
                            # Fire duplicate detected trigger
                            self.trigger_system.fire_trigger(
                                TriggerType.DUPLICATE_DETECTED,
                                data={
                                    'duplicate_signal': signal,
                                    'similar_to': kept_signal,
                                    'similarity_score': similarity.similarity_score,
                                    'reason': 'similarity_threshold'
                                }
                            )
                            break

                    if not is_duplicate:
                        kept_signals.append(signal)
                        self.duplicate_cache.add(signal_id)

                        # Limit number of similar signals per symbol
                        if len(kept_signals) >= self.config.max_similar_signals:
                            break

                filtered_signals.extend(kept_signals)

            self.logger.info(f"🔍 Duplicate filtering: {len(signals)} → {len(filtered_signals)} signals")
            return filtered_signals

        except Exception as e:
            self.logger.error(f"❌ Error in duplicate filtering: {e}")
            return signals  # Return original on error

    def group_similar_signals(self, signals: List[TacticalSignal]) -> List[SimilarityGroup]:
        """
        Group signals into similarity clusters.

        Args:
            signals: List of signals to group

        Returns:
            List of SimilarityGroup objects
        """
        if not self.config.enable_clustering:
            return []

        try:
            groups = []
            processed_indices = set()  # Use indices instead of objects

            for i, signal in enumerate(signals):
                if i in processed_indices:
                    continue

                # Find all similar signals
                similar_results = self.find_similar_signals(signal, signals)
                similar_signals = [result.signal_b for result in similar_results]

                if len(similar_signals) >= 2:  # Need at least 2 similar signals for a group
                    # Create group
                    group_signals = [signal] + similar_signals
                    group_id = f"group_{signal.symbol}_{len(groups)}"

                    # Calculate centroid (average of group features)
                    centroid = self._calculate_group_centroid(group_signals)

                    # Calculate group metrics
                    avg_similarity = np.mean([r.similarity_score for r in similar_results])
                    group_confidence = np.mean([s.confidence for s in group_signals])
                    group_strength = np.mean([s.strength for s in group_signals])

                    # Determine pattern type
                    pattern_type = self._identify_pattern_type(group_signals)

                    group = SimilarityGroup(
                        group_id=group_id,
                        signals=group_signals,
                        centroid_signal=centroid,
                        avg_similarity=avg_similarity,
                        group_confidence=group_confidence,
                        group_strength=group_strength,
                        pattern_type=pattern_type
                    )

                    groups.append(group)

                    # Fire similarity group created trigger
                    self.trigger_system.fire_trigger(
                        TriggerType.SIMILARITY_GROUP_CREATED,
                        data={
                            'group': group,
                            'symbol': signal.symbol,
                            'group_size': len(group_signals),
                            'pattern_type': pattern_type,
                            'avg_confidence': group_confidence
                        }
                    )

                    # Fire specific pattern triggers
                    if pattern_type.endswith('_consensus'):
                        self.trigger_system.fire_trigger(
                            TriggerType.CONSENSUS_FORMED,
                            data={
                                'group': group,
                                'consensus_type': pattern_type,
                                'symbol': signal.symbol,
                                'confidence': group_confidence
                            }
                        )
                    elif pattern_type == 'conflicting_signals':
                        self.trigger_system.fire_trigger(
                            TriggerType.CONFLICT_DETECTED,
                            data={
                                'group': group,
                                'symbol': signal.symbol,
                                'conflicting_signals': group_signals
                            }
                        )

                    # Mark signals as processed
                    for s in group_signals:
                        # Find index of this signal
                        for j, orig_signal in enumerate(signals):
                            if (orig_signal.symbol == s.symbol and
                                orig_signal.side == s.side and
                                orig_signal.timestamp == s.timestamp and
                                abs(orig_signal.confidence - s.confidence) < 0.001):
                                processed_indices.add(j)
                                break

            self.logger.info(f"🔍 Signal clustering: {len(signals)} signals → {len(groups)} groups")
            return groups

        except Exception as e:
            self.logger.error(f"❌ Error in signal grouping: {e}")
            return []

    def _calculate_group_centroid(self, signals: List[TacticalSignal]) -> TacticalSignal:
        """Calculate the centroid signal for a group"""
        try:
            if not signals:
                return None

            # Use the first signal as template
            centroid = signals[0]

            # Average numerical features
            avg_strength = np.mean([s.strength for s in signals])
            avg_confidence = np.mean([s.confidence for s in signals])

            # Most common side (simple majority)
            sides = [s.side for s in signals]
            most_common_side = max(set(sides), key=sides.count)

            # Average features
            all_features = {}
            feature_names = set()
            for signal in signals:
                if signal.features:
                    feature_names.update(signal.features.keys())

            for feature_name in feature_names:
                values = []
                for signal in signals:
                    if signal.features and feature_name in signal.features:
                        values.append(safe_float(signal.features[feature_name]))
                if values:
                    all_features[feature_name] = np.mean(values)

            # Create centroid signal
            centroid_signal = TacticalSignal(
                symbol=centroid.symbol,
                side=most_common_side,
                strength=avg_strength,
                confidence=avg_confidence,
                signal_type="centroid",
                source="similarity_detector",
                features=all_features,
                timestamp=pd.Timestamp.now(),
                metadata={
                    'group_size': len(signals),
                    'centroid_type': 'average',
                    'original_signals': len(signals)
                }
            )

            return centroid_signal

        except Exception as e:
            self.logger.error(f"Error calculating group centroid: {e}")
            return signals[0] if signals else None

    def _identify_pattern_type(self, signals: List[TacticalSignal]) -> str:
        """Identify the type of pattern in a signal group"""
        try:
            if len(signals) < 2:
                return "single"

            # Check for consensus pattern (all signals agree)
            sides = set(s.side for s in signals)
            if len(sides) == 1:
                side = list(sides)[0]
                confidences = [s.confidence for s in signals]
                avg_confidence = np.mean(confidences)

                if avg_confidence > 0.8:
                    return f"strong_{side}_consensus"
                else:
                    return f"weak_{side}_consensus"

            # Check for conflicting signals
            if 'buy' in sides and 'sell' in sides:
                return "conflicting_signals"

            # Check for momentum pattern
            momentum_values = []
            for signal in signals:
                if signal.features and 'momentum_5' in signal.features:
                    momentum_values.append(signal.features['momentum_5'])

            if momentum_values:
                avg_momentum = np.mean(momentum_values)
                if abs(avg_momentum) > 10:
                    return "strong_momentum" if avg_momentum > 0 else "weak_momentum"

            return "mixed_signals"

        except Exception as e:
            self.logger.error(f"Error identifying pattern type: {e}")
            return "unknown"

    def update_signal_history(self, signals: List[TacticalSignal]):
        """Update the signal history for temporal similarity analysis"""
        try:
            # Add new signals
            self.signal_history.extend(signals)

            # Remove old signals outside time window
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.config.time_window_minutes)
            self.signal_history = [
                s for s in self.signal_history
                if isinstance(s.timestamp, (pd.Timestamp, datetime)) and s.timestamp >= cutoff_time
            ]

            # Limit history size
            max_history = 1000
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]

            self.logger.debug(f"📚 Signal history updated: {len(self.signal_history)} signals")

        except Exception as e:
            self.logger.error(f"❌ Error updating signal history: {e}")

    def detect_market_patterns(self, signals: List[TacticalSignal]) -> Dict[str, Any]:
        """
        Detect broader market patterns from signal similarities.

        Args:
            signals: Current signals to analyze

        Returns:
            Dictionary with pattern analysis results
        """
        if not self.config.enable_pattern_recognition:
            return {}

        try:
            patterns = {
                'consensus_signals': 0,
                'conflicting_signals': 0,
                'high_confidence_clusters': 0,
                'market_regime': 'neutral',
                'volatility_pattern': 'normal',
                'momentum_pattern': 'neutral'
            }

            # Analyze signal consensus by symbol
            signals_by_symbol = defaultdict(list)
            for signal in signals:
                signals_by_symbol[signal.symbol].append(signal)

            for symbol, symbol_signals in signals_by_symbol.items():
                if len(symbol_signals) < 2:
                    continue

                # Check for consensus
                sides = set(s.side for s in symbol_signals)
                if len(sides) == 1:
                    patterns['consensus_signals'] += 1
                elif len(sides) > 1:
                    patterns['conflicting_signals'] += 1

                # Check for high confidence clusters
                high_conf_signals = [s for s in symbol_signals if s.confidence > 0.8]
                if len(high_conf_signals) >= 2:
                    patterns['high_confidence_clusters'] += 1

            # Determine market regime
            total_signals = len(signals)
            if total_signals > 0:
                consensus_ratio = patterns['consensus_signals'] / len(signals_by_symbol)
                if consensus_ratio > 0.7:
                    patterns['market_regime'] = 'trending'
                elif consensus_ratio < 0.3:
                    patterns['market_regime'] = 'choppy'

            # Analyze volatility patterns
            vol_zscores = []
            for signal in signals:
                if signal.features and 'vol_zscore' in signal.features:
                    vol_zscores.append(signal.features['vol_zscore'])

            if vol_zscores:
                avg_vol = np.mean(vol_zscores)
                if avg_vol > 1.5:
                    patterns['volatility_pattern'] = 'high'
                elif avg_vol < -1.5:
                    patterns['volatility_pattern'] = 'low'

            # Analyze momentum patterns
            momentum_values = []
            for signal in signals:
                if signal.features and 'momentum_5' in signal.features:
                    momentum_values.append(signal.features['momentum_5'])

            if momentum_values:
                avg_momentum = np.mean(momentum_values)
                if avg_momentum > 5:
                    patterns['momentum_pattern'] = 'bullish'
                elif avg_momentum < -5:
                    patterns['momentum_pattern'] = 'bearish'

            # Fire pattern recognition triggers
            if patterns['high_confidence_clusters'] > 0:
                self.trigger_system.fire_trigger(
                    TriggerType.HIGH_CONFIDENCE_CLUSTER,
                    data={
                        'cluster_count': patterns['high_confidence_clusters'],
                        'total_signals': len(signals),
                        'patterns': patterns
                    }
                )

            # Fire market regime change trigger if significant pattern detected
            if patterns['market_regime'] != 'neutral':
                self.trigger_system.fire_trigger(
                    TriggerType.MARKET_REGIME_CHANGE,
                    data={
                        'regime': patterns['market_regime'],
                        'consensus_ratio': patterns['consensus_signals'] / max(1, len(signals_by_symbol)),
                        'patterns': patterns
                    }
                )

            # Fire general pattern recognized trigger
            self.trigger_system.fire_trigger(
                TriggerType.PATTERN_RECOGNIZED,
                data={
                    'patterns': patterns,
                    'signal_count': len(signals),
                    'symbol_count': len(signals_by_symbol)
                }
            )

            self.logger.info(f"📊 Market patterns detected: regime={patterns['market_regime']}, volatility={patterns['volatility_pattern']}, momentum={patterns['momentum_pattern']}")

            return patterns

        except Exception as e:
            self.logger.error(f"❌ Error detecting market patterns: {e}")
            return {}

    def process_signals(self, signals: List[TacticalSignal]) -> Tuple[List[TacticalSignal], Dict[str, Any]]:
        """
        Main processing method that applies all similarity detection features.

        Args:
            signals: Input signals to process

        Returns:
            Tuple of (filtered_signals, analysis_results)
        """
        try:
            self.logger.info(f"🔍 Processing {len(signals)} signals for similarity analysis")

            # Update signal history
            self.update_signal_history(signals)

            # Filter duplicates
            filtered_signals = self.filter_duplicate_signals(signals)

            # Group similar signals
            similarity_groups = self.group_similar_signals(filtered_signals)

            # Detect market patterns
            market_patterns = self.detect_market_patterns(filtered_signals)

            # Apply similarity-based prioritization
            prioritized_signals = self._prioritize_similar_signals(filtered_signals, similarity_groups)

            analysis_results = {
                'original_count': len(signals),
                'filtered_count': len(filtered_signals),
                'prioritized_count': len(prioritized_signals),
                'similarity_groups': len(similarity_groups),
                'market_patterns': market_patterns,
                'groups_detail': [
                    {
                        'group_id': g.group_id,
                        'size': len(g.signals),
                        'pattern_type': g.pattern_type,
                        'avg_similarity': g.avg_similarity,
                        'group_confidence': g.group_confidence
                    }
                    for g in similarity_groups
                ]
            }

            self.logger.info(f"✅ Similarity processing complete: {len(signals)} → {len(prioritized_signals)} signals")
            self.logger.info(f"   Groups: {len(similarity_groups)}, Patterns: {market_patterns.get('market_regime', 'unknown')}")

            return prioritized_signals, analysis_results

        except Exception as e:
            self.logger.error(f"❌ Error in signal processing: {e}")
            return signals, {'error': str(e)}

    def _prioritize_similar_signals(self, signals: List[TacticalSignal],
                                  groups: List[SimilarityGroup]) -> List[TacticalSignal]:
        """
        Prioritize signals based on similarity analysis.

        Strategy:
        - Keep centroid signals from strong groups
        - Boost confidence of signals in consensus groups
        - Reduce priority of conflicting signals
        """
        try:
            prioritized_signals = signals.copy()

            # Process each group
            for group in groups:
                if len(group.signals) < 2:
                    continue

                # For strong consensus groups, boost the centroid signal
                if group.pattern_type.endswith('_consensus') and group.group_confidence > 0.7:
                    # Add the centroid signal with boosted confidence
                    if group.centroid_signal:
                        boosted_centroid = group.centroid_signal
                        boosted_centroid.confidence = min(1.0, boosted_centroid.confidence * 1.2)
                        boosted_centroid.metadata = boosted_centroid.metadata or {}
                        boosted_centroid.metadata['similarity_boost'] = True
                        boosted_centroid.metadata['group_id'] = group.group_id

                        # Replace individual signals with boosted centroid
                        prioritized_signals = [s for s in prioritized_signals if s not in group.signals]
                        prioritized_signals.append(boosted_centroid)

                        self.logger.info(f"🚀 Boosted centroid signal for {group.group_id}: conf {group.group_confidence:.3f} → {boosted_centroid.confidence:.3f}")

                        # Fire signal boosted trigger
                        self.trigger_system.fire_trigger(
                            TriggerType.SIGNAL_BOOSTED,
                            data={
                                'boosted_signal': boosted_centroid,
                                'original_confidence': group.group_confidence,
                                'boosted_confidence': boosted_centroid.confidence,
                                'group_id': group.group_id,
                                'reason': 'consensus_centroid'
                            }
                        )

                # For conflicting groups, reduce confidence of weaker signals
                elif group.pattern_type == 'conflicting_signals':
                    # Keep only the strongest signal from each side
                    buy_signals = [s for s in group.signals if s.side == 'buy']
                    sell_signals = [s for s in group.signals if s.side == 'sell']

                    if buy_signals and sell_signals:
                        # Find strongest from each side
                        strongest_buy = max(buy_signals, key=lambda s: s.confidence * s.strength)
                        strongest_sell = max(sell_signals, key=lambda s: s.confidence * s.strength)

                        # Reduce confidence of weaker signals
                        for signal in group.signals:
                            if signal not in [strongest_buy, strongest_sell]:
                                original_confidence = signal.confidence
                                signal.confidence *= 0.8  # Reduce by 20%
                                signal.metadata = signal.metadata or {}
                                signal.metadata['conflicting_penalty'] = True

                                # Fire signal penalized trigger
                                self.trigger_system.fire_trigger(
                                    TriggerType.SIGNAL_PENALIZED,
                                    data={
                                        'penalized_signal': signal,
                                        'original_confidence': original_confidence,
                                        'penalized_confidence': signal.confidence,
                                        'group_id': group.group_id,
                                        'reason': 'conflicting_signals'
                                    }
                                )

            return prioritized_signals

        except Exception as e:
            self.logger.error(f"❌ Error in signal prioritization: {e}")
            return signals
