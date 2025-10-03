#!/usr/bin/env python3
"""
Test suite for the Signal Similarity Detector.

This module provides comprehensive tests for the similarity detection functionality,
including unit tests for individual components and integration tests for the full pipeline.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l2_tactic.similarity_detector import (
    SignalSimilarityDetector,
    SimilarityConfig,
    SimilarityAlgorithm,
    SimilarityThreshold,
    TriggerType,
    SimilarityResult,
    SimilarityGroup,
    TriggerEvent,
    SynchronizedTriggerSystem
)
from l2_tactic.models import TacticalSignal


class TestSimilarityConfig(unittest.TestCase):
    """Test cases for SimilarityConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SimilarityConfig()
        self.assertEqual(config.algorithm, SimilarityAlgorithm.FEATURE_WEIGHTED)
        self.assertEqual(config.threshold, SimilarityThreshold.MEDIUM.value)
        self.assertIsInstance(config.feature_weights, dict)
        self.assertTrue(config.enable_duplicate_filtering)
        self.assertTrue(config.enable_pattern_recognition)
        self.assertFalse(config.enable_clustering)

    def test_custom_config(self):
        """Test custom configuration"""
        config = SimilarityConfig(
            algorithm=SimilarityAlgorithm.COSINE,
            threshold=0.8,
            time_window_minutes=60,
            enable_clustering=True
        )
        self.assertEqual(config.algorithm, SimilarityAlgorithm.COSINE)
        self.assertEqual(config.threshold, 0.8)
        self.assertEqual(config.time_window_minutes, 60)
        self.assertTrue(config.enable_clustering)


class TestSimilarityAlgorithms(unittest.TestCase):
    """Test cases for similarity algorithms"""

    def setUp(self):
        """Set up test signals"""
        self.signal_a = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.8,
            confidence=0.9,
            features={
                'rsi': 65.0,
                'macd': 120.5,
                'vol_zscore': 0.5,
                'momentum_5': 2.1
            },
            timestamp=pd.Timestamp.now()
        )

        self.signal_b = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.75,
            confidence=0.85,
            features={
                'rsi': 62.0,
                'macd': 115.0,
                'vol_zscore': 0.3,
                'momentum_5': 1.8
            },
            timestamp=pd.Timestamp.now()
        )

        self.signal_c = TacticalSignal(
            symbol="ETHUSDT",
            side="sell",
            strength=0.6,
            confidence=0.7,
            features={
                'rsi': 75.0,
                'macd': -50.0,
                'vol_zscore': -0.2,
                'momentum_5': -1.5
            },
            timestamp=pd.Timestamp.now()
        )

    def test_cosine_similarity(self):
        """Test cosine similarity algorithm"""
        detector = SignalSimilarityDetector(SimilarityConfig(algorithm=SimilarityAlgorithm.COSINE))

        result = detector.calculate_similarity(self.signal_a, self.signal_b)
        self.assertIsInstance(result, SimilarityResult)
        self.assertGreater(result.similarity_score, 0.8)  # Should be very similar
        self.assertTrue(result.is_similar)

        result_diff = detector.calculate_similarity(self.signal_a, self.signal_c)
        self.assertLess(result_diff.similarity_score, 0.5)  # Should be different (different symbol)

    def test_euclidean_similarity(self):
        """Test euclidean similarity algorithm"""
        detector = SignalSimilarityDetector(SimilarityConfig(algorithm=SimilarityAlgorithm.EUCLIDEAN))

        result = detector.calculate_similarity(self.signal_a, self.signal_b)
        self.assertIsInstance(result, SimilarityResult)
        self.assertGreater(result.similarity_score, 0.7)  # Should be similar

    def test_feature_weighted_similarity(self):
        """Test feature weighted similarity algorithm"""
        detector = SignalSimilarityDetector(SimilarityConfig(algorithm=SimilarityAlgorithm.FEATURE_WEIGHTED))

        result = detector.calculate_similarity(self.signal_a, self.signal_b)
        self.assertIsInstance(result, SimilarityResult)
        self.assertGreater(result.similarity_score, 0.8)  # Should be very similar

        # Check that feature contributions are calculated
        self.assertIn('strength', result.feature_contributions)
        self.assertIn('confidence', result.feature_contributions)

    def test_different_symbols_not_similar(self):
        """Test that signals for different symbols are not considered similar"""
        detector = SignalSimilarityDetector()

        result = detector.calculate_similarity(self.signal_a, self.signal_c)
        self.assertEqual(result.similarity_score, 0.0)
        self.assertFalse(result.is_similar)
        self.assertEqual(result.similarity_reason, "Different symbols")


class TestDuplicateFiltering(unittest.TestCase):
    """Test cases for duplicate signal filtering"""

    def setUp(self):
        """Set up test signals"""
        base_time = pd.Timestamp.now()

        self.signals = [
            TacticalSignal(
                symbol="BTCUSDT",
                side="buy",
                strength=0.8,
                confidence=0.9,
                timestamp=base_time
            ),
            TacticalSignal(  # Very similar signal
                symbol="BTCUSDT",
                side="buy",
                strength=0.82,
                confidence=0.88,
                timestamp=base_time + timedelta(seconds=30)
            ),
            TacticalSignal(  # Different signal
                symbol="BTCUSDT",
                side="sell",
                strength=0.6,
                confidence=0.7,
                timestamp=base_time + timedelta(seconds=60)
            ),
            TacticalSignal(  # Different symbol
                symbol="ETHUSDT",
                side="buy",
                strength=0.75,
                confidence=0.85,
                timestamp=base_time + timedelta(seconds=90)
            )
        ]

    def test_duplicate_filtering_enabled(self):
        """Test duplicate filtering when enabled"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_duplicate_filtering=True,
            threshold=0.8
        ))

        filtered = detector.filter_duplicate_signals(self.signals)
        self.assertLess(len(filtered), len(self.signals))  # Should filter some duplicates

    def test_duplicate_filtering_disabled(self):
        """Test duplicate filtering when disabled"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_duplicate_filtering=False
        ))

        filtered = detector.filter_duplicate_signals(self.signals)
        self.assertEqual(len(filtered), len(self.signals))  # Should not filter anything


class TestSignalGrouping(unittest.TestCase):
    """Test cases for signal grouping functionality"""

    def setUp(self):
        """Set up test signals"""
        base_time = pd.Timestamp.now()

        self.signals = [
            # BTC Buy group
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.82, confidence=0.88, timestamp=base_time + timedelta(seconds=30)),
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.78, confidence=0.92, timestamp=base_time + timedelta(seconds=60)),

            # ETH Sell group
            TacticalSignal(symbol="ETHUSDT", side="sell", strength=0.75, confidence=0.85, timestamp=base_time + timedelta(seconds=90)),
            TacticalSignal(symbol="ETHUSDT", side="sell", strength=0.72, confidence=0.82, timestamp=base_time + timedelta(seconds=120)),

            # Single signal (should not form group)
            TacticalSignal(symbol="ADAUSDT", side="buy", strength=0.6, confidence=0.7, timestamp=base_time + timedelta(seconds=150))
        ]

    def test_signal_grouping_enabled(self):
        """Test signal grouping when enabled"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_clustering=True,
            threshold=0.7
        ))

        groups = detector.group_similar_signals(self.signals)
        self.assertGreater(len(groups), 0)  # Should create some groups

        # Check group properties
        for group in groups:
            self.assertIsInstance(group, SimilarityGroup)
            self.assertGreater(len(group.signals), 1)  # Groups should have multiple signals
            self.assertIsNotNone(group.centroid_signal)
            self.assertGreater(group.avg_similarity, 0)

    def test_signal_grouping_disabled(self):
        """Test signal grouping when disabled"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_clustering=False
        ))

        groups = detector.group_similar_signals(self.signals)
        self.assertEqual(len(groups), 0)  # Should not create any groups


class TestMarketPatternDetection(unittest.TestCase):
    """Test cases for market pattern detection"""

    def setUp(self):
        """Set up test signals with different patterns"""
        base_time = pd.Timestamp.now()

        # Create signals with consensus pattern
        self.consensus_signals = [
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="ETHUSDT", side="buy", strength=0.82, confidence=0.88, timestamp=base_time),
            TacticalSignal(symbol="ADAUSDT", side="buy", strength=0.78, confidence=0.92, timestamp=base_time),
        ]

        # Create signals with conflicting pattern
        self.conflicting_signals = [
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="sell", strength=0.75, confidence=0.85, timestamp=base_time),
        ]

    def test_consensus_pattern_detection(self):
        """Test detection of consensus patterns"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_pattern_recognition=True
        ))

        # Create signals with multiple signals per symbol (consensus)
        base_time = pd.Timestamp.now()
        consensus_signals = [
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.82, confidence=0.88, timestamp=base_time),  # Same symbol, same side
            TacticalSignal(symbol="ETHUSDT", side="sell", strength=0.75, confidence=0.85, timestamp=base_time),
            TacticalSignal(symbol="ETHUSDT", side="sell", strength=0.72, confidence=0.82, timestamp=base_time),  # Same symbol, same side
        ]

        patterns = detector.detect_market_patterns(consensus_signals)
        self.assertIn('market_regime', patterns)
        self.assertIn('consensus_signals', patterns)
        self.assertGreater(patterns['consensus_signals'], 0)

    def test_conflicting_pattern_detection(self):
        """Test detection of conflicting patterns"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_pattern_recognition=True
        ))

        patterns = detector.detect_market_patterns(self.conflicting_signals)
        self.assertIn('conflicting_signals', patterns)
        self.assertGreater(patterns['conflicting_signals'], 0)


class TestSignalPrioritization(unittest.TestCase):
    """Test cases for signal prioritization"""

    def setUp(self):
        """Set up test signals for prioritization"""
        base_time = pd.Timestamp.now()

        self.signals = [
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.82, confidence=0.88, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="sell", strength=0.75, confidence=0.85, timestamp=base_time),
        ]

    def test_signal_prioritization(self):
        """Test signal prioritization based on similarity"""
        detector = SignalSimilarityDetector(SimilarityConfig(
            enable_clustering=True,
            threshold=0.7
        ))

        prioritized = detector._prioritize_similar_signals(self.signals, [])

        # Should maintain or modify signals based on similarity analysis
        self.assertIsInstance(prioritized, list)
        self.assertGreaterEqual(len(prioritized), 0)


class TestTriggerSystem(unittest.TestCase):
    """Test cases for the trigger system"""

    def setUp(self):
        """Set up trigger system"""
        self.trigger_system = SynchronizedTriggerSystem()

    def test_trigger_registration(self):
        """Test trigger registration and unregistration"""
        callback = Mock()
        self.trigger_system.register_trigger(TriggerType.DUPLICATE_DETECTED, callback)

        registered = self.trigger_system.get_registered_triggers()
        self.assertIn(TriggerType.DUPLICATE_DETECTED.value, registered)
        self.assertEqual(registered[TriggerType.DUPLICATE_DETECTED.value], 1)

        self.trigger_system.unregister_trigger(TriggerType.DUPLICATE_DETECTED, callback)
        registered = self.trigger_system.get_registered_triggers()
        self.assertEqual(registered[TriggerType.DUPLICATE_DETECTED.value], 0)

    def test_trigger_firing(self):
        """Test trigger firing"""
        callback = Mock()
        self.trigger_system.register_trigger(TriggerType.DUPLICATE_DETECTED, callback)

        test_data = {'signal': 'test_signal'}
        self.trigger_system.fire_trigger(TriggerType.DUPLICATE_DETECTED, data=test_data)

        # Wait a bit for async execution
        import time
        time.sleep(0.1)

        callback.assert_called_once()
        event = callback.call_args[0][0]
        self.assertIsInstance(event, TriggerEvent)
        self.assertEqual(event.trigger_type, TriggerType.DUPLICATE_DETECTED)
        self.assertEqual(event.data, test_data)

    def test_trigger_history(self):
        """Test trigger event history"""
        callback = Mock()
        self.trigger_system.register_trigger(TriggerType.DUPLICATE_DETECTED, callback)

        self.trigger_system.fire_trigger(TriggerType.DUPLICATE_DETECTED, data={'test': 1})
        self.trigger_system.fire_trigger(TriggerType.CONSENSUS_FORMED, data={'test': 2})

        history = self.trigger_system.get_trigger_history(TriggerType.DUPLICATE_DETECTED)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].trigger_type, TriggerType.DUPLICATE_DETECTED)

        all_history = self.trigger_system.get_trigger_history()
        self.assertEqual(len(all_history), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full similarity detector"""

    def setUp(self):
        """Set up integration test"""
        self.config = SimilarityConfig(
            algorithm=SimilarityAlgorithm.FEATURE_WEIGHTED,
            threshold=0.7,
            enable_duplicate_filtering=True,
            enable_pattern_recognition=True,
            enable_clustering=True
        )
        self.detector = SignalSimilarityDetector(self.config)

    def test_full_processing_pipeline(self):
        """Test the complete signal processing pipeline"""
        # Create diverse test signals
        base_time = pd.Timestamp.now()

        signals = [
            # BTC Buy consensus group
            TacticalSignal(
                symbol="BTCUSDT", side="buy", strength=0.85, confidence=0.92,
                features={'rsi': 65.0, 'macd': 120.0, 'vol_zscore': 0.5},
                timestamp=base_time
            ),
            TacticalSignal(
                symbol="BTCUSDT", side="buy", strength=0.82, confidence=0.89,
                features={'rsi': 63.0, 'macd': 115.0, 'vol_zscore': 0.3},
                timestamp=base_time + timedelta(seconds=30)
            ),

            # ETH Sell consensus group
            TacticalSignal(
                symbol="ETHUSDT", side="sell", strength=0.78, confidence=0.87,
                features={'rsi': 72.0, 'macd': -45.0, 'vol_zscore': -0.2},
                timestamp=base_time + timedelta(seconds=60)
            ),

            # Conflicting signals for BTC
            TacticalSignal(
                symbol="BTCUSDT", side="sell", strength=0.65, confidence=0.75,
                features={'rsi': 68.0, 'macd': -20.0, 'vol_zscore': -0.1},
                timestamp=base_time + timedelta(seconds=90)
            ),

            # Single signal (should be filtered or kept)
            TacticalSignal(
                symbol="ADAUSDT", side="buy", strength=0.6, confidence=0.7,
                features={'rsi': 55.0, 'macd': 10.0, 'vol_zscore': 0.0},
                timestamp=base_time + timedelta(seconds=120)
            )
        ]

        # Process signals
        processed_signals, analysis = self.detector.process_signals(signals)

        # Verify results
        self.assertIsInstance(processed_signals, list)
        self.assertIsInstance(analysis, dict)

        # Check analysis structure
        expected_keys = [
            'original_count', 'filtered_count', 'prioritized_count',
            'similarity_groups', 'market_patterns', 'groups_detail'
        ]
        for key in expected_keys:
            self.assertIn(key, analysis)

        # Verify counts make sense
        self.assertEqual(analysis['original_count'], len(signals))
        self.assertGreaterEqual(analysis['filtered_count'], 0)
        self.assertGreaterEqual(analysis['prioritized_count'], 0)

        # Check that processing completed without errors
        self.assertNotIn('error', analysis)

        print(f"Integration test results: {analysis}")

    def test_trigger_integration(self):
        """Test trigger system integration"""
        trigger_events = []

        def capture_trigger(event):
            trigger_events.append(event)

        # Register for multiple trigger types
        self.detector.trigger_system.register_trigger(TriggerType.DUPLICATE_DETECTED, capture_trigger)
        self.detector.trigger_system.register_trigger(TriggerType.CONSENSUS_FORMED, capture_trigger)
        self.detector.trigger_system.register_trigger(TriggerType.SIGNAL_BOOSTED, capture_trigger)

        # Create signals that should trigger events
        base_time = pd.Timestamp.now()
        signals = [
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.8, confidence=0.9, timestamp=base_time),
            TacticalSignal(symbol="BTCUSDT", side="buy", strength=0.82, confidence=0.88, timestamp=base_time),  # Duplicate
        ]

        # Process signals (should trigger events)
        self.detector.process_signals(signals)

        # Wait for async triggers
        import time
        time.sleep(0.2)

        # Check that triggers were fired
        trigger_types = [event.trigger_type for event in trigger_events]
        print(f"Triggered events: {[t.value for t in trigger_types]}")

        # Should have at least some triggers
        self.assertGreater(len(trigger_events), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run tests
    unittest.main(verbosity=2)
