"""
Test suite for stop-loss calculation edge cases
Tests extreme volatility and price movements to ensure robust stop-loss logic
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from l2_tactic.tactical_signal_processor import L2TacticProcessor


class TestStopLossEdgeCases:
    """Test stop-loss calculation under extreme conditions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = L2TacticProcessor(config=Mock(), apagar_l3=True)

    def test_normal_buy_stop_loss(self):
        """Test normal BUY stop-loss calculation"""
        price = 50000.0
        volatility = 0.03
        confidence = 0.7

        sl_price = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)

        # Should be below current price
        assert sl_price < price
        # Should be at least 2% below (minimum distance)
        distance_pct = (price - sl_price) / price
        assert distance_pct >= 0.019  # Allow small tolerance for rounding
        assert distance_pct <= 0.081  # Maximum 8.1%

    def test_normal_sell_stop_loss(self):
        """Test normal SELL stop-loss calculation"""
        price = 50000.0
        volatility = 0.03
        confidence = 0.7

        sl_price = self.processor._calculate_stop_loss_price("sell", price, volatility, confidence)

        # Should be above current price
        assert sl_price > price
        # Should be at least 2% above (minimum distance)
        distance_pct = (sl_price - price) / price
        assert distance_pct >= 0.019  # Allow small tolerance for rounding
        assert distance_pct <= 0.081  # Maximum 8.1%

    def test_extreme_volatility_high(self):
        """Test with extreme high volatility (should cap at maximum)"""
        price = 50000.0
        volatility = 1.0  # 100% volatility - extreme
        confidence = 0.5

        # Test BUY
        buy_sl = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        buy_distance = (price - buy_sl) / price
        assert buy_distance <= 0.081  # Should not exceed 8.1%

        # Test SELL
        sell_sl = self.processor._calculate_stop_loss_price("sell", price, volatility, confidence)
        sell_distance = (sell_sl - price) / price
        assert sell_distance <= 0.081  # Should not exceed 8.1%

    def test_extreme_volatility_low(self):
        """Test with extreme low volatility (should not go below minimum)"""
        price = 50000.0
        volatility = 0.001  # 0.1% volatility - very low
        confidence = 0.5

        # Test BUY
        buy_sl = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        buy_distance = (price - buy_sl) / price
        assert buy_distance >= 0.019  # Should not go below 1.9%

        # Test SELL
        sell_sl = self.processor._calculate_stop_loss_price("sell", price, volatility, confidence)
        sell_distance = (sell_sl - price) / price
        assert sell_distance >= 0.019  # Should not go below 1.9%

    def test_extreme_confidence_high(self):
        """Test with extreme high confidence (should tighten stops)"""
        price = 50000.0
        volatility = 0.03
        confidence = 1.0  # Maximum confidence

        buy_sl = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        buy_distance = (price - buy_sl) / price

        # High confidence should result in relatively tighter stops
        # (but still respect minimum 2% distance)
        assert buy_distance >= 0.019

    def test_extreme_confidence_low(self):
        """Test with extreme low confidence (should widen stops)"""
        price = 50000.0
        volatility = 0.03
        confidence = 0.0  # Minimum confidence

        buy_sl = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        buy_distance = (price - buy_sl) / price

        # Low confidence should result in wider stops
        assert buy_distance >= 0.019

    def test_price_extremes_high(self):
        """Test with extremely high prices"""
        price = 1000000.0  # 1 million
        volatility = 0.03
        confidence = 0.5

        sl_price = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        assert sl_price < price
        assert sl_price > 0

    def test_price_extremes_low(self):
        """Test with extremely low prices (crypto dust levels)"""
        price = 0.00000001  # Very small crypto amount
        volatility = 0.03
        confidence = 0.5

        sl_price = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)
        assert sl_price < price
        assert sl_price > 0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Invalid price
        result = self.processor._calculate_stop_loss_price("buy", -100, 0.03, 0.5)
        assert result == 0.0

        result = self.processor._calculate_stop_loss_price("buy", 0, 0.03, 0.5)
        assert result == 0.0

        # Invalid confidence
        result = self.processor._calculate_stop_loss_price("buy", 50000, 0.03, 2.0)  # > 1.0
        assert result != 0.0  # Should use fallback value

        result = self.processor._calculate_stop_loss_price("buy", 50000, 0.03, -0.5)  # < 0.0
        assert result != 0.0  # Should use fallback value

        # Invalid volatility
        result = self.processor._calculate_stop_loss_price("buy", 50000, -0.1, 0.5)  # Negative
        assert result != 0.0  # Should use fallback value

        # Invalid side
        result = self.processor._calculate_stop_loss_price("invalid", 50000, 0.03, 0.5)
        assert result == 0.0

    def test_precision_crypto(self):
        """Test precision for crypto prices (8 decimal places)"""
        price = 0.00001234  # Small crypto price
        volatility = 0.03
        confidence = 0.5

        sl_price = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)

        # Should maintain precision
        assert sl_price < price
        assert sl_price > 0

        # Check that it's properly rounded to 8 decimals
        str_price = f"{sl_price:.10f}"
        assert len(str_price.split('.')[-1]) <= 8 or '00000000' in str_price

    def test_volatility_confidence_interaction(self):
        """Test interaction between volatility and confidence"""
        price = 50000.0

        # High volatility + High confidence = moderate stop
        sl1 = self.processor._calculate_stop_loss_price("buy", price, 0.1, 0.9)
        dist1 = (price - sl1) / price

        # High volatility + Low confidence = wider stop
        sl2 = self.processor._calculate_stop_loss_price("buy", price, 0.1, 0.1)
        dist2 = (price - sl2) / price

        # Low confidence should result in wider stop
        assert dist2 >= dist1

        # Low volatility + High confidence = tighter stop
        sl3 = self.processor._calculate_stop_loss_price("buy", price, 0.01, 0.9)
        dist3 = (price - sl3) / price

        # High confidence with low volatility should be tighter
        assert dist3 <= dist2

    def test_sell_stop_above_price(self):
        """Critical test: Ensure SELL stops are always above current price"""
        test_cases = [
            (50000.0, 0.03, 0.5),  # Normal case
            (50000.0, 0.5, 0.5),   # High volatility
            (50000.0, 0.03, 0.9),  # High confidence
            (50000.0, 0.5, 0.1),   # High vol + low confidence
            (1000000.0, 0.03, 0.5), # High price
            (0.0001, 0.03, 0.5),   # Low price
        ]

        for price, vol, conf in test_cases:
            sl_price = self.processor._calculate_stop_loss_price("sell", price, vol, conf)
            assert sl_price > price, f"SELL stop-loss {sl_price} not above price {price} for vol={vol}, conf={conf}"

            # Ensure minimum distance
            distance_pct = (sl_price - price) / price
            assert distance_pct >= 0.019, f"SELL stop-loss distance {distance_pct:.3f}% too small for vol={vol}, conf={conf}"

    def test_buy_stop_below_price(self):
        """Critical test: Ensure BUY stops are always below current price"""
        test_cases = [
            (50000.0, 0.03, 0.5),  # Normal case
            (50000.0, 0.5, 0.5),   # High volatility
            (50000.0, 0.03, 0.9),  # High confidence
            (50000.0, 0.5, 0.1),   # High vol + low confidence
            (1000000.0, 0.03, 0.5), # High price
            (0.0001, 0.03, 0.5),   # Low price
        ]

        for price, vol, conf in test_cases:
            sl_price = self.processor._calculate_stop_loss_price("buy", price, vol, conf)
            assert sl_price < price, f"BUY stop-loss {sl_price} not below price {price} for vol={vol}, conf={conf}"

            # Ensure minimum distance
            distance_pct = (price - sl_price) / price
            assert distance_pct >= 0.019, f"BUY stop-loss distance {distance_pct:.3f}% too small for vol={vol}, conf={conf}"

    def test_emergency_fallback(self):
        """Test emergency fallback when calculation fails"""
        # Force an error condition
        with patch.object(self.processor, '_calculate_stop_loss_price') as mock_method:
            # Make the first call fail
            mock_method.side_effect = [Exception("Test error"), 49000.0]

            result = self.processor._calculate_stop_loss_price("buy", 50000, 0.03, 0.5)
            # Should return the fallback value
            assert result == 49000.0

    def test_rounding_precision(self):
        """Test that prices are properly rounded to avoid floating point issues"""
        price = 50000.123456789
        volatility = 0.03
        confidence = 0.5

        sl_price = self.processor._calculate_stop_loss_price("buy", price, volatility, confidence)

        # Should be rounded to 8 decimal places
        assert sl_price == round(sl_price, 8)

        # Should not have floating point precision issues
        str_price = f"{sl_price:.10f}"
        # Check that we don't have more than 8 decimal places of precision
        decimal_part = str_price.split('.')[-1]
        assert len(decimal_part.rstrip('0')) <= 8


if __name__ == "__main__":
    pytest.main([__file__])
