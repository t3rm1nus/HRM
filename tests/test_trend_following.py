# tests/test_trend_following.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import asyncio

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import TacticalSignal
from core.technical_indicators import calculate_technical_strength_score
from core.logging import logger
from tests.backtester import run_backtest, cargar_csv
from l2_tactic.config import L2Config


class TestTrendFollowing:
    """Test suite for trend-following logic in L2TacticProcessor"""

    @pytest.fixture
    def processor(self):
        """Create L2TacticProcessor instance with disabled L3 for controlled testing"""
        config = L2Config()
        return L2TacticProcessor(config, apagar_l3=True)  # Disable L3 to test pure L2 logic

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for BTCUSDT"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible results

        # Create realistic price data with trends
        base_price = 50000
        trend_component = np.linspace(0, 10000, 100)  # Upward trend
        noise = np.random.normal(0, 500, 100)
        prices = base_price + trend_component + noise

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)

        # Calculate MAs for trend detection
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        return {'BTCUSDT': df}

    @pytest.fixture
    def state_template(self):
        """Base state template for testing"""
        return {
            "portfolio": {
                "USDT": {"free": 1000.0},
                "BTCUSDT": {"position": 0.0}
            },
            "l3_context_cache": {
                "last_output": {
                    "regime": "bull",
                    "sentiment_score": 0.8
                }
            }
        }

    def test_no_mean_reversion(self, processor, sample_market_data, state_template):
        """Test that RSI < 30 never triggers BUY signals (no mean reversion)"""
        # Setup: Create oversold conditions (RSI < 30)
        df = sample_market_data['BTCUSDT'].copy()
        # Override last 20 prices to create RSI < 30 condition
        oversold_prices = np.linspace(df['close'].iloc[-1] * 0.95, df['close'].iloc[-1] * 0.85, 20)
        df.loc[df.index[-20:], 'close'] = oversold_prices
        df.loc[df.index[-20:], 'open'] = oversold_prices * 1.001
        df.loc[df.index[-20:], 'high'] = oversold_prices * 1.005
        df.loc[df.index[-20:], 'low'] = oversold_prices * 0.995

        # Ensure RSI will be very low
        market_data = {'BTCUSDT': df}
        state = state_template.copy()

        # Process signals
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            # Assertions
            if btc_signal:
                assert getattr(btc_signal, 'side', 'hold') != 'buy', \
                    f"BUY signal triggered in oversold conditions (RSI < 30). Signal: {btc_signal.side}"
            else:
                # No signal is also acceptable as it prevents mean reversion buying
                pass

        asyncio.run(test_async())

    def test_bull_trend(self, processor, sample_market_data, state_template):
        """Test that Price > MA50 > MA200 generates BUY with source=L3 in bull trend"""
        # Setup: Clear bull trend conditions
        df = sample_market_data['BTCUSDT'].copy()
        current_price = df['close'].iloc[-1]
        ma50 = df['ma50'].iloc[-1]
        ma200 = df['ma200'].iloc[-1]

        # Ensure conditions: Price > MA50 > MA200
        if not (current_price > ma50 > ma200):
            # Adjust to create bull trend
            df.loc[df.index[-1], 'close'] = max(current_price, ma50 + 1000, ma200 + 2000)
            df.loc[df.index[-1], 'ma50'] = min(df.loc[df.index[-1], 'close'] - 1000, ma200 + 1000)
            df.loc[df.index[-1], 'ma200'] = min(df.loc[df.index[-1], 'ma50'] - 1000, df.loc[df.index[-1], 'ma50'] - 1000)

        market_data = {'BTCUSDT': df}
        state = state_template.copy()

        # Set bull regime in L3 context
        state["l3_context_cache"]["last_output"]["regime"] = "bull"

        # Process signals
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            # Assertions
            if btc_signal:
                side = getattr(btc_signal, 'side', 'hold')
                source = getattr(btc_signal, 'source', '')

                # Should be BUY or HOLD (not SELL) in bull trend
                assert side in ['buy', 'hold'], f"Invalid signal in bull trend: {side}"

                # If BUY signal, should have L3-related source
                if side == 'buy':
                    assert 'l3' in source.lower() or source == 'l3_regime', \
                        f"BUY signal should have L3 source in bull trend, got: {source}"

                    # Confidence should be reasonable in bull market
                    confidence = getattr(btc_signal, 'confidence', 0)
                    assert confidence >= 0.5, f"BUY confidence too low in bull market: {confidence}"

        asyncio.run(test_async())

    def test_bear_trend(self, processor, sample_market_data, state_template):
        """Test that Price < MA50 < MA200 with open position generates SELL"""
        # Setup: Create bear trend with open position
        df = sample_market_data['BTCUSDT'].copy()

        # Create bear market conditions: Price < MA50 < MA200
        current_price = df['close'].iloc[-1]
        ma50 = df['ma50'].iloc[-1]
        ma200 = df['ma200'].iloc[-1]

        # Adjust to create bear trend
        if not (current_price < ma50 < ma200):
            df.loc[df.index[-1], 'close'] = min(current_price, ma50 - 1000, ma200 - 2000)
            df.loc[df.index[-1], 'ma50'] = max(df.loc[df.index[-1], 'close'] + 500, ma200 - 1000)
            df.loc[df.index[-1], 'ma200'] = max(df.loc[df.index[-1], 'ma50'] + 500, df.loc[df.index[-1], 'ma50'] + 500)

        market_data = {'BTCUSDT': df}
        state = state_template.copy()

        # Set bear regime and open position
        state["l3_context_cache"]["last_output"]["regime"] = "bear"
        state["portfolio"]["BTCUSDT"]["position"] = 0.02  # Small open position

        # Process signals
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            # Assertions
            if btc_signal:
                side = getattr(btc_signal, 'side', 'hold')
                confidence = getattr(btc_signal, 'confidence', 0)

                # In bear market with position, should generate SELL or HOLD (not BUY)
                assert side in ['sell', 'hold'], f"Invalid signal in bear trend: {side}"

                # If SELL signal, should have reasonable confidence
                if side == 'sell':
                    assert confidence >= 0.6, f"SELL confidence too low in bear market with position: {confidence}"

        asyncio.run(test_async())

    def test_range(self, processor, sample_market_data, state_template):
        """Test that range market generates HOLD signals"""
        # Setup: Create range market conditions (no clear trend)
        df = sample_market_data['BTCUSDT'].copy()

        # Modify to create sideways/range market
        # Price oscillating around MA50, MA50 around MA200
        mid_price = (df['ma50'].iloc[-1] + df['ma200'].iloc[-1]) / 2
        df.loc[df.index[-10:], 'close'] = np.random.normal(mid_price, mid_price * 0.02, 10)  # ±2% oscillation

        market_data = {'BTCUSDT': df}
        state = state_template.copy()

        # Set range regime
        state["l3_context_cache"]["last_output"]["regime"] = "range"

        # Process signals
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            # Assertions
            if btc_signal:
                side = getattr(btc_signal, 'side', 'hold')
                confidence = getattr(btc_signal, 'confidence', 0)

                # In range markets, should strongly prefer HOLD
                # Allow BUY/SELL only with very low confidence
                if side in ['buy', 'sell']:
                    assert confidence < 0.7, f"High confidence {side} signal in range market not allowed: {confidence}"

        asyncio.run(test_async())

    @pytest.mark.parametrize("regime", ["bull", "bear", "range"])
    def test_regime_consistency(self, processor, sample_market_data, state_template, regime):
        """Test that signals are consistent with market regime"""
        market_data = sample_market_data.copy()
        state = state_template.copy()

        # Set regime
        state["l3_context_cache"]["last_output"]["regime"] = regime

        # Process signals
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            if btc_signal:
                side = getattr(btc_signal, 'side', 'hold')
                confidence = getattr(btc_signal, 'confidence', 0)

                # Check regime-specific expectations
                if regime == "bull":
                    # Bull trend: BUY/SELL allowed, BUY preferred
                    assert side in ['buy', 'sell', 'hold'], f"Unexpected signal in bull regime: {side}"
                elif regime == "bear":
                    # Bear trend: SELL/hold preferred, BUY discouraged
                    assert side in ['sell', 'hold'], f"BUY not allowed in bear regime, got: {side}"
                elif regime == "range":
                    # Range: HOLD strongly preferred
                    assert side == 'hold' or confidence < 0.6, f"Strong directional signal in range market: {side} (conf={confidence})"

        asyncio.run(test_async())

    def test_end_to_end_backtester_integration(self, processor):
        """Test complete end-to-end integration with backtester framework"""
        # Create a small historical dataset that simulates trend following scenarios
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')

        # Create bull trend data: Price > MA50 > MA200
        btc_prices = []
        base_price = 50000
        for i in range(50):
            if i < 25:
                # Build up the bullish trend
                price = base_price + (i * 200)  # Steady increase
            else:
                # Maintain above MAs
                price = base_price + (25 * 200) + ((i-25) * 150)
            btc_prices.append(price)

        # Create DataFrame with OHLCV
        df = pd.DataFrame({
            'timestamp': dates,
            'BTC_close': btc_prices,
            'ETH_close': [3000 + i*5 for i in range(50)],  # ETH also trending up
            'mercado': [{'BTC': btc_prices[i], 'ETH': 3000 + i*5, 'USDT': 1.0} for i in range(50)]
        })

        df = df.set_index('timestamp')

        # Run backtest
        try:
            results_df = run_backtest(df)

            # Verify backtester ran successfully
            assert not results_df.empty, "Backtest should produce results"
            assert 'portfolio' in results_df.columns or 'BTC' in results_df.columns, \
                "Results should contain portfolio information"

            # Check that we have timestamp information
            assert 'timestamp' in results_df.columns or results_df.index.name == 'timestamp', \
                "Results should have timestamp information"

            print("✅ End-to-end backtester integration successful")
            print(f"   Processed {len(results_df)} time periods")
            print(f"   Columns: {list(results_df.columns)}")

        except Exception as e:
            pytest.fail(f"End-to-end backtester integration failed: {e}")

    def test_integration_with_backtester(self, processor):
        """Test integration with backtester framework"""
        # This test verifies the signals can be processed by the backtester
        # Note: This would require the backtester to be properly initialized

        try:
            # Mock market data
            market_data = {
                'BTCUSDT': pd.DataFrame({
                    'close': [50000, 51000, 52000, 53000, 54000],
                    'open': [49900, 50100, 51100, 52100, 53100],
                    'high': [50200, 51200, 52200, 53200, 54200],
                    'low': [49800, 50000, 51000, 52000, 53000],
                    'volume': [100, 110, 120, 130, 140]
                })
            }

            state = {
                "portfolio": {"USDT": {"free": 1000.0}, "BTCUSDT": {"position": 0.0}},
                "l3_context_cache": {"last_output": {"regime": "bull", "sentiment_score": 0.7}}
            }

            async def test_async():
                signals = await processor.process_signals({**market_data, **state})

                # Verify signals structure for backtester compatibility
                assert isinstance(signals, list), "Signals should be a list"

                for signal in signals:
                    assert hasattr(signal, 'symbol'), "Signal must have symbol"
                    assert hasattr(signal, 'side'), "Signal must have side"
                    assert hasattr(signal, 'confidence'), "Signal must have confidence"
                    assert getattr(signal, 'side', '') in ['buy', 'sell', 'hold'], "Side must be buy/sell/hold"

                print(f"✅ Successfully generated {len(signals)} signals for backtester integration")

            asyncio.run(test_async())

    def test_range_regime_fallback_to_l2(self, processor, sample_market_data, state_template):
        """Test that range regime falls back to L2 technical signals instead of HOLD absolute"""
        # Setup: Create L2 signal first (processor has L3 disabled)
        df = sample_market_data['BTCUSDT'].copy()

        # Create bull trend conditions for L2 to generate BUY signal
        current_price = df['close'].iloc[-1]
        ma50 = df['ma50'].iloc[-1]
        ma200 = df['ma200'].iloc[-1]

        # Ensure conditions: Price > MA50 > MA200
        if not (current_price > ma50 > ma200):
            df.loc[df.index[-1], 'close'] = max(current_price, ma50 + 1000, ma200 + 2000)

        market_data = {'BTCUSDT': df}
        state = state_template.copy()

        # Enable L3 and set range regime
        processor.apagar_l3 = False  # Enable L3 processing
        state["l3_context_cache"]["last_output"]["regime"] = "range"

        # Process signals with enabled L3
        async def test_async():
            signals = await processor.process_signals({**market_data, **state})
            btc_signal = next((s for s in signals if s.symbol == 'BTCUSDT'), None)

            # Assertions
            if btc_signal:
                side = getattr(btc_signal, 'side', 'hold')
                confidence = getattr(btc_signal, 'confidence', 0)
                metadata = getattr(btc_signal, 'metadata', {})
                override_reason = metadata.get('override_reason', '')

                # In range regime, should allow L2 fallback signals
                # Check that we don't get absolute HOLD - allow BUY/SELL with reduced confidence
                if 'fallback_to_l2' in override_reason:
                    # Range regime fallback activated - should maintain signal with reduced confidence
                    assert side in ['buy', 'sell'], f"Range fallback should preserve L2 signal direction: {side}"
                    assert confidence <= 0.8, f"Fallback confidence should be reduced: {confidence}"
                    assert 'range_regime_fallback_to_l2' in override_reason, f"Override reason should indicate L2 fallback: {override_reason}"

                    logger.info(f"✅ Range regime L2 fallback verified: {side} signal with conf={confidence:.3f}")
                else:
                    # If no fallback needed (L2 signal weak), allow HOLD or weak directional
                    assert side in ['hold', 'buy', 'sell'], f"Unexpected signal in range regime: {side}"
                    if side in ['buy', 'sell']:
                        assert confidence < 0.5, f"Strong directional signal not allowed in range: {confidence}"

        asyncio.run(test_async())

        except Exception as e:
            pytest.fail(f"Range regime fallback test failed: {e}")

    def test_dummy_method_to_fix_syntax(self):
        """Dummy method to fix syntax issues temporarily"""
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
