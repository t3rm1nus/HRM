#!/usr/bin/env python
# Test script for sentiment analysis fix

import asyncio
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from l3_strategy.sentiment_inference import download_reddit, download_news, infer_sentiment
from l3_strategy.l3_processor import generate_l3_output
from core.state_manager import initialize_state

async def test_sentiment():
    print("🧪 Testing sentiment analysis flow...")

    # Test downloading sentiment data
    try:
        print("📥 Testing Reddit download...")
        df_reddit = await download_reddit()
        print(f"✅ Reddit download: {len(df_reddit)} posts")

        print("📥 Testing News download...")
        df_news = download_news()
        print(f"✅ News download: {len(df_news)} articles")

        # Combine texts
        df_all = pd.concat([df_reddit, df_news], ignore_index=True)
        df_all.dropna(subset=['text'], inplace=True)
        texts_list = df_all['text'].tolist()
        print(f"📊 Combined {len(texts_list)} texts for analysis")

        # Test sentiment inference
        print("🧠 Testing sentiment inference...")
        sentiment_results = infer_sentiment(texts_list[:10])  # Test with first 10 texts
        print(f"✅ Sentiment inference: {len(sentiment_results)} results")

        # Calculate average sentiment score
        if sentiment_results:
            avg_sentiment = sum((probs[2] - probs[0]) for probs in sentiment_results) / len(sentiment_results)
            avg_sentiment = (avg_sentiment + 1) / 2  # Normalize to 0-1
            print(f"📈 Average sentiment score: {avg_sentiment:.4f}")

            if abs(avg_sentiment - 0.5) > 0.01:  # Check if significantly different from neutral
                print("🥳 SUCCESS: Sentiment score is no longer zero/neutral!")
                return True
            else:
                print("⚠️  Sentiment score is still neutral, but data was downloaded")
                return False
        else:
            print("❌ No sentiment results generated")
            return False

    except Exception as e:
        print(f"❌ Error in sentiment test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_l3_integration():
    print("\n🎯 Testing L3 integration with sentiment...")

    try:
        # Create minimal market data
        market_data = {
            'BTCUSDT': [{'open': 50000, 'high': 50500, 'low': 49900, 'close': 50250, 'volume': 1.2}],
            'ETHUSDT': [{'open': 3500, 'high': 3550, 'low': 3480, 'close': 3520, 'volume': 10}]
        }

        # Download some sentiment data
        df_reddit = await download_reddit()
        df_news = download_news()
        df_all = pd.concat([df_reddit, df_news], ignore_index=True)
        df_all.dropna(subset=['text'], inplace=True)
        texts_list = df_all['text'].tolist()[:5]  # Just 5 texts for testing

        # Initialize state
        state = initialize_state(['BTCUSDT', 'ETHUSDT'], 3000.0)
        state['market_data'] = market_data

        # Generate L3 output with sentiment
        print("🎯 Generating L3 output with sentiment data...")
        l3_output = generate_l3_output(state, texts_for_sentiment=texts_list)
        print(f"✅ L3 output generated: regime={l3_output.get('regime')}, sentiment_score={l3_output.get('sentiment_score', 0):.4f}")

        return abs(l3_output.get('sentiment_score', 0) - 0.0) > 0.001  # Check if > 0

    except Exception as e:
        print(f"❌ Error in L3 integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🔄 Starting sentiment analysis tests...\n")

    sentiment_ok = await test_sentiment()
    l3_ok = await test_l3_integration()

    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print(f"Sentiment download & analysis: {'✅ PASS' if sentiment_ok else '❌ FAIL'}")
    print(f"L3 integration with sentiment: {'✅ PASS' if l3_ok else '❌ FAIL'}")

    if sentiment_ok and l3_ok:
        print("🎉 ALL TESTS PASSED - Sentiment analysis is now working!")
    else:
        print("⚠️  Some tests failed - sentiment analysis may need further fixes")

if __name__ == "__main__":
    asyncio.run(main())
