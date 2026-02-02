import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from dotenv import load_dotenv
import asyncpraw
import requests
import logging
import asyncio
import aiohttp

# Configurar logger para sentiment
logger = logging.getLogger(__name__)

# ======== CONFIG ========
load_dotenv()
DATA_DIR = "data/datos_inferencia"
os.makedirs(DATA_DIR, exist_ok=True)

# Fechas (últimos 90 días)
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=90)

# 🔹 API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Modelo BERT preentrenado L3 - LAZY LOADING
MODEL_DIR = "models/L3/sentiment"
tokenizer = None
model = None

def _load_bert_models():
    """Load BERT tokenizer and model lazily, only when needed"""
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return tokenizer, model

    try:
        logger.info("🤖 SENTIMENT: Loading BERT models lazily...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()  # evaluation mode
        logger.info("✅ SENTIMENT: BERT models loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ SENTIMENT: Failed to load BERT models: {e}")
        raise

# Cache para análisis de sentimiento BERT (6 horas para coincidir con cache de textos)
SENTIMENT_BERT_CACHE_FILE = os.path.join(DATA_DIR, "sentiment_bert_cache.json")
SENTIMENT_BERT_CACHE_DURATION = 21600  # 6 horas en segundos (21600 / 3600 = 6 hours)

def _load_sentiment_bert_cache():
    """Carga análisis BERT completo desde cache si está fresco"""
    try:
        if not os.path.exists(SENTIMENT_BERT_CACHE_FILE):
            logger.debug(f"📭 SENTIMENT: BERT cache file does not exist: {SENTIMENT_BERT_CACHE_FILE}")
            return None

        with open(SENTIMENT_BERT_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Verificar si el cache está fresco
        cache_timestamp = cache_data.get('timestamp')
        if not cache_timestamp:
            logger.warning("⚠️ SENTIMENT: BERT cache missing timestamp")
            return None

        cache_time = datetime.fromisoformat(cache_timestamp)
        current_time = datetime.now()
        age_seconds = (current_time - cache_time).total_seconds()

        if age_seconds > SENTIMENT_BERT_CACHE_DURATION:
            age_hours = age_seconds / 3600
            logger.info(f"📅 SENTIMENT: BERT cache EXPIRED (edad: {age_hours:.1f}h > {SENTIMENT_BERT_CACHE_DURATION/3600:.1f}h) - Will download fresh data")
            return None

        # Validate cache data structure
        sentiment_results = cache_data.get('sentiment_results', [])
        texts_count = cache_data.get('texts_count', 0)
        original_texts = cache_data.get('original_texts', [])

        # Check if data is valid
        if not isinstance(sentiment_results, list) or not isinstance(original_texts, list):
            logger.warning("⚠️ SENTIMENT: BERT cache has invalid data types")
            return None

        if len(sentiment_results) != texts_count or len(original_texts) != texts_count:
            logger.warning(f"⚠️ SENTIMENT: BERT cache data length mismatch: results={len(sentiment_results)}, texts={len(original_texts)}, count={texts_count}")
            return None

        # Check if all texts are strings
        if not all(isinstance(t, str) for t in original_texts):
            logger.warning("⚠️ SENTIMENT: BERT cache contains non-string texts")
            return None

        # Check if sentiment results are valid probability lists
        if not all(isinstance(r, list) and len(r) == 3 and all(isinstance(p, (int, float)) for p in r) for r in sentiment_results):
            logger.warning("⚠️ SENTIMENT: BERT cache contains invalid sentiment results")
            return None

        age_hours = age_seconds / 3600
        sentiment_score = cache_data.get('sentiment_score', 'N/A')
        logger.info(f"✅ SENTIMENT: BERT cache VALID (edad: {age_hours:.1f}h < {SENTIMENT_BERT_CACHE_DURATION/3600:.1f}h, {texts_count} textos, score: {sentiment_score})")
        return cache_data

    except Exception as e:
        logger.warning(f"⚠️ SENTIMENT: Error cargando cache BERT: {e}")
        return None

def _save_sentiment_bert_cache(sentiment_results, sentiment_score, texts_count, original_texts=None):
    """Guarda análisis BERT completo en cache (resultados detallados + score agregado + textos originales)"""
    try:
        cache_data = {
            'sentiment_results': sentiment_results,  # Los resultados detallados de BERT por texto
            'sentiment_score': float(sentiment_score),  # Score agregado
            'texts_count': texts_count,
            'timestamp': datetime.now().isoformat(),
            'original_texts': original_texts  # GUARDAR TEXTOS ORIGINALES PARA CACHE VÁLIDO
        }

        with open(SENTIMENT_BERT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"💾 SENTIMENT: Cache BERT guardado completo ({texts_count} textos, score: {sentiment_score:.3f})")
        return True  # Return success flag

    except Exception as e:
        logger.warning(f"⚠️ SENTIMENT: Error guardando cache BERT: {e}")
        return False  # Return failure flag

# =========================
# DESCARGA DE DATOS
# =========================
async def download_reddit(subreddits=["CryptoCurrency", "Bitcoin", "Ethereum"], limit=500):
    logger.info(f"🔄 SENTIMENT: Iniciando descarga de Reddit - Subreddits: {subreddits}, Limit: {limit}")

    # 🚨 CRITICAL FIX: download_reddit should NEVER download fresh data automatically
    # This function should only return cached data if available and valid
    # Fresh data downloads should only happen through main.py's controlled update_sentiment_texts()
    cached_data = _load_sentiment_bert_cache()

    if cached_data and cached_data.get('original_texts'):
        # Check if BERT cache is actually valid (not just exists)
        cache_timestamp = cached_data.get('timestamp')
        if cache_timestamp:
            cache_time = datetime.fromisoformat(cache_timestamp)
            current_time = datetime.now()
            age_seconds = (current_time - cache_time).total_seconds()

            # Only use cached texts if BERT cache is still fresh
            if age_seconds <= SENTIMENT_BERT_CACHE_DURATION:
                original_texts = cached_data['original_texts']
                # Validate that original_texts is a list of strings
                if not isinstance(original_texts, list) or not all(isinstance(t, str) for t in original_texts):
                    logger.warning("⚠️ BERT cache contains invalid original_texts, returning empty DataFrame")
                    return pd.DataFrame()
                # Estimar cuántos son posts de Reddit (aprox. primera mitad de textos)
                reddit_texts = original_texts[:len(original_texts)//2]  # Mitad para Reddit
                logger.info(f"✅ BERT cache válido - Usando {len(reddit_texts)} textos reales de Reddit del cache")
                # Convertir textos de cache a DataFrame consistente
                posts_data = [{"date": END_DATE - timedelta(hours=i), "text": text} for i, text in enumerate(reddit_texts)]
                return pd.DataFrame(posts_data)
            else:
                logger.warning(f"⏳ BERT cache expirado ({age_seconds/3600:.1f}h > {SENTIMENT_BERT_CACHE_DURATION/3600:.1f}h) - Devolviendo DataFrame vacío (esperando actualización controlada)")
                return pd.DataFrame()  # Return empty DataFrame instead of downloading fresh data
        else:
            logger.warning("⚠️ BERT cache missing timestamp - returning empty DataFrame")
            return pd.DataFrame()
    else:
        logger.info("📭 No BERT cache available - returning empty DataFrame")
        return pd.DataFrame()

    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        logger.warning("⚠️ SENTIMENT: Reddit API keys no configuradas - Usando datos sintéticos de respaldo")
        return _generate_synthetic_reddit_data()

    logger.info("🔑 SENTIMENT: Reddit API keys configuradas - Iniciando conexión")

    # Usar context manager para asegurar cierre de sesiones HTTP
    posts = []
    total_posts = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    try:
        async with asyncpraw.Reddit(client_id=REDDIT_CLIENT_ID,
                                    client_secret=REDDIT_CLIENT_SECRET,
                                    user_agent=REDDIT_USER_AGENT) as reddit:

            for sub in subreddits:
                logger.info(f"📱 SENTIMENT: Descargando subreddit r/{sub}...")
                try:
                    subreddit = await reddit.subreddit(sub)
                    count = 0
                    async for post in subreddit.hot(limit=limit):
                        posts.append({
                            "date": datetime.fromtimestamp(post.created_utc),
                            "text": f"{post.title} {post.selftext}"
                        })
                        count += 1
                        total_posts += 1
                        if count >= limit:
                            break

                    logger.info(f"✅ SENTIMENT: r/{sub} - Descargados {count} posts")
                    consecutive_failures = 0  # Reset on success

                except Exception as e:
                    consecutive_failures += 1
                    logger.error(f"❌ SENTIMENT: Error descargando r/{sub}: {e}")

                    # Si demasiados fallos, devolver DataFrame vacío
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"🚨 SENTIMENT: Demasiados fallos consecutivos en Reddit ({consecutive_failures}), no se descargarán datos")
                        return pd.DataFrame()  # Devolver DataFrame vacío

    except Exception as e:
        logger.error(f"❌ SENTIMENT: Error general en conexión Reddit: {e}")
        # No usar datos sintéticos, devolver DataFrame vacío
        logger.warning("⚠️ SENTIMENT: Error en Reddit API, no se procesarán datos")
        return pd.DataFrame()  # Devolver DataFrame vacío
    finally:
        # Asegurar que las sesiones se cierren
        try:
            import asyncio
            # Forzar limpieza de tareas pendientes
            pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if pending_tasks:
                logger.debug(f"🧹 Limpiando {len(pending_tasks)} tareas pendientes de Reddit")
        except:
            pass

    # Si no se obtuvieron posts, devolver DataFrame vacío
    if total_posts == 0:
        logger.warning("⚠️ SENTIMENT: No se obtuvieron posts de Reddit, no se procesarán datos")
        return pd.DataFrame()  # Devolver DataFrame vacío

    logger.info(f"📊 SENTIMENT: Reddit total descargado: {total_posts} posts de {len(subreddits)} subreddits")
    return pd.DataFrame(posts)

def _generate_synthetic_reddit_data(num_posts=50):
    """Genera datos de Reddit sintéticos FUERA DE LÍNEA cuando las APIs fallan - EQUILIBRADOS PARA EVITAR SESGO POSITIVO"""
    logger.info(f"🎭 SENTIMENT: Generando {num_posts} posts de Reddit sintéticos equilibrados (fuera de línea)")

    # Distribución realista basada en análisis de sentimiento histórico crypto (50% neutral, 25% positivo, 25% negativo)
    neutral_posts = [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Crypto market is volatile but the technology is solid"
        } for i in range(num_posts//2)
    ]

    positive_posts = [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Some institutional adoption happening, cautiously optimistic about BTC fundamentals"
        } for i in range(num_posts//4)
    ]

    negative_posts = [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Concerned about the recent dip, is this the beginning of a correction?"
        } for i in range(num_posts//4)
    ]

    # Combinar posts equitativamente
    synthetic_posts = neutral_posts[:num_posts//4] + positive_posts[:num_posts//4] + negative_posts[:num_posts//4] + neutral_posts[num_posts//4:num_posts//2]

    # Añadir más variación
    import random
    extra_sentiments = [
        "Bullish on BTC long term despite short term fluctuations",
        "Ethereum gas fees are killing me, when will layer 2 solve this?",
        "Crypto regulation might be coming, better days ahead for adoption",
        "Market manipulation is real, but fundamentals remain strong",
        "Just bought the dip, feeling confident about recovery",
        "Bear market incoming? Time to accumulate more coins",
        "DeFi yields are incredible, APY over 100% on some protocols",
        "NFT market crashed but utility NFTs will survive",
        "Centralized exchanges are risky, self-custody is the future",
        "Bitcoin as digital gold narrative is gaining traction"
    ]

    # Mezclar con posts adicionales
    for i in range(max(0, num_posts - len(synthetic_posts))):
        synthetic_posts.append({
            "date": (END_DATE - timedelta(hours=random.randint(1, 24))),
            "text": random.choice(extra_sentiments)
        })

    logger.debug(f"🎭 SENTIMENT: Posts sintéticos Reddit generados con variación de sentimiento")
    return pd.DataFrame(synthetic_posts)

def download_news(query="crypto OR bitcoin OR ethereum OR blockchain"):
    logger.info(f"📰 SENTIMENT: Iniciando descarga de noticias - Query: '{query}'")

    # 🚨 CRITICAL FIX: download_news should NEVER download fresh data automatically
    # This function should only return cached data if available and valid
    # Fresh data downloads should only happen through main.py's controlled update_sentiment_texts()
    cached_data = _load_sentiment_bert_cache()

    if cached_data and cached_data.get('original_texts'):
        # Check if BERT cache is actually valid (not just exists)
        cache_timestamp = cached_data.get('timestamp')
        if cache_timestamp:
            cache_time = datetime.fromisoformat(cache_timestamp)
            current_time = datetime.now()
            age_seconds = (current_time - cache_time).total_seconds()

            # Only use cached texts if BERT cache is still fresh
            if age_seconds <= SENTIMENT_BERT_CACHE_DURATION:
                original_texts = cached_data['original_texts']
                # Validate that original_texts is a list of strings
                if not isinstance(original_texts, list) or not all(isinstance(t, str) for t in original_texts):
                    logger.warning("⚠️ BERT cache contains invalid original_texts, returning empty DataFrame")
                    return pd.DataFrame()
                # Estimar cuántos son artículos de noticias (aprox. segunda mitad de textos)
                news_texts = original_texts[len(original_texts)//2:]  # Mitad para noticias
                logger.info(f"✅ BERT cache válido - Usando {len(news_texts)} textos reales de noticias del cache")
                # Convertir textos de cache a DataFrame consistente
                news_data = [{"date": (END_DATE - timedelta(hours=i)).isoformat(), "text": text} for i, text in enumerate(news_texts)]
                return pd.DataFrame(news_data)
            else:
                logger.warning(f"⏳ BERT cache expirado ({age_seconds/3600:.1f}h > {SENTIMENT_BERT_CACHE_DURATION/3600:.1f}h) - Devolviendo DataFrame vacío (esperando actualización controlada)")
                return pd.DataFrame()  # Return empty DataFrame instead of downloading fresh data
        else:
            logger.warning("⚠️ BERT cache missing timestamp - returning empty DataFrame")
            return pd.DataFrame()
    else:
        logger.info("📭 No BERT cache available - returning empty DataFrame")
        return pd.DataFrame()

    if not NEWS_API_KEY:
        logger.warning("⚠️ SENTIMENT: NEWS_API_KEY no configurada - Usando datos sintéticos de respaldo")
        return _generate_synthetic_news_data()

    logger.info("🔑 SENTIMENT: News API key configurada - Iniciando descarga")

    # 🛠️ Rate limiting fix: Simpler approach - fetch last 24 hours with one request, higher pageSize
    START_DATE_LIMITED = END_DATE - timedelta(days=1)  # Last 24 hours only
    logger.info(f"📅 SENTIMENT: Descargando noticias desde {START_DATE_LIMITED.date()} hasta {END_DATE.date()}")

    url = (
        f"https://newsapi.org/v2/everything?q={query}&language=en"
        f"&from={START_DATE_LIMITED.date()}&to={END_DATE.date()}"
        f"&sortBy=publishedAt&pageSize=50&apiKey={NEWS_API_KEY}"
    )

    max_retries = 3
    retry_delay = 5  # Increased retry delay

    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 SENTIMENT: Attempt {attempt+1}/{max_retries} - Requesting news data...")

            import time
            if attempt > 0:
                # Longer delay on retries
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"⚠️ SENTIMENT: Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

            response = requests.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            if "articles" in data and data["articles"]:
                all_articles = [{
                    "date": a["publishedAt"],
                    "text": a.get("title","") + " " + str(a.get("content",""))
                } for a in data["articles"]]

                total_articles = len(all_articles)
                logger.info(f"✅ SENTIMENT: Successfully downloaded {total_articles} articles")
                return pd.DataFrame(all_articles)
            else:
                logger.warning("⚠️ SENTIMENT: No articles found in response")
                if attempt < max_retries - 1:
                    continue
                else:
                    break

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ SENTIMENT: Rate limit reached, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning("❌ SENTIMENT: Rate limit persists, using synthetic data")
                    break
            else:
                logger.error(f"❌ SENTIMENT: HTTP error {response.status_code}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break

        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ SENTIMENT: Connection error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                break
        except Exception as e:
            logger.error(f"❌ SENTIMENT: Unexpected error: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                break

    # Fallback to LAST REAL CACHED DATA first, then synthetic data
    logger.warning("🚨 SENTIMENT: News API failed after retries, checking for cached data...")

    # Try to load cached data from previous successful downloads
    try:
        # Look for files that might contain cached news data
        import glob
        import os

        news_cache_files = [
            os.path.join(DATA_DIR, f"sentiment_l2_{pd.Timestamp.now().date()}.json"),
            os.path.join(DATA_DIR, f"sentiment_l2_{(pd.Timestamp.now() - pd.Timedelta(days=1)).date()}.json")
        ]

        for cache_file in news_cache_files:
            if os.path.exists(cache_file):
                try:
                    # Load JSON array (not JSON Lines)
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)

                    if cached_data and isinstance(cached_data, list) and len(cached_data) > 0:
                        cached_df = pd.DataFrame(cached_data)
                        # Filter to keep only text fields that look like news articles
                        cached_df = cached_df[cached_df['text'].str.len() > 100]  # Longer texts are likely real news

                        if len(cached_df) >= 5:  # At least some articles
                            logger.info(f"✅ SENTIMENT: Using cached news data ({len(cached_df)} articles) from {cache_file}")
                            return cached_df.head(num_articles//2).rename(columns={'date': 'date', 'text': 'text'})  # Limit to half the requested amount

                except Exception as cache_error:
                    logger.debug(f"Cached file {cache_file} not usable: {cache_error}")
                    continue

    except Exception as cache_check_error:
        logger.debug(f"Error checking cache: {cache_check_error}")

    # Final fallback to synthetic data if no cached data found
    logger.warning("❌ SENTIMENT: No cached data available, using synthetic data")
    synthetic_df = _generate_synthetic_news_data()
    logger.info(f"📊 SENTIMENT: Synthetic data generated: {len(synthetic_df)} articles")
    return synthetic_df

def _generate_synthetic_news_data(num_articles=20):
    """Genera datos de noticias sintéticos cuando las APIs fallan"""
    logger.info(f"🎭 SENTIMENT: Generando {num_articles} artículos de noticias sintéticos")

    # Artículos de ejemplo con sentimiento variado
    synthetic_articles = [
        {
            "date": (END_DATE - timedelta(hours=i)).isoformat(),
            "text": "Bitcoin shows strong momentum as institutional adoption increases globally"
        } for i in range(num_articles//4)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)).isoformat(),
            "text": "Ethereum network upgrade boosts developer activity and ecosystem growth"
        } for i in range(num_articles//4)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)).isoformat(),
            "text": "Cryptocurrency market faces regulatory uncertainty but innovation continues"
        } for i in range(num_articles//4)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)).isoformat(),
            "text": "DeFi sector demonstrates resilience despite market volatility challenges"
        } for i in range(num_articles - 3*(num_articles//4))
    ]

    # Añadir variación aleatoria
    import random
    sentiments = ["bullish", "bearish", "neutral", "optimistic", "cautious"]
    for article in synthetic_articles:
        sentiment = random.choice(sentiments)
        if sentiment == "bullish":
            article["text"] += ". Market sentiment remains positive with strong buying pressure."
        elif sentiment == "bearish":
            article["text"] += ". Concerns about market correction persist among investors."
        elif sentiment == "optimistic":
            article["text"] += ". Analysts remain optimistic about long-term growth potential."
        elif sentiment == "cautious":
            article["text"] += ". Investors adopt cautious approach amid economic uncertainty."
        else:
            article["text"] += ". Market conditions remain stable with mixed signals."

    logger.debug(f"🎭 SENTIMENT: Artículos sintéticos generados con variación de sentimiento")
    return pd.DataFrame(synthetic_articles)

def get_cached_sentiment_score(max_age_hours=6, market_data=None):
    """
    FIX: Get cached sentiment score if available and fresh, otherwise provide meaningful heuristic.

    Instead of always returning None when sentiment is unavailable, provide market-based sentiment.
    """
    try:
        if not os.path.exists(SENTIMENT_BERT_CACHE_FILE):
            logger.debug(f"⚠️ SENTIMENT: BERT cache file does not exist, using heuristic")
            return _generate_heuristic_sentiment_score(market_data)

        with open(SENTIMENT_BERT_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Check timestamp
        cache_timestamp = cache_data.get('timestamp')
        if not cache_timestamp:
            logger.debug("⚠️ SENTIMENT: BERT cache missing timestamp, using heuristic")
            return _generate_heuristic_sentiment_score(market_data)

        cache_time = datetime.fromisoformat(cache_timestamp)
        current_time = datetime.now()
        age_seconds = (current_time - cache_time).total_seconds()
        age_hours = age_seconds / 3600

        if age_seconds > (max_age_hours * 3600):
            logger.debug(f"📅 SENTIMENT: BERT cache expired ({age_hours:.1f}h > {max_age_hours}h), using heuristic")
            return _generate_heuristic_sentiment_score(market_data)

        sentiment_score = cache_data.get('sentiment_score')
        if sentiment_score is None:
            logger.debug("⚠️ SENTIMENT: BERT cache missing sentiment_score, using heuristic")
            return _generate_heuristic_sentiment_score(market_data)

        logger.info(f"✅ SENTIMENT: BERT cache fresh ({age_hours:.1f}h < {max_age_hours}h), returning cached score: {sentiment_score:.4f}")
        return sentiment_score

    except Exception as e:
        logger.debug(f"⚠️ SENTIMENT: Error checking cached sentiment score: {e}, using heuristic")
        return _generate_heuristic_sentiment_score(market_data)


def _generate_heuristic_sentiment_score(market_data=None):
    """
    Generate meaningful sentiment score based on market conditions when sentiment data is unavailable.

    Instead of returning neutral 0.5, provide market-aware sentiment:
    - Bull markets: Slightly bullish sentiment (0.55-0.65)
    - Bear markets: Slightly bearish sentiment (0.35-0.45)
    - Range markets: Neutral sentiment (0.45-0.55)
    - High volatility: More cautious sentiment
    """
    try:
        if not market_data:
            logger.debug("⚠️ SENTIMENT: No market data available for heuristic")
            return 0.5  # True neutral fallback

        # Get BTC data for sentiment heuristic
        btc_data = market_data.get('BTCUSDT', {})
        if isinstance(btc_data, dict) and 'historical_data' in btc_data:
            df = btc_data['historical_data']
            if isinstance(df, pd.DataFrame) and len(df) >= 20:
                # Calculate recent price change (last 20 periods)
                recent_prices = df['close'].tail(20)
                if len(recent_prices) >= 2:
                    price_change_pct = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

                    # Calculate volatility (standard deviation of returns)
                    returns = recent_prices.pct_change().dropna()
                    volatility = returns.std() if len(returns) > 0 else 0.02

                    # Base sentiment on price action
                    base_sentiment = 0.5  # Neutral starting point

                    # Price momentum adjustment
                    if price_change_pct > 0.05:  # Strong upward momentum
                        base_sentiment += 0.10  # More bullish
                    elif price_change_pct > 0.02:  # Moderate upward momentum
                        base_sentiment += 0.05
                    elif price_change_pct < -0.05:  # Strong downward momentum
                        base_sentiment -= 0.10  # More bearish
                    elif price_change_pct < -0.02:  # Moderate downward momentum
                        base_sentiment -= 0.05

                    # Volatility adjustment (higher volatility = more cautious)
                    if volatility > 0.05:  # High volatility
                        # Pull sentiment toward neutral
                        base_sentiment = 0.5 + (base_sentiment - 0.5) * 0.7
                    elif volatility > 0.03:  # Moderate volatility
                        base_sentiment = 0.5 + (base_sentiment - 0.5) * 0.85

                    # RSI adjustment if available
                    if 'rsi' in df.columns and len(df) > 0:
                        rsi = df['rsi'].iloc[-1]
                        if rsi > 70:  # Overbought
                            base_sentiment -= 0.05
                        elif rsi < 30:  # Oversold
                            base_sentiment += 0.05

                    # Ensure bounds
                    sentiment_score = max(0.35, min(0.65, base_sentiment))

                    logger.info(f"🧠 SENTIMENT: Heuristic score generated: {sentiment_score:.3f} "
                               f"(price_change: {price_change_pct:.2%}, volatility: {volatility:.3f})")

                    return sentiment_score

        # Fallback if data is insufficient
        logger.debug("⚠️ SENTIMENT: Insufficient market data for heuristic analysis")
        return 0.5

    except Exception as e:
        logger.debug(f"⚠️ SENTIMENT: Error generating heuristic sentiment: {e}")
        return 0.5  # Safe neutral fallback

def should_use_full_bert_cache(text_count):
    """
    Check if full BERT cache should be used based on count matching
    This is more strict than sentiment score cache - requires exact count match
    """
    try:
        if not os.path.exists(SENTIMENT_BERT_CACHE_FILE):
            return False

        with open(SENTIMENT_BERT_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # For full BERT cache, we need exact count match AND fresh timestamp
        cache_timestamp = cache_data.get('timestamp')
        if not cache_timestamp:
            return False

        cache_time = datetime.fromisoformat(cache_timestamp)
        current_time = datetime.now()
        age_seconds = (current_time - cache_time).total_seconds()
        if age_seconds > (SENTIMENT_BERT_CACHE_DURATION):
            return False

        cached_count = cache_data.get('texts_count', 0)
        return cached_count == text_count

    except Exception as e:
        logger.debug(f"⚠️ Error checking full BERT cache usability: {e}")
        return False

# =========================
# LIMPIEZA DE RECURSOS
# =========================
def cleanup_http_resources():
    """Limpia recursos HTTP no cerrados para prevenir memory leaks."""
    try:
        # Limpiar sesiones aiohttp si existen
        import asyncio
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            # Obtener todas las tareas pendientes
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]

            # Cancelar tareas relacionadas con HTTP si es necesario
            http_tasks = [task for task in pending_tasks if 'http' in str(task).lower() or 'aiohttp' in str(task).lower()]
            if http_tasks:
                logger.debug(f"🧹 Cancelando {len(http_tasks)} tareas HTTP pendientes")

                for task in http_tasks:
                    try:
                        task.cancel()
                    except Exception as e:
                        logger.debug(f"Error cancelando tarea HTTP: {e}")

        # Forzar garbage collection
        import gc
        gc.collect()

        logger.debug("✅ Recursos HTTP limpiados correctamente")

    except Exception as e:
        logger.warning(f"⚠️ Error durante limpieza de recursos HTTP: {e}")

# Registrar cleanup al salir
import atexit
atexit.register(cleanup_http_resources)

# =========================
# INFERENCIA
# =========================
def infer_sentiment(texts, batch_size=16, force_save=False):
    logger.info(f"🧠 SENTIMENT: Iniciando inferencia de sentimiento - {len(texts)} textos, batch_size={batch_size}")

    # Check if sentiment analysis is disabled via environment variable
    from comms.config import APAGAR_L3
    if APAGAR_L3:
        logger.info("⚠️ SENTIMENT: Sentiment analysis disabled by APAGAR_L3 environment variable")
        # Return neutral sentiment for all texts
        neutral_probs = [0.33, 0.34, 0.33]
        return [neutral_probs] * len(texts)

    if not texts or len(texts) == 0:
        logger.warning("⚠️ SENTIMENT: No hay textos para analizar")
        return []

    # Filtrar textos vacíos y asegurar que sean strings
    valid_texts = [str(t).strip() for t in texts if t and str(t).strip()]
    if len(valid_texts) != len(texts):
        logger.info(f"🧹 SENTIMENT: Filtrados {len(texts) - len(valid_texts)} textos vacíos, quedan {len(valid_texts)}")

    # ✅ CRITICAL FIX: Always check for valid cached analysis results first
    # This prevents re-analyzing the same texts when cache is still valid
    bert_cache_data = _load_sentiment_bert_cache()
    if bert_cache_data and not force_save:
        cached_results = bert_cache_data.get('sentiment_results', [])
        cached_count = bert_cache_data.get('texts_count', 0)
        cached_texts = bert_cache_data.get('original_texts', [])

        # Use cached results if we have exact text match (same analysis already done)
        if (cached_results and len(cached_results) == len(valid_texts) and
            cached_count == len(valid_texts) and cached_texts == valid_texts):
            logger.info("✅ SENTIMENT: Usando cache BERT completo - análisis ya realizado!")
            results = cached_results.copy()
            sentiment_score = bert_cache_data.get('sentiment_score', 0.5)
            logger.info(f"🎯 SENTIMENT: Cache completo usado - score: {sentiment_score:.3f}, {len(results)} textos procesados instantáneamente")
            return results

    # If we get here, we need to do fresh analysis
    logger.info(f"🔄 SENTIMENT: Realizando análisis fresco de {len(valid_texts)} textos")

    # Full processing (only when cache is expired/missing or force_save=True)
    results = []
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size

    logger.info(f"📊 SENTIMENT: Procesando {total_batches} batches de inferencia completa...")

    for batch_idx, i in enumerate(range(0, len(valid_texts), batch_size), 1):
        batch = valid_texts[i:i+batch_size]
        batch_size_actual = len(batch)

        logger.debug(f"🔢 SENTIMENT: Batch {batch_idx}/{total_batches} - {batch_size_actual} textos")

        try:
            # Load BERT models lazily (if not already loaded)
            tokenizer, model = _load_bert_models()

            # Tokenizar
            encodings = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")

            # Inferencia
            with torch.no_grad():
                outputs = model(**encodings)
                raw_probs = torch.softmax(outputs.logits, dim=1).tolist()

                # Convert to 3-class format (negative, neutral, positive)
                processed_probs = []
                for prob in raw_probs:
                    if len(prob) == 2:  # Binary model (negative, positive)
                        neg_prob, pos_prob = prob[0], prob[1]
                        # Assume neutral is split between them when close to 0.5
                        if abs(neg_prob - 0.5) < 0.1 and abs(pos_prob - 0.5) < 0.1:
                            # Very neutral
                            processed_probs.append([neg_prob, 0.8, pos_prob])  # High neutral
                        elif pos_prob > neg_prob:
                            # Mostly positive
                            neutral = min(pos_prob, neg_prob) * 0.5
                            processed_probs.append([neg_prob - neutral/2, neutral, pos_prob + neutral/2])
                        else:
                            # Mostly negative
                            neutral = min(pos_prob, neg_prob) * 0.5
                            processed_probs.append([neg_prob + neutral/2, neutral, pos_prob - neutral/2])
                    elif len(prob) == 3:  # Already 3-class
                        processed_probs.append(prob)
                    else:  # Unexpected format, use neutral
                        processed_probs.append([0.33, 0.34, 0.33])

                results.extend(processed_probs)

            # Log progreso cada 5 batches
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                logger.info(f"✅ SENTIMENT: Completado batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.1f}%)")

        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error en batch {batch_idx}: {e}")
            # Agregar probabilidades neutras para textos fallidos
            results.extend([[0.33, 0.34, 0.33]] * batch_size_actual)

    # Calcular score promedio y guardar cache BERT completa
    if results:
        try:
            # Calcular score promedio (simplificado: clase 2 - clase 0)
            avg_sentiment = sum((probs[2] - probs[0]) for probs in results) / len(results)
            # Normalizar a rango 0-1
            sentiment_score = (avg_sentiment + 1) / 2

            # CRITICAL FIX: Guardar cache inmediatamente para asegurar persistencia, incluyendo textos originales
            success = _save_sentiment_bert_cache(results, sentiment_score, len(valid_texts), valid_texts)
            if success:
                logger.info(f"✅ BERT cache saved: score={sentiment_score:.4f}, texts={len(valid_texts)}")
            else:
                logger.error("❌ Failed to save BERT cache!")

        except Exception as e:
            logger.debug(f"⚠️ SENTIMENT: Error guardando cache BERT completa: {e}")

    logger.info(f"🎯 SENTIMENT: Inferencia completa finalizada - {len(results)} resultados generados")
    return results

def save_sentiment_results(df_reddit, df_news):
    """Save sentiment analysis results given downloaded data"""
    logger.info("💾 SENTIMENT: Saving sentiment analysis results...")

    try:
        # Combinar
        df_all = pd.concat([df_reddit, df_news], ignore_index=True)
        df_all.dropna(subset=['text'], inplace=True)

        if df_all.empty:
            logger.warning("⚠️ SENTIMENT: No data to save")
            return None

        # Inferencia
        texts_list = df_all['text'].tolist()
        sentiment_results = infer_sentiment(texts_list)

        df_all['sentiment_probs'] = sentiment_results
        df_all['predicted_class'] = df_all['sentiment_probs'].apply(lambda x: int(np.argmax(x)) if x else 1)

        # Guardar CSV de inferencia
        csv_path = os.path.join(DATA_DIR, f"sentiment_inference_{END_DATE.date()}.csv")
        df_all.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV de inferencia guardado en '{csv_path}'")

        # Guardar JSON para L2
        json_path = os.path.join(DATA_DIR, f"sentiment_l2_{END_DATE.date()}.json")
        # Convert DataFrame to dict, converting dates to ISO format strings
        json_data = df_all.to_dict('records')
        for record in json_data:
            if 'date' in record and hasattr(record['date'], 'isoformat'):
                record['date'] = record['date'].isoformat()

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ JSON listo para L2 guardado en '{json_path}'")

        # Also save a summary file
        summary_path = os.path.join(DATA_DIR, f"sentiment_summary_{END_DATE.date()}.json")
        summary = {
            "total_texts": len(df_all),
            "execution_date": END_DATE.isoformat(),
            "csv_file": csv_path,
            "json_file": json_path,
            "sentiment_distribution": df_all['predicted_class'].value_counts().to_dict()
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Resumen guardado en '{summary_path}'")

        return df_all

    except Exception as e:
        logger.error(f"❌ SENTIMENT: Error saving sentiment results: {e}")
        raise

async def run_sentiment_analysis_and_save_async():
    """Run complete sentiment analysis and save results to files"""
    logger.info("🚀 SENTIMENT: Running complete sentiment analysis with file saving...")

    try:
        print("⏳ Descargando datos de sentimiento...")
        df_reddit = await download_reddit()
        df_news = download_news()

        result = save_sentiment_results(df_reddit, df_news)
        print(f"✅ SENTIMENT: Complete analysis completed and saved: {len(result) if result is not None else 0} texts processed")
        return result

    except Exception as e:
        logger.error(f"❌ SENTIMENT: Error in complete sentiment analysis: {e}")
        raise
    else:
        logger.info("✅ SENTIMENT: Analysis completed successfully")

def run_sentiment_analysis_and_save():
    """Sync wrapper for the async sentiment analysis"""
    import asyncio
    return asyncio.run(run_sentiment_analysis_and_save_async())

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Run the async sentiment analysis
    run_sentiment_analysis_and_save()
