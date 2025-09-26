import os
import json
import pandas as pd
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

# Modelo BERT preentrenado L3
MODEL_DIR = "models/L3/sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # modo evaluación

# Cache para datos de sentimiento
SENTIMENT_CACHE_FILE = os.path.join(DATA_DIR, "sentiment_cache.json")
SENTIMENT_CACHE_DURATION = 3600  # 1 hora en segundos

def _load_sentiment_cache():
    """Carga datos de sentimiento desde cache si están frescos"""
    try:
        if not os.path.exists(SENTIMENT_CACHE_FILE):
            return None

        with open(SENTIMENT_CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Verificar si el cache está fresco
        cache_timestamp = cache_data.get('timestamp')
        if not cache_timestamp:
            return None

        cache_time = datetime.fromisoformat(cache_timestamp)
        current_time = datetime.now()
        age_seconds = (current_time - cache_time).total_seconds()

        if age_seconds > SENTIMENT_CACHE_DURATION:
            logger.debug(f"📅 SENTIMENT: Cache expirado (edad: {age_seconds:.0f}s > {SENTIMENT_CACHE_DURATION}s)")
            return None

        logger.info(f"✅ SENTIMENT: Cache cargado (edad: {age_seconds:.0f}s)")
        return cache_data

    except Exception as e:
        logger.warning(f"⚠️ SENTIMENT: Error cargando cache: {e}")
        return None

def _save_sentiment_cache(sentiment_score, texts_count):
    """Guarda datos de sentimiento en cache"""
    try:
        cache_data = {
            'sentiment_score': float(sentiment_score),
            'texts_count': texts_count,
            'timestamp': datetime.now().isoformat()
        }

        with open(SENTIMENT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

        logger.debug("💾 SENTIMENT: Cache guardado")

    except Exception as e:
        logger.warning(f"⚠️ SENTIMENT: Error guardando cache: {e}")

# =========================
# DESCARGA DE DATOS
# =========================
async def download_reddit(subreddits=["CryptoCurrency", "Bitcoin", "Ethereum"], limit=500):
    logger.info(f"🔄 SENTIMENT: Iniciando descarga de Reddit - Subreddits: {subreddits}, Limit: {limit}")

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

                    # Si demasiados fallos, usar datos sintéticos
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"🚨 SENTIMENT: Demasiados fallos consecutivos en Reddit ({consecutive_failures}), usando datos sintéticos")
                        synthetic_df = _generate_synthetic_reddit_data()
                        logger.info(f"📊 SENTIMENT: Datos sintéticos Reddit generados: {len(synthetic_df)} posts")
                        return synthetic_df

    except Exception as e:
        logger.error(f"❌ SENTIMENT: Error general en conexión Reddit: {e}")
        # Fallback a datos sintéticos
        logger.warning("⚠️ SENTIMENT: Error en Reddit API, usando datos sintéticos de respaldo")
        synthetic_df = _generate_synthetic_reddit_data()
        logger.info(f"📊 SENTIMENT: Datos sintéticos Reddit generados: {len(synthetic_df)} posts")
        return synthetic_df
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

    # Si no se obtuvieron posts, usar datos sintéticos
    if total_posts == 0:
        logger.warning("⚠️ SENTIMENT: No se obtuvieron posts de Reddit, usando datos sintéticos de respaldo")
        synthetic_df = _generate_synthetic_reddit_data()
        logger.info(f"📊 SENTIMENT: Datos sintéticos Reddit generados: {len(synthetic_df)} posts")
        return synthetic_df

    logger.info(f"📊 SENTIMENT: Reddit total descargado: {total_posts} posts de {len(subreddits)} subreddits")
    return pd.DataFrame(posts)

def _generate_synthetic_reddit_data(num_posts=50):
    """Genera datos de Reddit sintéticos cuando las APIs fallan"""
    logger.info(f"🎭 SENTIMENT: Generando {num_posts} posts de Reddit sintéticos")

    # Posts de ejemplo con sentimiento variado
    synthetic_posts = [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Bitcoin is mooning! Institutional adoption is accelerating 🚀🚀🚀"
        } for i in range(num_posts//5)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "ETH 2.0 upgrade looks promising, staking rewards are amazing"
        } for i in range(num_posts//5)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Crypto market is volatile but the technology is solid"
        } for i in range(num_posts//5)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Concerned about the recent dip, is this the beginning of a correction?"
        } for i in range(num_posts//5)
    ] + [
        {
            "date": (END_DATE - timedelta(hours=i)),
            "text": "Just HODLing through the volatility, diamond hands 💎🙌"
        } for i in range(num_posts - 4*(num_posts//5))
    ]

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

    if not NEWS_API_KEY:
        logger.warning("⚠️ SENTIMENT: NEWS_API_KEY no configurada - Usando datos sintéticos de respaldo")
        return _generate_synthetic_news_data()

    logger.info("🔑 SENTIMENT: News API key configurada - Iniciando descarga")

    all_articles = []
    total_articles = 0
    current_start = START_DATE

    # 🛠️ RATE LIMITING: Limitar a los últimos 3 días para backtesting (en lugar de 7 días)
    START_DATE_LIMITED = END_DATE - timedelta(days=3)
    current_start = max(current_start, START_DATE_LIMITED)

    logger.info(f"📅 SENTIMENT: Descargando noticias desde {current_start.date()} hasta {END_DATE.date()}")

    consecutive_failures = 0
    max_consecutive_failures = 3

    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=1), END_DATE)  # Un día a la vez para mejor control

        logger.debug(f"🕐 SENTIMENT: Descargando período {current_start.date()} - {current_end.date()}")

        url = (
            f"https://newsapi.org/v2/everything?q={query}&language=en"
            f"&from={current_start.date()}&to={current_end.date()}"
            f"&sortBy=publishedAt&pageSize=30&apiKey={NEWS_API_KEY}"  # Reducir a 30 artículos por día
        )

        # 🛠️ RATE LIMITING: Agregar delay entre requests para evitar 429 errors
        import time
        time.sleep(1.5)  # 1.5 segundos entre requests (NewsAPI free tier permite ~1 request/segundo)

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)  # Aumentar timeout a 15 segundos
                response.raise_for_status()  # Levantar excepción para códigos de error HTTP

                data = response.json()

                if "articles" in data:
                    articles_in_period = len(data["articles"])
                    all_articles.extend([{
                        "date": a["publishedAt"],
                        "text": a.get("title","") + " " + str(a.get("content",""))
                    } for a in data["articles"]])

                    total_articles += articles_in_period
                    logger.debug(f"✅ SENTIMENT: {current_start.date()} - {articles_in_period} artículos")
                    consecutive_failures = 0  # Reset counter on success
                else:
                    logger.warning(f"⚠️ SENTIMENT: No se encontraron artículos para {current_start.date()}")

                break  # Salir del loop de reintentos si fue exitoso

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Too Many Requests
                    consecutive_failures += 1
                    if attempt < max_retries - 1 and consecutive_failures < max_consecutive_failures:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"⚠️ SENTIMENT: Rate limit alcanzado, esperando {wait_time}s antes de reintentar...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ SENTIMENT: Rate limit persistente, saltando día {current_start.date()}")
                        break
                else:
                    logger.error(f"❌ SENTIMENT: Error HTTP {response.status_code} descargando noticias {current_start.date()}: {e}")
                    consecutive_failures += 1
                    break

            except requests.exceptions.RequestException as e:
                consecutive_failures += 1
                if attempt < max_retries - 1 and consecutive_failures < max_consecutive_failures:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"⚠️ SENTIMENT: Error de conexión, reintentando en {wait_time}s... ({e})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ SENTIMENT: Error persistente descargando noticias {current_start.date()}: {e}")
                    break

            except Exception as e:
                logger.error(f"❌ SENTIMENT: Error procesando noticias {current_start.date()}: {e}")
                consecutive_failures += 1
                break

        # 🛠️ FALLBACK: Si demasiados fallos consecutivos, usar datos sintéticos
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(f"🚨 SENTIMENT: Demasiados fallos consecutivos ({consecutive_failures}), usando datos sintéticos de respaldo")
            synthetic_df = _generate_synthetic_news_data()
            logger.info(f"📊 SENTIMENT: Datos sintéticos generados: {len(synthetic_df)} artículos")
            return synthetic_df

        current_start = current_end + timedelta(days=1)

    # 🛠️ FALLBACK: Si no se obtuvieron artículos, generar datos sintéticos
    if total_articles == 0:
        logger.warning("⚠️ SENTIMENT: No se obtuvieron artículos de News API, usando datos sintéticos de respaldo")
        synthetic_df = _generate_synthetic_news_data()
        logger.info(f"📊 SENTIMENT: Datos sintéticos generados: {len(synthetic_df)} artículos")
        return synthetic_df

    logger.info(f"📊 SENTIMENT: News total descargado: {total_articles} artículos")
    return pd.DataFrame(all_articles)

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
def infer_sentiment(texts, batch_size=16):
    logger.info(f"🧠 SENTIMENT: Iniciando inferencia de sentimiento - {len(texts)} textos, batch_size={batch_size}")

    if not texts or len(texts) == 0:
        logger.warning("⚠️ SENTIMENT: No hay textos para analizar")
        return []

    # Filtrar textos vacíos
    valid_texts = [t for t in texts if t and str(t).strip()]
    if len(valid_texts) != len(texts):
        logger.info(f"🧹 SENTIMENT: Filtrados {len(texts) - len(valid_texts)} textos vacíos, quedan {len(valid_texts)}")

    # Intentar usar cache si hay suficientes textos
    if len(valid_texts) >= 10:  # Solo usar cache si hay al menos 10 textos
        cache_data = _load_sentiment_cache()
        if cache_data:
            logger.info("✅ SENTIMENT: Usando datos de cache para inferencia")
            sentiment_score = cache_data.get('sentiment_score', 0.5)
            # Convertir score único a probabilidades por texto (simplificado)
            # En un sistema real, guardaríamos las probabilidades individuales
            neutral_prob = [0.33, 0.34, 0.33]  # Probabilidades neutras por defecto
            if sentiment_score > 0.6:
                bullish_prob = [0.1, 0.8, 0.1]  # Más bullish
            elif sentiment_score < 0.4:
                bearish_prob = [0.8, 0.1, 0.1]  # Más bearish
            else:
                bullish_prob = neutral_prob

            results = [bullish_prob] * len(valid_texts)
            logger.info(f"🎯 SENTIMENT: Cache usado - score: {sentiment_score:.3f}, textos: {len(valid_texts)}")
            return results

    results = []
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size  # Calcular número total de batches

    logger.info(f"📊 SENTIMENT: Procesando {total_batches} batches de inferencia...")

    for batch_idx, i in enumerate(range(0, len(valid_texts), batch_size), 1):
        batch = valid_texts[i:i+batch_size]
        batch_size_actual = len(batch)

        logger.debug(f"🔢 SENTIMENT: Batch {batch_idx}/{total_batches} - {batch_size_actual} textos")

        try:
            # Tokenizar
            encodings = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")

            # Inferencia
            with torch.no_grad():
                outputs = model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1).tolist()

            results.extend(probs)

            # Log progreso cada 5 batches
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                logger.info(f"✅ SENTIMENT: Completado batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.1f}%)")

        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error en batch {batch_idx}: {e}")
            # Agregar probabilidades neutras para textos fallidos
            results.extend([[0.33, 0.34, 0.33]] * batch_size_actual)

    # Calcular y guardar score promedio en cache
    if results:
        try:
            # Calcular score promedio (simplificado: clase 2 - clase 0)
            avg_sentiment = sum((probs[2] - probs[0]) for probs in results) / len(results)
            # Normalizar a rango 0-1
            sentiment_score = (avg_sentiment + 1) / 2
            _save_sentiment_cache(sentiment_score, len(valid_texts))
            logger.debug(f"💾 SENTIMENT: Score promedio guardado en cache: {sentiment_score:.3f}")
        except Exception as e:
            logger.debug(f"⚠️ SENTIMENT: Error calculando score para cache: {e}")

    logger.info(f"🎯 SENTIMENT: Inferencia completada - {len(results)} resultados generados")
    return results

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("⏳ Descargando datos de sentimiento...")
    df_reddit = download_reddit()
    df_news = download_news()

    # Combinar
    df_all = pd.concat([df_reddit, df_news], ignore_index=True)
    df_all.dropna(subset=['text'], inplace=True)
    print(f"✅ Total de registros: {len(df_all)}")

    # Inferencia
    print("🧠 Ejecutando inferencia de sentimiento...")
    df_all['sentiment_probs'] = infer_sentiment(df_all['text'].tolist())
    df_all['predicted_class'] = df_all['sentiment_probs'].apply(lambda x: int(np.argmax(x)))

    # Guardar CSV de inferencia
    csv_path = os.path.join(DATA_DIR, f"sentiment_inference_{END_DATE.date()}.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"✅ CSV de inferencia guardado en '{csv_path}'")

    # Guardar JSON para L2
    json_path = os.path.join(DATA_DIR, f"sentiment_l2_{END_DATE.date()}.json")
    df_all.to_dict('records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(df_all.to_dict('records'), f, ensure_ascii=False, indent=2)
    print(f"✅ JSON listo para L2 guardado en '{json_path}'")
