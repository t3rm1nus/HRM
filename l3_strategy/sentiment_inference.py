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

# =========================
# DESCARGA DE DATOS
# =========================
async def download_reddit(subreddits=["CryptoCurrency", "Bitcoin", "Ethereum"], limit=500):
    logger.info(f"🔄 SENTIMENT: Iniciando descarga de Reddit - Subreddits: {subreddits}, Limit: {limit}")

    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        logger.warning("⚠️ SENTIMENT: Reddit API keys no configuradas - Saltando descarga de Reddit")
        return pd.DataFrame()

    logger.info("🔑 SENTIMENT: Reddit API keys configuradas - Iniciando conexión")
    reddit = asyncpraw.Reddit(client_id=REDDIT_CLIENT_ID,
                              client_secret=REDDIT_CLIENT_SECRET,
                              user_agent=REDDIT_USER_AGENT)

    posts = []
    total_posts = 0

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

        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error descargando r/{sub}: {e}")

    logger.info(f"📊 SENTIMENT: Reddit total descargado: {total_posts} posts de {len(subreddits)} subreddits")
    return pd.DataFrame(posts)

def download_news(query="crypto OR bitcoin OR ethereum OR blockchain"):
    logger.info(f"📰 SENTIMENT: Iniciando descarga de noticias - Query: '{query}'")

    if not NEWS_API_KEY:
        logger.warning("⚠️ SENTIMENT: NEWS_API_KEY no configurada - Saltando descarga de noticias")
        return pd.DataFrame()

    logger.info("🔑 SENTIMENT: News API key configurada - Iniciando descarga")

    all_articles = []
    total_articles = 0
    current_start = START_DATE

    # Limitar a los últimos 7 días para rendimiento (en lugar de 90 días)
    START_DATE_LIMITED = END_DATE - timedelta(days=7)
    current_start = max(current_start, START_DATE_LIMITED)

    logger.info(f"📅 SENTIMENT: Descargando noticias desde {current_start.date()} hasta {END_DATE.date()}")

    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=1), END_DATE)  # Un día a la vez para mejor control

        logger.debug(f"🕐 SENTIMENT: Descargando período {current_start.date()} - {current_end.date()}")

        url = (
            f"https://newsapi.org/v2/everything?q={query}&language=en"
            f"&from={current_start.date()}&to={current_end.date()}"
            f"&sortBy=publishedAt&pageSize=50&apiKey={NEWS_API_KEY}"  # Limitar a 50 artículos por día
        )

        try:
            response = requests.get(url, timeout=10)  # Timeout de 10 segundos
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
            else:
                logger.warning(f"⚠️ SENTIMENT: No se encontraron artículos para {current_start.date()}")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ SENTIMENT: Error HTTP descargando noticias {current_start.date()}: {e}")
        except Exception as e:
            logger.error(f"❌ SENTIMENT: Error procesando noticias {current_start.date()}: {e}")

        current_start = current_end + timedelta(days=1)

    logger.info(f"📊 SENTIMENT: News total descargado: {total_articles} artículos")
    return pd.DataFrame(all_articles)

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
