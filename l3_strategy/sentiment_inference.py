import os
import json
import pandas as pd
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from dotenv import load_dotenv
import asyncpraw
import requests

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
async def download_reddit(subreddits=["CryptoCurrency", "Bitcoin", "Ethereum"], limit=1000):
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        print("⚠️ Reddit API keys no configuradas. Saltando Reddit.")
        return pd.DataFrame()

    reddit = asyncpraw.Reddit(client_id=REDDIT_CLIENT_ID,
                              client_secret=REDDIT_CLIENT_SECRET,
                              user_agent=REDDIT_USER_AGENT)
    posts = []
    for sub in subreddits:
        try:
            subreddit = await reddit.subreddit(sub)
            count = 0
            async for post in subreddit.hot(limit=limit):
                posts.append({
                    "date": datetime.fromtimestamp(post.created_utc),
                    "text": f"{post.title} {post.selftext}"
                })
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            print(f"⚠️ Error descargando Reddit {sub}: {e}")
    return pd.DataFrame(posts)

def download_news(query="crypto OR bitcoin OR ethereum"):
    if not NEWS_API_KEY:
        print("⚠️ NEWS_API_KEY no configurada. Saltando noticias.")
        return pd.DataFrame()

    all_articles = []
    current_start = START_DATE
    while current_start < END_DATE:
        current_end = min(current_start + timedelta(days=29), END_DATE)
        url = (
            f"https://newsapi.org/v2/everything?q={query}&language=en"
            f"&from={current_start.date()}&to={current_end.date()}"
            f"&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        try:
            response = requests.get(url)
            data = response.json()
            if "articles" in data:
                for a in data["articles"]:
                    all_articles.append({
                        "date": a["publishedAt"],
                        "text": a.get("title","") + " " + str(a.get("content",""))
                    })
            current_start = current_end + timedelta(days=1)
        except Exception as e:
            print(f"⚠️ Error descargando noticias: {e}")
            current_start = current_end + timedelta(days=1)
    return pd.DataFrame(all_articles)

# =========================
# INFERENCIA
# =========================
def infer_sentiment(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1).tolist()
        results.extend(probs)
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
