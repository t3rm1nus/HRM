import os
import pandas as pd
from datetime import datetime, timedelta
import snscrape.modules.twitter as sntwitter
import requests
import asyncpraw
from dotenv import load_dotenv

# ======== CONFIG ========
load_dotenv()
OUTPUT_DIR = "data/datos_para_modelos_l3/sentiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fechas (últimos 90 días)
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=90)

# 🔹 API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")




# =========================================================
# REDDIT con PRAW
# =========================================================
async def download_reddit(subreddits=["CryptoCurrency", "Bitcoin", "Ethereum"], limit=1000):
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        print("⚠️ Reddit API keys no configuradas en .env. Saltando Reddit.")
        return pd.DataFrame()

    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    posts = []
    try:
        for sub in subreddits:
            subreddit = await reddit.subreddit(sub)
            count = 0
            async for post in subreddit.hot(limit=limit):
                posts.append({
                    "date": datetime.fromtimestamp(post.created_utc),
                    "title": post.title,
                    "content": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "subreddit": sub,
                    "url": post.url
                })
                count += 1
                if count >= limit:
                    break
        df = pd.DataFrame(posts)
        path = os.path.join(OUTPUT_DIR, "reddit.csv")
        df.to_csv(path, index=False)
        print(f"✅ {len(df)} posts guardados en '{path}'")
        return df
    except Exception as e:
        print(f"⚠️ Error descargando Reddit: {e}")
        return pd.DataFrame()


# =========================================================
# NEWS (NewsAPI troceado en bloques de 30 días)
# =========================================================
def download_news(query="crypto OR bitcoin OR ethereum", start_date=START_DATE, end_date=END_DATE):
    if not NEWS_API_KEY:
        print("⚠️ NEWS_API_KEY no configurada en .env. Saltando noticias.")
        return pd.DataFrame()

    all_articles = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=29), end_date)
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&language=en&from={current_start.date()}&to={current_end.date()}"
            f"&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        try:
            response = requests.get(url)
            data = response.json()
            if "articles" in data:
                for a in data["articles"]:
                    all_articles.append({
                        "date": a["publishedAt"],
                        "title": a["title"],
                        "description": a["description"],
                        "content": a["content"],
                        "source": a["source"]["name"],
                        "url": a["url"]
                    })
                print(f"📰 {len(data['articles'])} artículos entre {current_start.date()} y {current_end.date()}")
            else:
                print(f"⚠️ Error en bloque {current_start.date()} - {current_end.date()}: {data}")
        except Exception as e:
            print(f"⚠️ Error descargando noticias: {e}")

        current_start = current_end + timedelta(days=1)

    df = pd.DataFrame(all_articles)
    path = os.path.join(OUTPUT_DIR, "news.csv")
    df.to_csv(path, index=False)
    print(f"✅ {len(df)} artículos guardados en '{path}'")
    return df


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("⏳ Descargando datos de sentimiento...")

    import asyncio
    # df_twitter = await download_twitter() # Si tienes una versión asíncrona, usa await
    df_reddit = asyncio.run(download_reddit())
    df_news = download_news()

    print("\n📊 Resumen de datos descargados:")
    print(f"  - twitter: {len(df_twitter)} registros")
    print(f"  - reddit: {len(df_reddit)} registros")
    print(f"  - news: {len(df_news)} registros")
