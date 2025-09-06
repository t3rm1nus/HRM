import pandas as pd
import os

data_folder = 'data/datos_para_modelos_l3/sentiment'
reddit_file = os.path.join(data_folder, 'reddit.csv')
news_file = os.path.join(data_folder, 'news.csv')

df_reddit = pd.read_csv(reddit_file)
df_news = pd.read_csv(news_file)

# Asegurarse de que ambas tengan columna 'text'
df_reddit = df_reddit.rename(columns={'post': 'text'}) if 'post' in df_reddit.columns else df_reddit
df_news = df_news.rename(columns={'title': 'text'}) if 'title' in df_news.columns else df_news

df = pd.concat([df_reddit, df_news], ignore_index=True)
df.to_csv(os.path.join(data_folder, 'sentiment_data.csv'), index=False)
print("✅ CSV combinado creado en 'sentiment_data.csv'")
