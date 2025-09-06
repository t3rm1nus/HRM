import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Ruta de tus datos
data_dir = "data/datos_para_modelos_l3/sentiment"
files = ["tweets.csv", "reddit.csv", "news.csv"]

dfs = []
for file in files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Crear columna 'text' si no existe
        if 'text' not in df.columns:
            if 'title' in df.columns and 'content' in df.columns:
                df['text'] = df['title'].astype(str) + '. ' + df['content'].astype(str)
            else:
                print(f"⚠️ {file} no tiene 'text' ni 'title'+'content', se salta.")
                continue
        # Crear columna 'label' a partir de 'score' si no existe
        if 'label' not in df.columns:
            if 'score' in df.columns:
                df['label'] = df['score'].apply(lambda x: 1 if x > 0 else 0)
            else:
                print(f"⚠️ {file} no tiene 'label' ni 'score', se salta.")
                continue
        dfs.append(df[['text', 'label']])
    else:
        print(f"⚠️ {file} no encontrado, se salta.")

if not dfs:
    raise ValueError("No hay datos válidos para entrenar.")

data = pd.concat(dfs, ignore_index=True)
print(f"Se cargaron {len(data)} textos para entrenar.")

# Tokenizer y modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenizar textos
train_encodings = tokenizer(list(data['text']), truncation=True, padding=True, max_length=128)

# Dataset compatible con PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, list(data['label']))

# Entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_dir='./logs',
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# Guardar modelo final en models/L3/sentiment
save_dir = "models/L3/sentiment"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Modelo guardado en {save_dir}")
