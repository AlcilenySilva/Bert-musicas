
!pip install torch transformers datasets scikit-learn pandas



import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Desativar W&B
os.environ["WANDB_DISABLED"] = "true"

# Carregar os arquivos CSV do Colab
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")

# Criar cópias para evitar FutureWarning
train_df = train_df.copy()
test_df = test_df.copy()

# Tratar valores ausentes
train_df["explicit"] = train_df["explicit"].fillna(0).astype(int)
test_df["explicit"] = test_df["explicit"].fillna(0).astype(int)

# Definir o tokenizer
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Preparação do dataset
class MusicDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = list(dataframe["lyrics"].astype(str))
        self.labels = list(dataframe["explicit"].astype(int))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return {**{key: val.squeeze(0) for key, val in encoding.items()}, "labels": torch.tensor(label, dtype=torch.long)}

# Criar datasets
train_dataset = MusicDataset(train_df)
test_dataset = MusicDataset(test_df)

# Criar modelo BERT
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ajustes nos hiperparametros
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# Função de avaliação
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Criar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Treinar o modelo
trainer.train()

# Avaliar no conjunto de teste
metrics = trainer.evaluate()
print(metrics)
print("\n===== Resultados de Avaliação =====")
print(f"Epoch: {metrics['epoch']:.1f}")
print(f"Loss de Validação: {metrics['eval_loss']:.4f}")
print(f"Acurácia: {metrics['eval_accuracy'] * 100:.2f}%")
print(f"Precisão: {metrics['eval_precision'] * 100:.2f}%")
print(f"Recall: {metrics['eval_recall'] * 100:.2f}%")
print(f"F1 Score: {metrics['eval_f1'] * 100:.2f}%")
print(f"Tempo de Execução: {metrics['eval_runtime']:.2f} segundos")
print(f"Amostras por Segundo: {metrics['eval_samples_per_second']:.2f}")
print(f"Passos por Segundo: {metrics['eval_steps_per_second']:.2f}")
