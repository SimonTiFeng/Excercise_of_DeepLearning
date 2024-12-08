import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_data():
    dataset = load_dataset("ag_news")
    df = pd.DataFrame(dataset["train"])
    df["label"] = df["label"]  
    return df

data = load_data()
data = data.groupby("label").apply(lambda x: x.sample(1250)).reset_index(drop=True)

plt.bar(data["label"].value_counts().index, data["label"].value_counts().values)
plt.xlabel("Class")
plt.ylabel("Sample Count")
plt.title("Class Distribution")
plt.show()

data["text_length"] = data["text"].apply(len)
data["text_length"].hist(bins=50)
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.title("Text Length Distribution")
plt.show()

from collections import Counter
from itertools import combinations

def calculate_overlap(data):
    tokenized_texts = data.groupby("label")["text"].apply(lambda x: " ".join(x).split())
    vocab_sets = [set(tokens) for tokens in tokenized_texts]
    overlaps = {
        f"{i}-{j}": len(vocab_sets[i] & vocab_sets[j]) / len(vocab_sets[i] | vocab_sets[j])
        for i, j in combinations(range(len(vocab_sets)), 2)
    }
    return overlaps

vocab_overlaps = calculate_overlap(data)
plt.bar(vocab_overlaps.keys(), vocab_overlaps.values())
plt.xlabel("Class Pairs")
plt.ylabel("Vocabulary Overlap")
plt.title("Vocabulary Overlap Between Classes")
plt.show()

class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
)
train_dataset = AGNewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len=128)
test_dataset = AGNewsDataset(test_texts.tolist(), test_labels.tolist(), tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=3):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accs.append(correct / len(train_loader.dataset))

        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()

                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_losses.append(val_loss / len(test_loader))
        val_accs.append(val_correct / len(test_loader.dataset))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs, all_preds, all_labels

train_losses, val_losses, train_accs, val_accs, all_preds, all_labels = train_model(
    model, train_loader, test_loader, optimizer, criterion, epochs=3
)

report = classification_report(all_labels, all_preds, target_names=["World", "Sports", "Business", "Tech"], digits=4)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)
