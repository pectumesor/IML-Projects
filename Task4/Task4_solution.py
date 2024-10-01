import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.model_selection import train_test_split

torch.mps.set_per_process_memory_fraction(0.0)
LEARNING_RATE = 0.00001
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 30
CHECKPOINT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

train_val = pd.read_csv("train.csv")
train, val = train_test_split(train_val, test_size=0.2, random_state=42)
test_val = pd.read_csv("test_no_score.csv")

train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test_val)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


def tokenize_function(data):
    return tokenizer(data['title'], data['sentence'], truncation=True, max_length=MAX_LEN, padding='max_length')


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


tokenized_train = tokenized_train.remove_columns(["title", "sentence","__index_level_0__"])
tokenized_train = tokenized_train.rename_column("score", "labels")
tokenized_val = tokenized_val.remove_columns(["title", "sentence","__index_level_0__"])
tokenized_val = tokenized_val.rename_column("score", "labels")
tokenized_test = tokenized_test.remove_columns(["title", "sentence"])
tokenized_train.set_format("torch")
tokenized_val.set_format("torch")
tokenized_test.set_format("torch")

train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_val, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_test, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)

#Classification model with 1 label is equivalent to regression
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=1, ignore_mismatched_sizes=True).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_steps
)

progress_bar = tqdm(range(num_steps))
best_acc = 1000000

for epoch in range(EPOCHS):
    for batch in train_dataloader:
        batch_inputs, batch_masks, batch_labels = (batch["input_ids"].to(DEVICE),
                                                   batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE))
        model.zero_grad()
        outputs = model(batch_inputs, batch_masks)
        logits = outputs[0][:,0]
        loss = criterion(logits, batch_labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in eval_dataloader:
            batch_inputs, batch_masks, batch_labels = (batch["input_ids"].to(DEVICE),
                                                       batch["attention_mask"].to(DEVICE), batch["labels"].to(DEVICE))
            outputs = model(batch_inputs, batch_masks)
            logits = outputs[0][:, 0]
            loss = criterion(logits, batch_labels)
            total_loss += loss.item()
        if total_loss / len(eval_dataloader) < best_acc:
            print(f"Found a new model at Epoch: {epoch + 1} \n")
            best_acc = total_loss / len(eval_dataloader)
            #save the best weights with the best accuracy so far
            torch.save(model.state_dict(), 'best-model-parameters.pt')
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(eval_dataloader)} \n")
#load the weights with the best accuracy
model.load_state_dict(torch.load('best-model-parameters.pt'))
model.eval()
results = []
for batch in test_dataloader:
    batch_inputs, batch_masks = (batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
    model.zero_grad()
    with torch.no_grad():
        outputs = model(batch_inputs, batch_masks)
    predictions = outputs
    logits = outputs[0][:, 0]
    predicted = logits.cpu().numpy()
    results.append(predicted)

with open("result.txt", "w") as f:
    for val in np.concatenate(results):
        if val < 0:
            val = 0
        elif val > 10:
            val = 10
        f.write(f"{val}\n")