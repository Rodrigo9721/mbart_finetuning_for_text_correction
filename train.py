import torch
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import gc

gc.collect()
torch.cuda.empty_cache()

# Clase CustomDataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]['incorrect'], self.dataframe.iloc[idx]['correct']

# Función para convertir ejemplos en características
def convert_examples_to_features(examples, tokenizer, max_length=512):
    features = []
    for example in examples:
        input_text = example[0]
        target_text = example[1]
        inputs = tokenizer.encode_plus(input_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        targets = tokenizer.encode_plus(target_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

        features.append({
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten()
        })
    return features

# Leer el archivo JSON y cargar los pares de oraciones
data = pd.read_json('data.json')

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_data)
val_dataset = CustomDataset(val_data)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", model_max_length=512)

encoded_train_dataset = convert_examples_to_features(train_dataset, tokenizer, max_length=512)
encoded_val_dataset = convert_examples_to_features(val_dataset, tokenizer, max_length=512)

class CustomTorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = CustomTorchDataset(encoded_train_dataset)
val_dataset = CustomTorchDataset(encoded_val_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    evaluation_strategy="steps",
    logging_dir="./logs",
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=2,
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Guarda el modelo
trainer.save_model("./trained_model_es")
