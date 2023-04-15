from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, AutoTokenizer

from datasets import load_dataset

import pandas as pd
df = pd.read_csv('ner.csv')
df.head()
print(df.read())

with open('train.csv', 'r') as f:
    print(f.read())

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# loading the model and specify the number of expected labels
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# Specify where to save the checkpoints from the training:
training_args = TrainingArguments(output_dir="test_trainer")
# The evaluate library provides a simple accuracy function you can load with the evaluate.load function:
# TODO

metric = evaluate.load("accuracy")


# Call computing on metric to calculate the accuracy of your predictions.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# monitor your evaluation metrics during fine-tuning


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# Create a Trainer object with your model, training arguments, training and test datasets, and evaluation function:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
print("开始训练")
trainer.train()
print("训练结束")
