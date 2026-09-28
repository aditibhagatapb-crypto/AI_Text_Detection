from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset


# --------------------------------------------------
# 1. MODEL
# --------------------------------------------------

MODEL_NAME = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

# Example dataset structure:
# label = 0 -> Human-written
# label = 1 -> AI-generated

dataset = load_dataset("fakes_news_detection")


# --------------------------------------------------
# 3. PREPARE TEXT
# --------------------------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# 4. LOAD ROBERTA
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# --------------------------------------------------
# 5. TRAINING SETTINGS
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir="./model",

    eval_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_dir="./logs",

    save_strategy="epoch",

    load_best_model_at_end=True
)


# --------------------------------------------------
# 6. TRAINER
# --------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=tokenized_dataset["test"],

    tokenizer=tokenizer
)


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nStarting RoBERTa training...\n")

trainer.train()


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

trainer.save_model("./model")

tokenizer.save_pretrained("./model")

print("\nTraining completed successfully.")

print("Model saved inside the 'model' folder.")