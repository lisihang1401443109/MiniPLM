from datasets import load_dataset, Dataset

ds = load_dataset("JeanKaddour/minipile", split="train")

# num_samples_to_take = int(0.1 * len(ds))
num_samples_to_take = 10000
ds1 = ds.take(num_samples_to_take)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

from transformers import Qwen2Config, Qwen2Model, Qwen2Tokenizer
import os

# Define configuration for ~100M parameters
config = Qwen2Config(
    hidden_size=1024,          # Size of the hidden layers
    intermediate_size=5504,   # Size of the feed-forward layers
    num_hidden_layers=12,     # Number of transformer blocks
    num_attention_heads=8,    # Number of attention heads,  # Maximum sequence length
)

# Initialize the model with the defined configuration
model = Qwen2Model(config)
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Directory to save the model
save_directory = "./checkpoints/qwen_100m"
os.makedirs(save_directory, exist_ok=True)


def tokenize_function(examples):
    # Tokenize each sample's text (adjust field name if needed)
    return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length, padding="max_length")

train_list = list(ds1)
train_dataset = Dataset.from_list(train_list)
# train_dataset = ds1


train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)




training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,            # adjust number of epochs as needed
    per_device_train_batch_size=8,   # adjust based on your GPU memory
    save_steps=1000,
    save_total_limit=2,
    logging_steps=500,
    prediction_loss_only=True,
)

# --- Step 6: Initialize Trainer and train the model ---
trainer = Trainer(
    model=model.cuda(),
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()


# Save model and configuration
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)