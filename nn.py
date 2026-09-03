from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Load Dataset
dataset = load_dataset("Abirate/english_quotes")

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Apply LoRA
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, config)

# Training Settings
args = TrainingArguments(
    output_dir="model",
    num_train_epochs=1,
    per_device_train_batch_size=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"]
)

# Start Fine-tuning
trainer.train()


model.save_pretrained("lora_model")