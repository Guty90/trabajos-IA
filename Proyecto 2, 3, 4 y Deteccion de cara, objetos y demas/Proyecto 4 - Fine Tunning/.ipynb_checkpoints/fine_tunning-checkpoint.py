import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# -----------------------------
# 1. Cargar dataset
# -----------------------------
dataset = load_dataset("json", data_files="tutor_programacion.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)

# -----------------------------
# 2. Cargar modelo base
# -----------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


# -----------------------------
# 3. Configurar LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 4. Preprocesar el dataset
# -----------------------------
def format_instruction(example):
    prompt = (
        "Eres un tutor experto en algoritmos.\n"
        "Explica de forma clara y paso a paso.\n\n"
        f"Instrucción: {example['instruction']}\n"
        "Respuesta:\n"
    )
    return tokenizer(
        prompt + example["response"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


tokenized = dataset.map(format_instruction)

# -----------------------------
# 5. Entrenamiento
# -----------------------------
training_args = TrainingArguments(
    output_dir="./lora-tutor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=50,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


trainer.train()

# -----------------------------
# 6. Guardar adaptadores LoRA
# -----------------------------
model.save_pretrained("./lora-tutor")
print("Entrenamiento completado. Adaptadores guardados.")