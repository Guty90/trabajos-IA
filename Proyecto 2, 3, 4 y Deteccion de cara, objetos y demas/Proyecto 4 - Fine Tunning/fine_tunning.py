# ===== INSTALACIÓN DE DEPENDENCIAS =====
# !pip install -q transformers datasets peft accelerate bitsandbytes

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== VERIFICAR GPU =====
print(f"🔥 GPU disponible: {torch.cuda.is_available()}")
print(f"🎯 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# ===== CARGAR DATASET =====
dataset = load_dataset("json", data_files="tutor_programacion.jsonl")
dataset = dataset["train"].train_test_split(test_size=0.1)
print(f"📊 Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

# ===== CARGAR MODELO =====
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

# ===== CONFIGURAR LORA =====
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== PREPROCESAR DATASET =====
def format_instruction(example):
    prompt = (
        "Eres un tutor experto en algoritmos.\n"
        "Explica de forma clara y paso a paso.\n\n"
        f"Instrucción: {example['instruction']}\n"
        "Respuesta:\n"
    )
    full_text = prompt + example["response"] + tokenizer.eos_token
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Enmascarar el prompt
    prompt_length = len(tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
    tokenized["labels"][:prompt_length] = [-100] * prompt_length
    
    return tokenized

tokenized = dataset.map(
    format_instruction,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizando dataset"
)

# ===== CONFIGURAR ENTRENAMIENTO (CORREGIDO) =====
training_args = TrainingArguments(
    output_dir="./lora-tutor",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    
    logging_steps=10,
    save_steps=50,
    eval_steps=50,  # ✅ Cambió de evaluation_strategy
    eval_strategy="steps",  # ✅ ESTE ES EL CAMBIO IMPORTANTE
    save_total_limit=2,
    
    warmup_steps=50,
    weight_decay=0.01,
    
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ===== ENTRENAR =====
print("\n🚀 Iniciando entrenamiento...\n")
trainer.train()

# ===== GUARDAR =====
model.save_pretrained("./lora-tutor-final")
tokenizer.save_pretrained("./lora-tutor-final")
print("\n✅ Modelo guardado en ./lora-tutor-final/")

# ===== PROBAR =====
print("\n🧪 Probando el modelo...\n")
model.eval()

test_prompt = """Eres un tutor experto en algoritmos.
Explica de forma clara y paso a paso.

Instrucción: ¿Qué es una lista enlazada?
Respuesta:
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("\n" + "="*50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("="*50)