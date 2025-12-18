"""
Script SIMPLE para consultar tu modelo LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# ===== CONFIGURACIÓN =====
RUTA_ADAPTADORES = "."
MODELO_BASE = "mistralai/Mistral-7B-Instruct-v0.2"

# ===== CARGAR MODELO (solo una vez) =====
print("🔧 Cargando modelo...")
print("⏳ Esto tomará 1-2 minutos...\n")

tokenizer = AutoTokenizer.from_pretrained(RUTA_ADAPTADORES)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="offload",
    load_in_8bit=True
)

model = PeftModel.from_pretrained(base_model, RUTA_ADAPTADORES)
model.eval()

print("✅ Modelo listo!\n")
print("⚠️ AVISO: Cada respuesta tardará ~1-3 minutos en CPU")
print("=" * 60)

# ===== FUNCIÓN SIMPLE =====
def preguntar(texto):
    prompt = f"""Eres un tutor experto en algoritmos.
Explica de forma clara y paso a paso.

Instrucción: {texto}
Respuesta:
"""
    
    print("\n⏳ Generando respuesta (puede tardar 1-3 min)...", end="", flush=True)
    inicio = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Reducido para que sea más rápido
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta = respuesta.split("Respuesta:")[-1].strip()
    
    tiempo = time.time() - inicio
    print(f"\r⏱️ Generado en {tiempo:.1f} segundos\n")
    
    return respuesta

# ===== USAR =====
while True:
    pregunta = input("\n💬 Pregunta (o 'salir'): ")
    
    if pregunta.lower() in ['salir', 'exit', 'quit']:
        print("👋 ¡Adiós!")
        break
    
    if pregunta.strip():
        respuesta = preguntar(pregunta)
        print(f"\n🤖 {respuesta}\n")
        print("=" * 60)