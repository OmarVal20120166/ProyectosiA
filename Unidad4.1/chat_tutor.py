import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# -----------------------------
# Configuración
# -----------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./tutor_algoritmos_lora_v3"   # ajusta si tu carpeta es distinta

SYSTEM_BY_MODE = {
    "bigtech": (
        "Eres un ingeniero de software en una Big Tech. "
        "Explica para qué sirve el algoritmo con ejemplos reales (apps, sistemas). "
        "Sé claro, concreto y sin muletillas."
    ),
    "socratico": (
        "Eres un tutor socrático de algoritmos. "
        "NO des la solución completa. "
        "Haz preguntas guía y da pistas graduales."
    ),
    "general": (
        "Eres un tutor de algoritmos. "
        "Responde claro y útil sin dar la solución completa."
    ),
}

def build_prompt(mode: str, user_text: str) -> str:
    system = SYSTEM_BY_MODE.get(mode, SYSTEM_BY_MODE["general"])
    return (
        system + "\n\n"
        "### Instruction:\n" + user_text.strip() + "\n"
        "### Input:\n\n"
        "### Response:\n"
    )

# -----------------------------
# Carga de modelo
# -----------------------------
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()



# -----------------------------
# Loop interactivo
# -----------------------------
mode = "socratico"

print("\n🧠 Tutor listo.")
print("Comandos:")
print("  /modo bigtech")
print("  /modo socratico")
print("  /modo general")
print("  salir\n")

while True:
    user = input("Tú: ").strip()

    if user.lower() in ["salir", "exit", "quit"]:
        print("Hasta luego 👋")
        break

    # Cambio de modo
    if user.startswith("/modo"):
        parts = user.split()
        if len(parts) == 2 and parts[1] in SYSTEM_BY_MODE:
            mode = parts[1]
            print(f"✅ Modo cambiado a: {mode}\n")
        else:
            print("❌ Modo inválido. Usa: bigtech | socratico | general\n")
        continue

    prompt = build_prompt(mode, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )

    print()
