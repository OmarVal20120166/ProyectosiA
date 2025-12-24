import json
import re
from difflib import SequenceMatcher

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./tutor_algoritmos_lora_v3"
DATASET_PATH = "dataset_final_v3_with_source.jsonl"

PREFIX = "Va, del temario:"

# Candados de similitud
MATCH_THRESHOLD = 0.62   # mínimo aceptable
MARGIN = 0.08            # diferencia entre top1 y top2

# ============================================================
# UTILIDADES: normalización y similitud
# ============================================================

def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def best_two_matches(rows, query: str):
    q = normalize(query)
    scored = []

    for row in rows:
        instr = normalize(row.get("instruction", ""))
        inp = normalize(row.get("input", ""))
        combined = (instr + " " + inp).strip()
        score = similarity(q, combined)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    top1 = scored[0] if scored else (0.0, None)
    top2 = scored[1] if len(scored) > 1 else (0.0, None)
    return top1, top2

# ============================================================
# CARGA DE DATASET
# ============================================================

print("Cargando dataset...")
DATASET_ROWS = load_jsonl(DATASET_PATH)
print(f"Dataset cargado: {len(DATASET_ROWS)} ejemplos")

# ============================================================
# CARGA DE MODELO (solo para consistencia del proyecto)
# NOTA: NO se usa para generar contenido
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ============================================================
# CHAT LOOP (RESPUESTA 100% DATASET)
# ============================================================

print("\n🧠 Tutor listo")
print("Escribe 'salir' para terminar.\n")

while True:
    user = input("Tú: ").strip()

    if not user:
        continue

    if user.lower() in ["salir", "exit", "quit"]:
        break

    # Buscar top1 y top2
    (score1, row1), (score2, row2) = best_two_matches(DATASET_ROWS, user)

    # Candado duro
    if (
        row1 is None
        or score1 < MATCH_THRESHOLD
        or (score1 - score2) < MARGIN
    ):
        print("\nTutor: No tengo esa información en el dataset.\n")
        continue

    output = (row1.get("output", "") or "").strip()
    pref = (row1.get("preferred_prefix", PREFIX) or PREFIX).strip()

    if not output:
        print("\nTutor: No tengo esa información en el dataset.\n")
        continue

    # Respuesta literal del dataset (sin inventar)
    if output.startswith(pref):
        answer = output
    else:
        answer = f"{pref} {output}"

    print(f"\nTutor: {answer}\n")
