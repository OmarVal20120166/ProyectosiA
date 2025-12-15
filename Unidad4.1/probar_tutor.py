import argparse, json, re, math
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEMS = {
    "bigtech": (
        "Eres un ingeniero de software en una Big Tech. "
        "Explica para qué sirve el algoritmo con un ejemplo real. "
        "Sé claro, concreto y sin muletillas.\n\n"
    ),
    "socratico": (
        "Eres un tutor socrático de algoritmos. "
        "NO des la solución completa. "
        "Haz preguntas guía y da pistas graduales. "
        "Evita código completo.\n\n"
    ),
    "general": (
        "Eres un tutor de algoritmos. "
        "Responde claro y útil sin dar la solución completa.\n\n"
    ),
}

def build_prompt(mode, instruction, inp=""):
    sys = SYSTEMS.get(mode, SYSTEMS["general"])
    return (
        sys
        + "### Instruction:\n" + instruction.strip() + "\n"
        + "### Input:\n" + (inp.strip() if inp else "") + "\n"
        + "### Response:\n"
    )

def has_code(txt: str) -> bool:
    if "```" in txt:
        return True
    # heurística simple
    return len(re.findall(r"\b(def|class|for|while|return|import)\b", txt)) >= 4

def distinct_2(txt: str) -> float:
    toks = re.findall(r"\w+|[^\w\s]", txt.lower())
    if len(toks) < 3:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return len(set(bigrams)) / max(1, len(bigrams))

def repetition_rate(txt: str) -> float:
    toks = re.findall(r"\w+", txt.lower())
    if not toks:
        return 0.0
    return 1.0 - (len(set(toks)) / len(toks))

@torch.no_grad()
def response_logprob(model, tokenizer, prompt, response, max_length):
    """
    Calcula logprob promedio SOLO en tokens de respuesta (como tu masking).
    Esto te da una métrica tipo "qué tan probable" ve el modelo su propia respuesta.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    resp_ids = tokenizer(response, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (prompt_ids + resp_ids)[:max_length]

    input_ids = torch.tensor([ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0]  # [seq, vocab]

    # logprob token a token (predict next token)
    # tokens evaluables: desde len(prompt_ids) hasta len(ids)-1 (porque el último no tiene next)
    start = min(len(prompt_ids), len(ids) - 1)
    end = len(ids) - 1
    if end <= start:
        return None

    logprobs = []
    for t in range(start, end):
        next_id = ids[t + 1]
        lp = torch.log_softmax(logits[t], dim=-1)[next_id].item()
        logprobs.append(lp)

    return sum(logprobs) / len(logprobs)

def main():
    ap = argparse.ArgumentParser("Evaluación rápida del tutor (LoRA)")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora", default="./tutor_algoritmos_lora_v2")
    ap.add_argument("--eval_set", default="eval_set_v3.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--compute_logprob", action="store_true")
    ap.add_argument("--out_report", default="eval_report.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    rows = []
    with open(args.eval_set, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    for ex in eval_items:
        prompt = build_prompt(ex.get("mode","general"), ex["instruction"], ex.get("input",""))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

        row = {
            "id": ex.get("id"),
            "mode": ex.get("mode"),
            "len_chars": len(gen),
            "has_code": has_code(gen),
            "distinct2": distinct_2(gen),
            "repetition": repetition_rate(gen),
            "generated": gen,
            "reference": ex.get("reference","")
        }

        if args.compute_logprob:
            lp = response_logprob(model, tokenizer, prompt, gen, args.max_length)
            row["avg_logprob"] = lp

        rows.append(row)

    # Resumen
    n = len(rows)
    modes = Counter([r["mode"] for r in rows])
    code_rate = sum(1 for r in rows if r["has_code"]) / max(1, n)
    avg_len = sum(r["len_chars"] for r in rows) / max(1, n)
    avg_dist2 = sum(r["distinct2"] for r in rows) / max(1, n)
    avg_rep = sum(r["repetition"] for r in rows) / max(1, n)

    report = {
        "n": n,
        "by_mode": dict(modes),
        "code_rate": code_rate,
        "avg_len_chars": avg_len,
        "avg_distinct2": avg_dist2,
        "avg_repetition": avg_rep,
        "rows": rows
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ Reporte guardado en:", args.out_report)
    print("📊 by_mode:", dict(modes))
    print("🧪 code_rate:", round(code_rate, 3),
          "| avg_len_chars:", round(avg_len, 1),
          "| avg_distinct2:", round(avg_dist2, 3),
          "| avg_repetition:", round(avg_rep, 3))

    if args.compute_logprob:
        vals = [r.get("avg_logprob") for r in rows if r.get("avg_logprob") is not None]
        if vals:
            print("🔍 avg_logprob (más alto=mejor):", round(sum(vals)/len(vals), 4))

if __name__ == "__main__":
    main()
