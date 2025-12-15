import json, re, argparse, random
from collections import Counter, defaultdict
from hashlib import md5

FILLERS_RE = re.compile(
    r'^(claro[, ]+)?(mi\s+querid[oa]\s+\w+[, ]*)',
    flags=re.IGNORECASE
)

def detect_mode(inst: str) -> str:
    t = (inst or "").lower()
    if "vida real" in t or "para qué sirve" in t or "para que sirve" in t:
        return "bigtech"
    if any(k in t for k in ["no sé", "no se", "atascad", "cómo empiezo", "como empiezo",
                            "ayuda", "pista", "guíame", "guia", "no entiendo"]):
        return "socratico"
    return "general"

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    # normaliza espacios
    s = re.sub(r"\s+", " ", s).strip()
    # arregla espacios antes de puntuación
    s = re.sub(r"\s+([?.!,;:])", r"\1", s)
    # asegura espacio después de puntuación si está pegada a palabra
    s = re.sub(r"([?.!,;:])([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])", r"\1 \2", s)
    return s.strip()

def clean_output(o: str) -> str:
    o = normalize_text(o)
    # quita muletillas al inicio tipo "Claro mi querido Piltillo..."
    o = FILLERS_RE.sub("", o).strip()
    o = re.sub(r"^claro[, ]*", "", o, flags=re.IGNORECASE).strip()
    # evita que quede empezando con coma
    o = re.sub(r"^[, ]+", "", o).strip()
    return o

def norm_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[¿?¡!.,:;\"'()\[\]]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def stable_hash(s: str) -> str:
    return md5((s or "").encode("utf-8")).hexdigest()[:10]

def main():
    ap = argparse.ArgumentParser("Rebuild v3: limpieza + eval_set")
    ap.add_argument("--in_path", default="dataset_final.jsonl")
    ap.add_argument("--out_path", default="dataset_final_v3_cleaned.jsonl")
    ap.add_argument("--eval_out", default="eval_set_v3.jsonl")
    ap.add_argument("--dedup_by_instruction", action="store_true")
    ap.add_argument("--min_output_chars", type=int, default=25)
    ap.add_argument("--eval_size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    items = []
    dropped_short = 0

    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            inst = normalize_text(obj.get("instruction", ""))
            inp = normalize_text(obj.get("input", ""))
            out = clean_output(obj.get("output", ""))

            if len(out) < args.min_output_chars:
                dropped_short += 1
                continue

            mode = detect_mode(inst)

            items.append({
                "id": stable_hash(inst + "||" + out),
                "mode": mode,
                "instruction": inst,
                "input": inp,
                "output": out
            })

    if args.dedup_by_instruction:
        dedup = {}
        for it in items:
            k = norm_key(it["instruction"])
            if k not in dedup:
                dedup[k] = it
        items = list(dedup.values())

    # Reporte rápido
    modes = Counter([i["mode"] for i in items])

    # --- construir eval_set estratificado por modo ---
    by_mode = defaultdict(list)
    for it in items:
        by_mode[it["mode"]].append(it)

    # objetivo: repartir eval_size entre modos proporcional, mínimo 5 si existe
    eval_items = []
    total = sum(len(v) for v in by_mode.values())
    for mode, group in by_mode.items():
        if not group:
            continue
        target = max(5, int(args.eval_size * (len(group) / total)))
        random.shuffle(group)
        eval_items.extend(group[:target])

    # recorta exacto eval_size y evita duplicados por id
    seen = set()
    final_eval = []
    for it in eval_items:
        if it["id"] in seen:
            continue
        seen.add(it["id"])
        final_eval.append(it)
    final_eval = final_eval[:args.eval_size]

    # guardar dataset limpio
    with open(args.out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # guardar eval_set (solo prompt + referencia)
    with open(args.eval_out, "w", encoding="utf-8") as f:
        for it in final_eval:
            f.write(json.dumps({
                "id": it["id"],
                "mode": it["mode"],
                "instruction": it["instruction"],
                "input": it["input"],
                "reference": it["output"]
            }, ensure_ascii=False) + "\n")

    print("✅ Dataset limpio:", args.out_path, "| ejemplos:", len(items))
    print("✅ Eval set:", args.eval_out, "| ejemplos:", len(final_eval))
    print("🧹 Eliminados por output muy corto:", dropped_short)
    print("📊 Por modo:", dict(modes))

if __name__ == "__main__":
    main()
