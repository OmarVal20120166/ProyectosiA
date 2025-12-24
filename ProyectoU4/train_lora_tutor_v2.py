import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser("LoRA fine-tune Tutor de Algoritmos (Qwen2.5) - Rápido y eficiente (padding dinámico)")
    p.add_argument("--dataset_path", type=str, default="dataset_final_v3_cleaned.jsonl")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="./tutor_algoritmos_lora_v3")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    return p.parse_args()


SYSTEM_BY_MODE = {
    "bigtech": (
        "Eres un ingeniero de software en una Big Tech. "
        "Explica para qué sirve el algoritmo con un ejemplo real. "
        "Sé claro y sin muletillas."
    ),
    "socratico": (
        "Eres un tutor socrático de algoritmos. "
        "NO des la solución completa. "
        "Haz preguntas guía y da pistas graduales. "
        "Evita código completo."
    ),
    "general": (
        "Eres un tutor de algoritmos. "
        "Responde claro y útil sin dar la solución completa."
    ),
}


def build_prompt(mode, instruction, inp):
    mode = (mode or "general").strip().lower()
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    system = SYSTEM_BY_MODE.get(mode, SYSTEM_BY_MODE["general"])

    prompt = system + "\n\n"
    prompt += "### Instruction:\n" + instruction + "\n"
    prompt += "### Input:\n" + (inp if inp else "") + "\n"
    prompt += "### Response:\n"
    return prompt


def main():
    # Evita warnings/bugs de paralelismo de tokenizers en Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Boost NVIDIA (Ada/Ampere): acelera matmuls con pérdida mínima
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    args = parse_args()
    print("=== Configuración ===")
    print(vars(args))
    print("=====================")

    raw = load_dataset("json", data_files=args.dataset_path, split="train")
    split = raw.train_test_split(test_size=args.eval_ratio, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # En 4060 Ti es mejor fp16 para velocidad/compatibilidad
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Recomendado para LoRA
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # -----------------------------
    # Tokenización + masking (SIN padding fijo)
    # -----------------------------
    def tokenize_and_mask(batch):
        instructions = batch["instruction"]
        inputs_ = batch.get("input", [""] * len(instructions))
        outputs = batch["output"]
        modes = batch.get("mode", ["general"] * len(instructions))

        input_ids_list = []
        attention_list = []
        labels_list = []

        for m, ins, inp, out in zip(modes, instructions, inputs_, outputs):
            prompt = build_prompt(m, ins, inp)
            out = (out or "").strip()

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            out_ids = tokenizer(out, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

            ids = (prompt_ids + out_ids)[:args.max_length]
            labels = ([-100] * len(prompt_ids) + out_ids)[:args.max_length]
            attn = [1] * len(ids)

            input_ids_list.append(ids)
            attention_list.append(attn)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "attention_mask": attention_list, "labels": labels_list}

    # num_proc=1 para evitar broncas multiproceso en Windows
    train_tok = train_ds.map(tokenize_and_mask, batched=True, remove_columns=train_ds.column_names, num_proc=1)
    eval_tok = eval_ds.map(tokenize_and_mask, batched=True, remove_columns=eval_ds.column_names, num_proc=1)

    # -----------------------------
    # Collator con padding dinámico (al más largo del batch)
    # Importante: label_pad_token_id=-100 para mantener masking correcto
    # -----------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # -----------------------------
    # TrainingArguments (SIN evaluation_strategy)
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=25,
        do_eval=True,                 # tu versión soporta esto
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_checkpointing=False, # <- más rápido en 4060 Ti
        optim="adamw_torch",
    )

    # Tip: si tu transformers soporta torch_compile (algunas versiones sí), puede acelerar.
    # training_args.torch_compile = True  # (si te da error, bórralo)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("device del modelo:", next(model.parameters()).device)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("✅ Entrenamiento terminado. LoRA guardado en:", args.output_dir)


if __name__ == "__main__":
    main()
