import argparse
import os
import inspect

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
    p = argparse.ArgumentParser(
        "LoRA fine-tune Tutor de Algoritmos (Qwen2.5) - Rápido y eficiente (padding dinámico)"
    )
    p.add_argument("--dataset_path", type=str, default="dataset_final_v3_with_source.jsonl")
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
    "general": (
        "Eres un Tutor Inteligente de Algoritmos. "
        "Explicas de forma clara, paso a paso y con paciencia. "
        "Si el conocimiento proviene del temario, usa el prefijo 'Va, del temario:'. "
        "Si es conocimiento externo, usa el prefijo 'Dato extra (no del dataset):'."
    )
}


def build_prompt(mode, instruction, inp, source=None):
    """Construye el prompt de entrenamiento.

    Agregamos 'source' para que el modelo aprenda a distinguir:
      - source='dataset'  -> conocimiento del temario/dataset
      - source='extra'    -> conocimiento externo/general
    """
    mode = (mode or "general").strip().lower()
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    source = (source or "dataset").strip().lower()

    system = SYSTEM_BY_MODE.get(mode, SYSTEM_BY_MODE["general"])

    prompt = system + "\n\n"
    prompt += "### Source:\n" + source + "\n"
    prompt += "### Instruction:\n" + instruction + "\n"
    prompt += "### Input:\n" + (inp if inp else "") + "\n"
    prompt += "### Response:\n"
    return prompt


def main():
    # Reduce ruido/bugs de multiprocess en Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DATASETS_DISABLE_MULTIPROCESSING"] = "1"

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Nota: target_modules puede variar por arquitectura.
    # Para Qwen2.5 normalmente funciona con estos nombres; si te da error, ajustamos.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # --- Fix para LoRA + gradient checkpointing ---
    # Si no se habilita esto, el backward puede fallar con:
    # RuntimeError: element 0 of tensors does not require grad
    try:
        model.enable_input_require_grads()
    except Exception:
        # Fallback: forzar grad en embeddings de entrada
        if hasattr(model, 'get_input_embeddings') and model.get_input_embeddings() is not None:
            model.get_input_embeddings().weight.requires_grad_(True)

    # Para checkpointing, cache debe estar apagado
    if hasattr(model, 'config'):
        model.config.use_cache = False

    def tokenize_and_mask(batch):
        instructions = batch["instruction"]
        inputs_ = batch.get("input", [""] * len(instructions))
        outputs = batch["output"]
        modes = batch.get("mode", ["general"] * len(instructions))
        sources = batch.get("source", ["dataset"] * len(instructions))
        prefixes = batch.get("preferred_prefix", ["Va, del temario:"] * len(instructions))

        input_ids_list = []
        attention_list = []
        labels_list = []

        for m, ins, inp, out, src, pref in zip(modes, instructions, inputs_, outputs, sources, prefixes):
            prompt = build_prompt(m, ins, inp, src)
            out = (out or "").strip()
            pref = (pref or "").strip()

            if pref and not out.startswith(pref):
                out = f"{pref} {out}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            out_ids = tokenizer(out, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]

            ids = (prompt_ids + out_ids)[:args.max_length]
            labels = ([-100] * len(prompt_ids) + out_ids)[:args.max_length]
            attn = [1] * len(ids)

            input_ids_list.append(ids)
            attention_list.append(attn)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_list,
            "labels": labels_list,
        }

    train_tok = train_ds.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=1,
    )
    eval_tok = eval_ds.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=eval_ds.column_names,
        num_proc=1,
    )

    # ---- TrainingArguments compatible con varias versiones de transformers ----
    ta_kwargs = dict(
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
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_checkpointing=True,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        ta_kwargs["evaluation_strategy"] = "steps"
        ta_kwargs["eval_steps"] = 100
    elif "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["eval_steps"] = 100
    else:
        # fallback para versiones viejas
        ta_kwargs["do_eval"] = True

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Entrenamiento finalizado. Modelo guardado en:", args.output_dir)


if __name__ == "__main__":
    main()
