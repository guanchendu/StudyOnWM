"""
Distill a smaller Qwen3 student from a local teacher model.

This script uses a local Qwen teacher checkpoint as the teacher, builds a
smaller student with fewer transformer layers, initializes the student from
selected teacher layers, and then trains with:

1. next-token cross-entropy on the corpus
2. KL distillation loss from the teacher logits

Supported training data:
- .txt: one training sample per non-empty line
- .jsonl:
  {"text": "..."}
  {"prompt": "...", "completion": "..."}
  {"messages": [{"role": "user", "content": "..."}, ...]}

Example:
python qwen_distill_student_2026-04-21.py \
  --teacher-dir /Users/guanchendu/Code/StudyOnWM/models/Qwen3-4B-Instruct-2507 \
  --train-file /Users/guanchendu/Code/StudyOnWM/data/distill_corpus.jsonl \
  --output-dir /Users/guanchendu/Code/StudyOnWM/outputs/qwen3-student-8l
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install a Python environment with: "
        "pip install torch transformers sentencepiece"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen3-student-8l-2026-04-21"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a smaller Qwen student model from a local teacher checkpoint."
    )
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=str(DEFAULT_TEACHER_DIR),
        help="Local path to the teacher model directory.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Training corpus (.txt or .jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the distilled student checkpoint.",
    )
    parser.add_argument(
        "--student-layers",
        type=int,
        default=8,
        help="Number of transformer layers in the student model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-step batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Stop after this many optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for the linear scheduler.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature.",
    )
    parser.add_argument(
        "--alpha-kd",
        type=float,
        default=0.7,
        help="Weight for KL distillation loss.",
    )
    parser.add_argument(
        "--alpha-ce",
        type=float,
        default=0.3,
        help="Weight for next-token cross-entropy.",
    )
    parser.add_argument(
        "--distill-topk",
        type=int,
        default=64,
        help="Approximate KL loss using teacher top-k logits. Use 0 for full vocab KL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Training device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print loss every N optimizer steps.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable student gradient checkpointing to reduce memory.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str, requested: str):
    if requested != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    if path.suffix == ".txt":
        lines = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                lines.append({"text": line})
        return lines

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {exc}"
                    ) from exc
        return records

    raise ValueError("Only .txt and .jsonl training files are supported.")


def render_record(record: dict, tokenizer) -> str:
    if "text" in record:
        return str(record["text"]).strip()

    if "prompt" in record and "completion" in record:
        messages = [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(record["completion"])},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "messages" in record:
        return tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    raise ValueError(
        "Each JSONL row must contain either text, prompt+completion, or messages."
    )


class TokenizedTextDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer, max_length: int):
        self.examples = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            token_ids = encoded["input_ids"]
            if len(token_ids) >= 2:
                self.examples.append(torch.tensor(token_ids, dtype=torch.long))

        if not self.examples:
            raise ValueError("No valid training samples were built from the corpus.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index]


def build_collate_fn(pad_token_id: int):
    def collate(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(batch, batch_first=True, padding_value=pad_token_id)
        attention_mask = (input_ids != pad_token_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def select_teacher_layers(total_layers: int, student_layers: int) -> list[int]:
    if student_layers < 1:
        raise ValueError("student_layers must be >= 1.")
    if student_layers > total_layers:
        raise ValueError(
            f"student_layers={student_layers} exceeds teacher layers={total_layers}."
        )
    if student_layers == 1:
        return [total_layers - 1]

    selected = []
    for index in range(student_layers):
        teacher_index = round(index * (total_layers - 1) / (student_layers - 1))
        selected.append(int(teacher_index))
    return selected


def build_student_config(teacher_config, student_layers: int):
    config = copy.deepcopy(teacher_config)
    config.num_hidden_layers = student_layers
    if hasattr(config, "max_window_layers"):
        config.max_window_layers = min(student_layers, teacher_config.max_window_layers)
    config.use_cache = False
    return config


def initialize_student_from_teacher(student, teacher, layer_mapping: list[int]) -> None:
    student.model.embed_tokens.load_state_dict(teacher.model.embed_tokens.state_dict())
    if hasattr(student.model, "norm") and hasattr(teacher.model, "norm"):
        student.model.norm.load_state_dict(teacher.model.norm.state_dict())
    if hasattr(student.model, "rotary_emb") and hasattr(teacher.model, "rotary_emb"):
        student.model.rotary_emb.load_state_dict(teacher.model.rotary_emb.state_dict())
    student.lm_head.load_state_dict(teacher.lm_head.state_dict())

    for student_index, teacher_index in enumerate(layer_mapping):
        student.model.layers[student_index].load_state_dict(
            teacher.model.layers[teacher_index].state_dict()
        )


def count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    topk: int,
) -> torch.Tensor:
    student_logits = student_logits[:, :-1, :].float() / temperature
    teacher_logits = teacher_logits[:, :-1, :].float() / temperature
    valid_mask = attention_mask[:, 1:].bool()

    if topk > 0:
        k = min(topk, teacher_logits.size(-1))
        teacher_topk_values, teacher_topk_indices = torch.topk(teacher_logits, k=k, dim=-1)
        student_topk_values = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
        teacher_probs = F.softmax(teacher_topk_values, dim=-1)
        student_log_probs = F.log_softmax(student_topk_values, dim=-1)
    else:
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    token_kl = token_kl.masked_select(valid_mask)
    if token_kl.numel() == 0:
        return student_logits.new_zeros(())
    return token_kl.mean() * (temperature ** 2)


def maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def train_distillation(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    teacher_dir = Path(args.teacher_dir).expanduser().resolve()
    train_file = Path(args.train_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.dtype)

    print(f"[info] teacher_dir = {teacher_dir}")
    print(f"[info] train_file  = {train_file}")
    print(f"[info] output_dir  = {output_dir}")
    print(f"[info] device      = {device}")
    print(f"[info] dtype       = {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_records = load_records(train_file)
    rendered_texts = [render_record(record, tokenizer) for record in raw_records]
    dataset = TokenizedTextDataset(rendered_texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer.pad_token_id),
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher.to(device)
    teacher.eval()
    teacher.config.use_cache = False
    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher_config = AutoConfig.from_pretrained(teacher_dir, trust_remote_code=True)
    layer_mapping = select_teacher_layers(
        total_layers=teacher_config.num_hidden_layers,
        student_layers=args.student_layers,
    )
    student_config = build_student_config(teacher_config, args.student_layers)
    student = AutoModelForCausalLM.from_config(student_config, trust_remote_code=True)
    initialize_student_from_teacher(student, teacher, layer_mapping)
    student.to(device)
    student.train()

    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(student)

    total_params, trainable_params = count_parameters(student)
    print(
        "[info] student params = "
        f"{total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable"
    )
    print(f"[info] selected teacher layers -> student layers: {layer_mapping}")
    print(f"[info] dataset size = {len(dataset)}")

    optimizer = AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    planned_steps = updates_per_epoch * args.epochs
    total_steps = planned_steps if args.max_steps <= 0 else min(planned_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    best_running_loss = None
    pending_micro_steps = 0
    last_loss_value = None
    last_ce_value = None
    last_kd_value = None

    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(dataloader, start=1):
            batch = move_batch_to_device(batch, device)

            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            student_outputs = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            ce_loss = compute_ce_loss(student_outputs.logits, batch["labels"])
            kd_loss = compute_kd_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                attention_mask=batch["attention_mask"],
                temperature=args.temperature,
                topk=args.distill_topk,
            )
            loss = args.alpha_kd * kd_loss + args.alpha_ce * ce_loss
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            pending_micro_steps += 1
            last_loss_value = float(loss.detach().cpu())
            last_ce_value = float(ce_loss.detach().cpu())
            last_kd_value = float(kd_loss.detach().cpu())

            if pending_micro_steps == args.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pending_micro_steps = 0

                loss_value = last_loss_value
                ce_value = last_ce_value
                kd_value = last_kd_value
                last_loss_value = loss_value
                last_ce_value = ce_value
                last_kd_value = kd_value
                if best_running_loss is None or loss_value < best_running_loss:
                    best_running_loss = loss_value

                if global_step % args.log_every == 0 or global_step == 1:
                    print(
                        f"[train] step={global_step} epoch={epoch + 1} "
                        f"loss={loss_value:.4f} ce={ce_value:.4f} kd={kd_value:.4f} "
                        f"best={best_running_loss:.4f}"
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if pending_micro_steps > 0 and (args.max_steps <= 0 or global_step < args.max_steps):
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pending_micro_steps = 0
            if best_running_loss is None or (
                last_loss_value is not None and last_loss_value < best_running_loss
            ):
                best_running_loss = last_loss_value
            print(
                f"[train] step={global_step} epoch={epoch + 1} "
                f"loss={last_loss_value:.4f} ce={last_ce_value:.4f} "
                f"kd={last_kd_value:.4f} best={best_running_loss:.4f}"
            )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if global_step == 0:
        raise RuntimeError(
            "No optimizer step was taken. Reduce grad_accum_steps or increase dataset size."
        )

    student.eval()
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "teacher_dir": str(teacher_dir),
        "train_file": str(train_file),
        "student_layers": args.student_layers,
        "teacher_layers": teacher_config.num_hidden_layers,
        "layer_mapping": layer_mapping,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "alpha_kd": args.alpha_kd,
        "alpha_ce": args.alpha_ce,
        "distill_topk": args.distill_topk,
        "device": device,
        "dtype": str(dtype),
    }
    (output_dir / "distill_run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] Student checkpoint saved to: {output_dir}")


def main() -> None:
    args = parse_args()
    train_distillation(args)


if __name__ == "__main__":
    main()
