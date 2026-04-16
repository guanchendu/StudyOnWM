import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install with: pip install torch transformers sentencepiece"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str):
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def build_chat_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_reply(
    model_dir: Path,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = choose_device()
    dtype = choose_dtype(device)

    print(f"Loading model from: {model_dir}")
    print(f"Selected device: {device}")
    print(f"Selected dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    chat_text = build_chat_text(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n=== Qwen Demo ===")
    print(f"User prompt: {user_prompt}")
    print("\nAssistant reply:")
    print(reply)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a local Qwen chat example with a downloaded model."
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Local path of the downloaded Qwen model.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt sent to Qwen.",
    )
    parser.add_argument(
        "--prompt",
        default="请用三句话介绍一下什么是 world model。",
        help="User prompt sent to Qwen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_reply(
        model_dir=Path(args.model_dir).expanduser().resolve(),
        system_prompt=args.system_prompt,
        user_prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
