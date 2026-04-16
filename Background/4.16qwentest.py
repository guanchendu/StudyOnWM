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

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)  #“是否信任模型作者提供的自定义代码，并允许 transformers 去加载它。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    chat_text = build_chat_text(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt")  #对chat 进行token 话
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0:
        model.generation_config.do_sample = True
        model.generation_config.temperature = temperature
        model.generation_config.top_p = top_p
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    """
    表示取出第一个样本的输出。
    
    “从输入长度之后开始切，只保留新生成的 token。”
    
    所以第一行本质上是在做：
    
    “把模型回答部分单独切出来。”
    """
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    ## 所以要用 tokenizer.decode(...) 把它解码回来。
    ## 表示解码时跳过特殊 token，比如：
    ## 表示把字符串前后的空格、换行去掉，让最终结果更整洁。

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
    )  ##模型放在哪个目录
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful AI assistant.",
        help="System prompt sent to Qwen.",
    )  ##系统提示词
    parser.add_argument(
        "--prompt",
        default="使用一句话解释世界模型",
        help="User prompt sent to Qwen.",
    )  #用户真正输入的问题
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of newly generated tokens.",
    )  #最多允许模型新生成多少个 token
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )  ## 生成时的随机程度
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold.",
    ) #模型每一步不会在所有词里乱选 它只会在“概率加起来前 90% 的那一批候选词”里挑
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
