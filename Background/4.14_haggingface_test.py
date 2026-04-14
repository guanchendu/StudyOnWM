import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    from transformers import pipeline
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Please install with: pip install transformers torch sentencepiece"
    ) from exc


def choose_device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


def run_sentiment_demo(text: str, model_name: str, device: str | int):
    """
    这个函数做的是“输入一句话，判断它情绪偏正面还是负面”。
    :param text: text input
    :param model_name: which model the user choosed
    :param device: the current device depends on the bands of laptops
    :return: a json structer
    """
    classifier = pipeline(
        task="sentiment-analysis",
        model=model_name,
        device=device,
    ) ## 创建了一个pipeline

    result = classifier(text)  #这一步相当于“把一句文本送进模型”。

    print("\n=== Sentiment Demo ===")
    print(f"Model: {model_name}")
    print(f"Input: {text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    '''
    label：模型认为是正向情绪
    score：模型对这个判断的置信度，越接近 1 越自信

    '''


def run_generation_demo(prompt: str, model_name: str, device: str | int):
    generator = pipeline(
        task="text-generation",
        model=model_name,
        device=device,
    )
    result = generator(
        prompt,
        max_new_tokens=40,  #意思是最多新生成 40 个 token
        do_sample=True,   #表示生成时采用“采样”而不是每一步都贪心选最大概率词。
        temperature=0.8,  #越小越保守 越大越随机
        top_k=50,   #每一步生成时，只从概率最高的前 50 个候选 token 里采样。
        top_p=0.95,  #它不是固定取前多少个词，而是取“累计概率达到 95% 的那一批候选词”里去采样。
        repetition_penalty=1.05,  #给重复内容一点惩罚，减少模型疯狂重复同一串词。
    )   #先创建一个文本生成 pipeline：
    print("\n=== Generation Demo ===")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="A simple Hugging Face playground script for quick experiments."
    )
    parser.add_argument(
        "--task",
        choices=["sentiment", "generate", "both"],
        default="both",
        help="Which Hugging Face demo to run.",
    )
    parser.add_argument(
        "--text",
        default="i am donald trump",
        help="Input text for sentiment analysis.",
    )
    parser.add_argument(
        "--prompt",
        default="Deep learning changes the world because",
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--sentiment-model",
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        help="Model used for sentiment analysis.",
    )
    parser.add_argument(
        "--generation-model",
        default="sshleifer/tiny-gpt2",
        help="Model used for text generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device()

    print("Running Hugging Face demo...")
    print(f"Selected device: {device}")

    if args.task in {"sentiment", "both"}:
        run_sentiment_demo(
            text=args.text,
            model_name=args.sentiment_model,
            device=device,
        )

    if args.task in {"generate", "both"}:
        run_generation_demo(
            prompt=args.prompt,
            model_name=args.generation_model,
            device=device,
        )


if __name__ == "__main__":
    main()
