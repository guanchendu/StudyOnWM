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
    classifier = pipeline(
        task="sentiment-analysis",
        model=model_name,
        device=device,
    )
    result = classifier(text)
    print("\n=== Sentiment Demo ===")
    print(f"Model: {model_name}")
    print(f"Input: {text}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_generation_demo(prompt: str, model_name: str, device: str | int):
    generator = pipeline(
        task="text-generation",
        model=model_name,
        device=device,
    )
    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
    )
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
        default="I am excited to try Hugging Face models today.",
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
