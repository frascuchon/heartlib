"""
Fast instrumental music generation using HeartMuLaInstrumentalPipeline.

Optimisations active by default (vs run_music_generation.py):
  - cfg_scale=1.0    : no classifier-free guidance → ~2× fewer HeartMuLa forward passes
  - codec_steps=5    : 5 Euler steps instead of 10  → ~50% faster HeartCodec
  - codec_guidance=1.0: no codec CFG                → ~2× fewer estimator calls per step

Expected combined speedup: ~60-70% wall-time reduction on the 3B model.

Example (30 s clip, MPS):
    python examples/run_instrumental_generation.py \\
        --model_path ./ckpt --version 3B \\
        --tags "piano,guitar,cinematic,emotional,no vocals" \\
        --max_audio_length_ms 30000 \\
        --lazy_load true \\
        --save_path ./assets/instrumental.wav
"""

from heartlib import HeartMuLaInstrumentalPipeline
import argparse
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got: {value}")


def str2dtype(value):
    value = value.lower()
    if value == "float32" or value == "fp32":
        return torch.float32
    elif value == "float16" or value == "fp16":
        return torch.float16
    elif value == "bfloat16" or value == "bf16":
        return torch.bfloat16
    else:
        raise argparse.ArgumentTypeError(f"Dtype not recognized: {value}")


def str2device(value):
    return torch.device(value.lower())


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_mula_dtype(device: str) -> str:
    if device == "mps":
        return "float16"
    if device == "cpu":
        return "float32"
    return "bfloat16"


def parse_args():
    _device = _default_device()
    parser = argparse.ArgumentParser(
        description="Generate instrumental music with HeartMuLa (optimised pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Root directory of the downloaded checkpoints.")
    parser.add_argument("--version", type=str, default="3B",
                        choices=["300M", "400M", "3B", "7B"],
                        help="HeartMuLa model size.")
    parser.add_argument("--tags", type=str,
                        default="./assets/tags_instrumental.txt",
                        help="Comma-separated style tags or path to a .txt file.")
    parser.add_argument("--save_path", type=str, default="./assets/instrumental.wav",
                        help="Output audio file (.wav or .mp3).")
    parser.add_argument("--max_audio_length_ms", type=int, default=30_000,
                        help="Maximum generation length in milliseconds.")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mula_device", type=str2device, default=_device)
    parser.add_argument("--codec_device", type=str2device, default=_device)
    parser.add_argument("--mula_dtype", type=str2dtype, default=_default_mula_dtype(_device))
    parser.add_argument("--codec_dtype", type=str2dtype, default="float32")
    parser.add_argument("--lazy_load", type=str2bool, default=True,
                        help="Load each model just before use and free immediately after.")
    parser.add_argument("--codec_steps", type=int, default=5,
                        help="Euler steps for HeartCodec (5=fast, 10=quality).")
    parser.add_argument("--codec_guidance", type=float, default=1.0,
                        help="HeartCodec guidance scale (1.0=disabled/fastest, 1.25=default quality).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve tags: accept inline string or file path
    tags = args.tags
    if tags.endswith(".txt") and __import__("os").path.isfile(tags):
        with open(tags, encoding="utf-8") as f:
            tags = f.read().strip()

    pipe = HeartMuLaInstrumentalPipeline.from_pretrained(
        pretrained_path=args.model_path,
        version=args.version,
        device={
            "mula": torch.device(args.mula_device),
            "codec": torch.device(args.codec_device),
        },
        dtype={
            "mula": args.mula_dtype,
            "codec": args.codec_dtype,
        },
        lazy_load=args.lazy_load,
    )

    pipe(
        tags=tags,
        save_path=args.save_path,
        max_audio_length_ms=args.max_audio_length_ms,
        temperature=args.temperature,
        topk=args.topk,
        codec_steps=args.codec_steps,
        codec_guidance=args.codec_guidance,
    )

    print(f"Generated instrumental saved to {args.save_path}")
