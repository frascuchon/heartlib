"""
HeartMuLa — Remote Generation Client

Calls the deployed Modal HTTP endpoint and saves the resulting WAV locally.
No dependencies beyond the Python standard library.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

SERVICE_URL = (
    "https://frascuchon--heartmula-service-heartmulaservice-generate.modal.run"
)

# API key can also be set via environment variable HEARTMULA_API_KEY
_ENV_API_KEY = os.environ.get("HEARTMULA_API_KEY", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music via the remote HeartMuLa service."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=SERVICE_URL,
        help="Service endpoint URL.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="piano,cinematic,emotional,no vocals,instrumental",
        help="Comma-separated style tags.",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default="",
        help="Inline lyrics (leave empty for instrumental).",
    )
    parser.add_argument(
        "--lyrics-file",
        type=str,
        default=None,
        help="Path to a .txt file with lyrics (takes priority over --lyrics).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="3B",
        choices=["300M", "400M", "3B", "7B"],
        help="Model variant.",
    )
    parser.add_argument(
        "--max-audio-length-ms",
        type=int,
        default=30_000,
        help="Maximum audio duration in milliseconds.",
    )
    parser.add_argument(
        "--min-audio-length-ms",
        type=int,
        default=0,
        help="Minimum audio duration in milliseconds (EOS is suppressed until reached).",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./assets/remote_output.wav",
        help="Path for the output WAV file.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=_ENV_API_KEY,
        help="API key for the service (or set HEARTMULA_API_KEY env var).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lyrics = args.lyrics
    if args.lyrics_file is not None:
        with open(args.lyrics_file, "r", encoding="utf-8") as f:
            lyrics = f.read()

    payload = {
        "tags": args.tags,
        "lyrics": lyrics,
        "version": args.version,
        "max_audio_length_ms": args.max_audio_length_ms,
        "min_audio_length_ms": args.min_audio_length_ms,
        "cfg_scale": args.cfg_scale,
        "temperature": args.temperature,
        "topk": args.topk,
        "response_format": "wav",
    }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.url,
        data=body,
        headers=headers,
        method="POST",
    )

    print(f"Sending request to {args.url} ...")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            audio_bytes = resp.read()
            generation_ms = resp.headers.get("X-Generation-Ms")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {e.reason}: {error_body}")
        raise SystemExit(1)
    elapsed = time.monotonic() - t0

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    gen_info = f"{generation_ms} ms (server)" if generation_ms else f"{elapsed:.1f}s (total round-trip)"
    print(f"Done — generation time: {gen_info}")
    print(f"Audio saved to: {out_path}  ({len(audio_bytes) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
