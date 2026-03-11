"""
HeartMuLa music generation on Modal GPU.

Instala las dependencias opcionales primero:
    pip install -e ".[modal]"

Uso:
    # Generación rápida (30s, GPU por defecto A10G)
    modal run scripts/run_modal.py \\
        --tags "piano,cinematic,emotional" \\
        --max-audio-length-ms 30000

    # Con letra y GPU específica
    modal run scripts/run_modal.py \\
        --tags "piano,happy" \\
        --lyrics-file ./assets/lyrics.txt \\
        --gpu a100 \\
        --max-audio-length-ms 60000 \\
        --output ./assets/remote_output.wav

GPUs disponibles: t4 | a10g | a100 | a100-80gb | h100
"""

import modal
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Imagen remota
# ---------------------------------------------------------------------------

HEARTLIB_REPO = "https://github.com/HeartMuLa/heartlib.git"

_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"heartlib @ git+{HEARTLIB_REPO}",
        "huggingface_hub[cli]>=0.27",
        "soundfile",
    )
)

app = modal.App("heartmula-gen", image=_image)

# Volumen persistente para cachear checkpoints (~15 GB, se descarga una sola vez)
_ckpt_volume = modal.Volume.from_name("heartmula-ckpt", create_if_missing=True)
CKPT_DIR = "/ckpt"

_COMMON = dict(
    volumes={CKPT_DIR: _ckpt_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)

# ---------------------------------------------------------------------------
# Una función Modal por tipo de GPU (la GPU se fija en tiempo de importación)
# ---------------------------------------------------------------------------

def _run(tags, lyrics, version, max_audio_length_ms, min_audio_length_ms, cfg_scale, temperature, topk):
    """Lógica compartida de generación. Se ejecuta en remoto."""
    import tempfile
    import torch
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    mula_dir  = os.path.join(CKPT_DIR, f"HeartMuLa-oss-{version}")
    codec_dir = os.path.join(CKPT_DIR, "HeartCodec-oss")

    # tokenizer.json y gen_config.json viven en HeartMuLa/HeartMuLaGen (repo base)
    if not os.path.exists(os.path.join(CKPT_DIR, "tokenizer.json")):
        print("Descargando tokenizer y configuración base (HeartMuLaGen)...")
        snapshot_download(
            repo_id="HeartMuLa/HeartMuLaGen",
            local_dir=CKPT_DIR,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt"],  # solo configs
            token=hf_token,
        )
        _ckpt_volume.commit()

    if not os.path.exists(os.path.join(mula_dir, "config.json")):
        print(f"Descargando HeartMuLa {version}...")
        snapshot_download(
            repo_id=f"HeartMuLa/HeartMuLa-oss-{version}-happy-new-year",
            local_dir=mula_dir,
            token=hf_token,
        )
        _ckpt_volume.commit()

    if not os.path.exists(os.path.join(codec_dir, "config.json")):
        print("Descargando HeartCodec...")
        snapshot_download(
            repo_id="HeartMuLa/HeartCodec-oss-20260123",
            local_dir=codec_dir,
            token=hf_token,
        )
        _ckpt_volume.commit()

    from heartlib import HeartMuLaGenPipeline

    device = torch.device("cuda")
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    pipe = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=CKPT_DIR,
        version=version,
        device=device,
        dtype=dtype,
        lazy_load=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    with torch.inference_mode():
        pipe(
            {"lyrics": lyrics, "tags": tags},
            max_audio_length_ms=max_audio_length_ms,
            min_audio_length_ms=min_audio_length_ms,
            cfg_scale=cfg_scale,
            temperature=temperature,
            topk=topk,
            save_path=out_path,
        )

    audio_bytes = Path(out_path).read_bytes()
    os.unlink(out_path)
    print(f"Audio generado: {len(audio_bytes) / 1024:.1f} KB")
    return audio_bytes


@app.function(gpu="T4",      **_COMMON)
def _generate_t4(*args, **kwargs):        return _run(*args, **kwargs)

@app.function(gpu="A10G",    **_COMMON)
def _generate_a10g(*args, **kwargs):      return _run(*args, **kwargs)

@app.function(gpu="A100",    **_COMMON)
def _generate_a100(*args, **kwargs):      return _run(*args, **kwargs)

@app.function(gpu="A100-80GB", **_COMMON)
def _generate_a100_80gb(*args, **kwargs): return _run(*args, **kwargs)

@app.function(gpu="H100",    **_COMMON)
def _generate_h100(*args, **kwargs):      return _run(*args, **kwargs)


_GPU_FN = {
    "t4":        _generate_t4,
    "a10g":      _generate_a10g,
    "a100":      _generate_a100,
    "a100-80gb": _generate_a100_80gb,
    "h100":      _generate_h100,
}

# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def generate(
    tags: str = "piano,cinematic,emotional,no vocals,instrumental",
    lyrics: str = "",
    lyrics_file: str = "",
    version: str = "3B",
    max_audio_length_ms: int = 30_000,
    min_audio_length_ms: int = 0,
    cfg_scale: float = 1.5,
    temperature: float = 1.0,
    topk: int = 50,
    gpu: str = "a10g",
    output: str = "./assets/remote_output.wav",
):
    """
    Genera música con HeartMuLa en una GPU remota de Modal.

    Args:
        tags:                Tags de estilo separados por comas.
        lyrics:              Letra inline (string).
        lyrics_file:         Ruta a fichero .txt con la letra (prioridad sobre --lyrics).
        version:             Variante del modelo: 300M | 400M | 3B | 7B.
        max_audio_length_ms: Duración máxima en milisegundos.
        cfg_scale:           Classifier-free guidance (1.0 = sin CFG, más rápido).
        temperature:         Temperatura de muestreo.
        topk:                Top-k sampling.
        gpu:                 Tipo de GPU: t4 | a10g | a100 | a100-80gb | h100.
        output:              Ruta local donde guardar el .wav resultante.
    """
    gpu = gpu.lower()
    if gpu not in _GPU_FN:
        valid = " | ".join(_GPU_FN.keys())
        raise ValueError(f"GPU '{gpu}' no reconocida. Opciones: {valid}")

    lyrics_text = Path(lyrics_file).read_text(encoding="utf-8").strip() if lyrics_file else lyrics

    print(f"Lanzando generación en Modal GPU={gpu.upper()}...")
    print(f"  Tags    : {tags}")
    print(f"  Version : {version}")
    print(f"  Duración: {max_audio_length_ms / 1000:.0f}s")
    print(f"  CFG     : {cfg_scale}")

    audio_bytes = _GPU_FN[gpu].remote(
        tags, lyrics_text, version, max_audio_length_ms, min_audio_length_ms, cfg_scale, temperature, topk
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    print(f"\n✅ Audio guardado en: {out_path}  ({len(audio_bytes)/1024:.1f} KB)")
