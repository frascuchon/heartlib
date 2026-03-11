"""
HeartMuLa — Modal Web Service

Despliega HeartMuLa como un endpoint HTTP persistente en Modal.
El modelo se carga una sola vez por contenedor (via @modal.enter),
eliminando el overhead de carga en requests sucesivos.

Ver docs/modal_service.md para guía completa de despliegue y uso.
"""

import modal
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración de despliegue (sobreescribible por variables de entorno)
# ---------------------------------------------------------------------------

GPU_TYPE   = os.environ.get("HEARTMULA_GPU",        "A10G")
KEEP_WARM  = int(os.environ.get("HEARTMULA_MIN_CONTAINERS",  "0"))
MODEL_VER  = os.environ.get("HEARTMULA_VERSION",    "3B")
APP_NAME   = os.environ.get("HEARTMULA_APP_NAME",   "heartmula-service")
API_KEY    = os.environ.get("HEARTMULA_API_KEY",    "")

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
        "fastapi[standard]>=0.115",
        "pydantic>=2.0",
    )
)

app = modal.App(APP_NAME, image=_image)

_ckpt_volume = modal.Volume.from_name("heartmula-ckpt", create_if_missing=True)
CKPT_DIR = "/ckpt"

# ---------------------------------------------------------------------------
# Modelos Pydantic para la API
# ---------------------------------------------------------------------------

from fastapi import Header
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    tags: str = Field(
        default="piano,cinematic,emotional,no vocals,instrumental",
        description="Etiquetas de estilo separadas por comas.",
        examples=["piano,cinematic,emotional,no vocals", "upbeat,electronic,energetic"],
    )
    lyrics: str = Field(
        default="",
        description="Letra de la canción. Dejar vacío para instrumental.",
    )
    version: str = Field(
        default=MODEL_VER,
        description="Variante del modelo: 300M | 400M | 3B | 7B.",
        pattern="^(300M|400M|3B|7B)$",
    )
    max_audio_length_ms: int = Field(
        default=30_000,
        ge=5_000,
        le=300_000,
        description="Duración máxima del audio en milisegundos (5s – 300s).",
    )
    min_audio_length_ms: int = Field(
        default=0,
        ge=0,
        description="Duración mínima: el EOS se ignora hasta alcanzarla.",
    )
    cfg_scale: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Classifier-free guidance. 1.0 = sin CFG (más rápido, recomendado para instrumental).",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Temperatura de muestreo. Valores altos = más variedad.",
    )
    topk: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Top-k sampling.",
    )
    response_format: str = Field(
        default="wav",
        description="Formato de respuesta: 'wav' (binario) o 'base64' (JSON).",
        pattern="^(wav|base64)$",
    )


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 48000
    format: str = "wav"
    duration_ms: int
    version: str
    tags: str


class HealthResponse(BaseModel):
    status: str
    gpu: str
    version: str
    min_containers: int
    model_loaded: bool


class InfoResponse(BaseModel):
    app_name: str
    gpu: str
    version: str
    min_containers: int
    endpoints: dict


# ---------------------------------------------------------------------------
# Servicio principal
# ---------------------------------------------------------------------------

@app.cls(
    gpu=GPU_TYPE,
    min_containers=KEEP_WARM,
    volumes={CKPT_DIR: _ckpt_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("heartmula-api-key"),
    ],
    timeout=60 * 60,
)
class HeartMuLaService:

    @modal.enter()
    def load_model(self):
        """Carga el modelo una sola vez por ciclo de vida del contenedor."""
        import torch
        from huggingface_hub import snapshot_download
        from heartlib import HeartMuLaGenPipeline

        hf_token = os.environ.get("HF_TOKEN")
        mula_dir  = os.path.join(CKPT_DIR, f"HeartMuLa-oss-{MODEL_VER}")
        codec_dir = os.path.join(CKPT_DIR, "HeartCodec-oss")

        if not os.path.exists(os.path.join(CKPT_DIR, "tokenizer.json")):
            print("Descargando tokenizer y config base...")
            snapshot_download(
                repo_id="HeartMuLa/HeartMuLaGen",
                local_dir=CKPT_DIR,
                ignore_patterns=["*.bin", "*.safetensors", "*.pt"],
                token=hf_token,
            )
            _ckpt_volume.commit()

        if not os.path.exists(os.path.join(mula_dir, "config.json")):
            print(f"Descargando HeartMuLa {MODEL_VER}...")
            snapshot_download(
                repo_id=f"HeartMuLa/HeartMuLa-oss-{MODEL_VER}-happy-new-year",
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

        device = torch.device("cuda")
        dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.pipe = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=CKPT_DIR,
            version=MODEL_VER,
            device=device,
            dtype=dtype,
            lazy_load=False,
        )
        self._model_loaded = True
        print(f"Modelo listo — GPU={GPU_TYPE}, versión={MODEL_VER}, dtype={dtype}")

    # ------------------------------------------------------------------
    # Lógica de generación interna
    # ------------------------------------------------------------------

    def _run_generation(self, req: GenerateRequest) -> bytes:
        import tempfile, torch
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        with torch.inference_mode():
            self.pipe(
                {"lyrics": req.lyrics, "tags": req.tags},
                max_audio_length_ms=req.max_audio_length_ms,
                min_audio_length_ms=req.min_audio_length_ms,
                cfg_scale=req.cfg_scale,
                temperature=req.temperature,
                topk=req.topk,
                save_path=out_path,
            )
        audio_bytes = Path(out_path).read_bytes()
        os.unlink(out_path)
        return audio_bytes

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, authorization: str | None):
        """Raises HTTP 401 if API key auth is configured and the header is wrong."""
        from fastapi import HTTPException

        api_key = os.environ.get("HEARTMULA_API_KEY", "")
        if not api_key:
            return  # auth not configured — open access

        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    # ------------------------------------------------------------------
    # POST /generate  →  audio/wav  (o JSON+base64)
    # ------------------------------------------------------------------

    @modal.web_endpoint(method="POST", docs=True)
    def generate(self, req: GenerateRequest, authorization: str | None = Header(default=None)):
        """
        Genera audio musical a partir de tags y/o letra.

        Devuelve el audio en formato WAV (binario) cuando `response_format=wav`
        o en JSON con el audio codificado en base64 cuando `response_format=base64`.
        """
        import time
        from fastapi.responses import Response

        self._check_auth(authorization)

        t0 = time.perf_counter()
        audio_bytes = self._run_generation(req)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        print(f"Generado en {elapsed_ms}ms — {len(audio_bytes)/1024:.1f} KB — tags={req.tags!r}")

        if req.response_format == "base64":
            import base64, json
            payload = GenerateResponse(
                audio_base64=base64.b64encode(audio_bytes).decode(),
                duration_ms=elapsed_ms,
                version=req.version,
                tags=req.tags,
            )
            return Response(
                content=payload.model_dump_json(),
                media_type="application/json",
            )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="output.wav"',
                "X-Generation-Ms": str(elapsed_ms),
                "X-Tags": req.tags,
            },
        )

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @modal.web_endpoint(method="GET", docs=True)
    def health(self):
        """Comprueba que el servicio y el modelo están listos."""
        from fastapi.responses import JSONResponse
        return JSONResponse(HealthResponse(
            status="ok",
            gpu=GPU_TYPE,
            version=MODEL_VER,
            min_containers=KEEP_WARM,
            model_loaded=getattr(self, "_model_loaded", False),
        ).model_dump())

    # ------------------------------------------------------------------
    # GET /info
    # ------------------------------------------------------------------

    @modal.web_endpoint(method="GET", docs=True)
    def info(self):
        """Devuelve la configuración de despliegue del servicio."""
        from fastapi.responses import JSONResponse
        return JSONResponse(InfoResponse(
            app_name=APP_NAME,
            gpu=GPU_TYPE,
            version=MODEL_VER,
            min_containers=KEEP_WARM,
            endpoints={
                "POST /generate": "Genera audio musical",
                "GET  /health":   "Estado del servicio",
                "GET  /info":     "Configuración del despliegue",
            },
        ).model_dump())
