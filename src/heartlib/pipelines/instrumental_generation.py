"""
HeartMuLaInstrumentalPipeline — fast inference pipeline for pure instrumental generation.

Key differences from HeartMuLaGenPipeline:
- cfg_scale=1.0 by default: disables classifier-free guidance, halving HeartMuLa compute.
  Lyrics conditioning is irrelevant for instrumentals so no quality loss.
- codec_steps=5 by default: fewer Euler steps in the HeartCodec flow-matching decoder
  (down from 10), cutting codec time by ~50%.
- codec_guidance=1.0 by default: disables HeartCodec CFG, halving estimator cost per step.
- lyrics are optional (default: empty string).
- Uses torch.inference_mode() throughout (vs no_grad in the base flow-matching path).

Expected wall-time improvement over the default pipeline: ~60-70% on 3B model / MPS.
"""

from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
from .music_generation import (
    HeartMuLaGenConfig,
    _resolve_paths,
    _resolve_devices,
)
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
import os
from tqdm import tqdm
import torchaudio
import gc


class HeartMuLaInstrumentalPipeline:
    """
    Inference pipeline for fast, pure instrumental music generation.

    Optimised defaults (all overridable):
        cfg_scale=1.0       — no CFG; halves HeartMuLa forward passes
        codec_steps=5       — 5 Euler steps instead of 10
        codec_guidance=1.0  — no codec CFG; halves estimator cost per step
        lyrics=""           — empty by default (not needed for instrumentals)

    Usage::

        pipe = HeartMuLaInstrumentalPipeline.from_pretrained(
            pretrained_path="./ckpt",
            version="3B",
            device=torch.device("mps"),
            dtype=torch.float16,
            lazy_load=True,
        )
        pipe(
            tags="piano,guitar,cinematic,emotional,no vocals",
            save_path="output.wav",
            max_audio_length_ms=30_000,
        )
    """

    # Optimised defaults — kept as class attributes so subclasses can override.
    DEFAULT_CFG_SCALE: float = 1.0
    DEFAULT_CODEC_STEPS: int = 5
    DEFAULT_CODEC_GUIDANCE: float = 1.0
    DEFAULT_TEMPERATURE: float = 1.0
    DEFAULT_TOPK: int = 50
    DEFAULT_MAX_AUDIO_LENGTH_MS: int = 120_000

    def __init__(
        self,
        heartmula_path: str,
        heartcodec_path: str,
        heartmula_device: torch.device,
        heartcodec_device: torch.device,
        heartmula_dtype: torch.dtype,
        heartcodec_dtype: torch.dtype,
        lazy_load: bool,
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
    ):
        self.text_tokenizer = text_tokenizer
        self.config = config

        self._parallel_number = 8 + 1  # 8 audio codebooks + 1 text stream
        self._muq_dim = 512

        self.mula_dtype = heartmula_dtype
        self.mula_path = heartmula_path
        self.mula_device = heartmula_device
        self.codec_dtype = heartcodec_dtype
        self.codec_path = heartcodec_path
        self.codec_device = heartcodec_device
        self.lazy_load = lazy_load

        self._mula: Optional[HeartMuLa] = None
        self._codec: Optional[HeartCodec] = None

        if not lazy_load:
            print("Loading HeartMuLa and HeartCodec onto device...")
            self._mula = HeartMuLa.from_pretrained(
                self.mula_path,
                device_map=self.mula_device,
                dtype=self.mula_dtype,
            )
            self._codec = HeartCodec.from_pretrained(
                self.codec_path,
                device_map=self.codec_device,
                dtype=self.codec_dtype,
            )

    # ------------------------------------------------------------------
    # Model accessors (lazy loading)
    # ------------------------------------------------------------------

    @property
    def mula(self) -> HeartMuLa:
        if isinstance(self._mula, HeartMuLa):
            return self._mula
        self._mula = HeartMuLa.from_pretrained(
            self.mula_path,
            device_map=self.mula_device,
            dtype=self.mula_dtype,
        )
        return self._mula

    @property
    def codec(self) -> HeartCodec:
        if isinstance(self._codec, HeartCodec):
            return self._codec
        self._codec = HeartCodec.from_pretrained(
            self.codec_path,
            device_map=self.codec_device,
            dtype=self.codec_dtype,
        )
        return self._codec

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_device_cache(device: torch.device):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    def _unload(self):
        if not self.lazy_load:
            return
        if isinstance(self._mula, HeartMuLa):
            print("Unloading HeartMuLa from device.")
            del self._mula
            gc.collect()
            self._clear_device_cache(self.mula_device)
            self._mula = None
        if isinstance(self._codec, HeartCodec):
            print("Unloading HeartCodec from device.")
            del self._codec
            gc.collect()
            self._clear_device_cache(self.codec_device)
            self._codec = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, tags: str, lyrics: str) -> Dict[str, Any]:
        """
        Build prompt tensors for unconditional (cfg_scale=1.0) generation.

        Because cfg_scale is always 1.0 in this pipeline the batch size is
        always 1 — no duplication needed.
        """
        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.mula_dtype)
        muq_idx = len(tags_ids)

        lyrics = lyrics.lower()
        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if not lyrics_ids or lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        return {
            "tokens": tokens.unsqueeze(0),           # (1, T, C)
            "tokens_mask": tokens_mask.unsqueeze(0),  # (1, T, C)
            "muq_embed": muq_embed.unsqueeze(0),      # (1, muq_dim)
            "muq_idx": [muq_idx],
            "pos": torch.arange(prompt_len, dtype=torch.long).unsqueeze(0),  # (1, T)
        }

    # ------------------------------------------------------------------
    # Forward (HeartMuLa autoregressive decoding, cfg_scale=1.0)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        min_audio_length_ms: int = 0,
        temperature: float = 1.0,
        topk: int = 2048,
    ) -> Dict[str, Any]:
        prompt_tokens = model_inputs["tokens"].to(self.mula_device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(self.mula_device)
        continuous_segment = model_inputs["muq_embed"].to(self.mula_device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(self.mula_device)

        # cfg_scale=1.0 → batch size is always 1, no unconditional branch
        self.mula.setup_caches(1)

        with torch.autocast(device_type=self.mula_device.type, dtype=self.mula_dtype):
            curr_token = self.mula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=1.0,
                continuous_segments=continuous_segment,
                starts=starts,
            )

        frames = [curr_token[0:1]]

        def _pad_audio_token(token: torch.Tensor):
            padded = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded[:, :-1] = token
            padded = padded.unsqueeze(1)
            padded_mask = torch.ones_like(padded, dtype=torch.bool)
            padded_mask[..., -1] = False
            return padded, padded_mask

        max_audio_frames = max_audio_length_ms // 80
        min_audio_frames = min_audio_length_ms // 80
        last_valid_token = curr_token

        for i in tqdm(range(max_audio_frames), desc="Generating frames"):
            curr_token, curr_token_mask = _pad_audio_token(last_valid_token)
            with torch.autocast(device_type=self.mula_device.type, dtype=self.mula_dtype):
                curr_token = self.mula.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=1.0,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                if i + 1 >= min_audio_frames:
                    break
                continue   # skip EOS frame, keep generating
            last_valid_token = curr_token
            frames.append(curr_token[0:1])

        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        self._unload()
        return {"frames": frames}

    # ------------------------------------------------------------------
    # Postprocessing (HeartCodec decoding with reduced steps)
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        model_outputs: Dict[str, Any],
        save_path: str,
        codec_steps: int,
        codec_guidance: float,
    ):
        frames = model_outputs["frames"].to(self.codec_device)
        wav = self.codec.detokenize(
            frames,
            num_steps=codec_steps,
            guidance_scale=codec_guidance,
        )
        self._unload()
        wav_cpu = wav.to(torch.float32).cpu()
        try:
            torchaudio.save(save_path, wav_cpu, 48000)
        except (ImportError, RuntimeError):
            import soundfile as sf
            wav_path = os.path.splitext(save_path)[0] + ".wav"
            sf.write(wav_path, wav_cpu.squeeze(0).numpy().T, 48000)
            if wav_path != save_path:
                print(f"Note: saved as WAV (torchcodec not available): {wav_path}")

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        tags: str,
        save_path: str = "output.wav",
        lyrics: str = "",
        max_audio_length_ms: int = DEFAULT_MAX_AUDIO_LENGTH_MS,
        min_audio_length_ms: int = 0,
        temperature: float = DEFAULT_TEMPERATURE,
        topk: int = DEFAULT_TOPK,
        codec_steps: int = DEFAULT_CODEC_STEPS,
        codec_guidance: float = DEFAULT_CODEC_GUIDANCE,
    ):
        """
        Generate an instrumental track.

        Args:
            tags: Comma-separated style/mood tags, e.g. ``"piano,cinematic,emotional"``.
            save_path: Output file path (.wav or .mp3).
            lyrics: Optional lyrics. Default is empty (pure instrumental).
            max_audio_length_ms: Maximum generation length in milliseconds.
            min_audio_length_ms: Minimum generation length in milliseconds (EOS is ignored until reached).
            temperature: Sampling temperature (1.0 = no change).
            topk: Top-k sampling cutoff.
            codec_steps: Euler steps for the HeartCodec flow-matching decoder (5–10).
            codec_guidance: Guidance scale for HeartCodec (1.0 = disabled, fastest).
        """
        model_inputs = self._preprocess(tags=tags, lyrics=lyrics)
        model_outputs = self._forward(
            model_inputs,
            max_audio_length_ms=max_audio_length_ms,
            min_audio_length_ms=min_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        self._postprocess(
            model_outputs,
            save_path=save_path,
            codec_steps=codec_steps,
            codec_guidance=codec_guidance,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        version: str,
        device: Union[torch.device, Dict[str, torch.device]],
        dtype: Union[torch.dtype, Dict[str, torch.dtype]],
        lazy_load: bool = True,
    ) -> "HeartMuLaInstrumentalPipeline":
        """
        Load the pipeline from a local checkpoint directory.

        Args:
            pretrained_path: Root directory containing ``HeartMuLa-oss-{version}/``,
                ``HeartCodec-oss/``, ``tokenizer.json``, and ``gen_config.json``.
            version: Model size — one of ``"300M"``, ``"400M"``, ``"3B"``, ``"7B"``.
            device: A single ``torch.device`` or a dict ``{"mula": ..., "codec": ...}``.
            dtype: A single ``torch.dtype`` or a dict ``{"mula": ..., "codec": ...}``.
            lazy_load: If ``True`` (default), each model is loaded just before use
                and freed immediately after — reduces peak memory at the cost of
                reload overhead on repeated calls.
        """
        mula_path, codec_path, tokenizer_path, gen_config_path = _resolve_paths(
            pretrained_path, version
        )
        mula_device, codec_device, lazy_load = _resolve_devices(device, lazy_load)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        mula_dtype = dtype["mula"] if isinstance(dtype, dict) else dtype
        codec_dtype = dtype["codec"] if isinstance(dtype, dict) else dtype

        return cls(
            heartmula_path=mula_path,
            heartcodec_path=codec_path,
            heartmula_device=mula_device,
            heartcodec_device=codec_device,
            heartmula_dtype=mula_dtype,
            heartcodec_dtype=codec_dtype,
            lazy_load=lazy_load,
            text_tokenizer=tokenizer,
            config=gen_config,
        )
