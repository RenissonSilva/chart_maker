"""Source separation with Demucs ``htdemucs_6s`` (drums, bass, other, vocals, guitar, piano)."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

MODEL_NAME = "htdemucs_6s"
CACHED_STEMS = ("drums", "guitar", "other", "piano")
CACHE_ENTRIES = 3  # stems take ~25 MB per minute of audio; keep only the most recent songs

_model = None


def _load_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        from huggingface_hub.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # silences the "unauthenticated requests" notice
        _model = get_model(MODEL_NAME)
    return _model


def audio_key(mix: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(mix).tobytes()).hexdigest()[:16] + "-" + MODEL_NAME


def separate(
    mix: np.ndarray,
    device: str = "cpu",
    progress: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Separate a (2, samples) 44.1 kHz mix into stems of the same shape."""
    import torch
    from demucs.apply import apply_model

    model = _load_model()
    if model.samplerate != 44100:
        raise RuntimeError(f"unexpected Demucs sample rate {model.samplerate}")
    wav = torch.from_numpy(np.ascontiguousarray(mix))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    length = mix.shape[1]

    def callback(info: dict) -> None:
        if progress is not None and info.get("state") == "end":
            progress(min(1.0, (info.get("segment_offset", 0) + 0.75 * 7.8 * 44100) / length))

    random.seed(0)  # Demucs draws a random time shift; fixing it makes the same song give the same chart
    with torch.inference_mode():
        out = apply_model(model, ((wav - mean) / std)[None], shifts=1, split=True, overlap=0.25,
                          device=device, callback=callback)[0]
    out = out * std + mean
    return {name: out[i].numpy() for i, name in enumerate(model.sources)}


def separate_cached(
    mix: np.ndarray,
    cache_dir: Path | None,
    device: str = "cpu",
    progress: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    """``separate`` with an on-disk FLAC cache of the stems the charting uses."""
    import soundfile as sf

    if cache_dir is None:
        stems = separate(mix, device, progress)
        return {name: stems[name] for name in CACHED_STEMS}
    entry = cache_dir / audio_key(mix)
    files = {name: entry / f"{name}.flac" for name in CACHED_STEMS}
    if all(f.exists() for f in files.values()):
        entry.touch()
        return {name: sf.read(f, dtype="float32", always_2d=True)[0].T.copy() for name, f in files.items()}

    stems = separate(mix, device, progress)
    entry.mkdir(parents=True, exist_ok=True)
    for name, f in files.items():
        sf.write(f, stems[name].T, 44100, subtype="PCM_16")
    entries = sorted((p for p in cache_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in entries[CACHE_ENTRIES:]:
        shutil.rmtree(old, ignore_errors=True)
    return {name: stems[name] for name in CACHED_STEMS}
