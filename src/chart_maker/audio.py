"""Audio I/O through ffmpeg: decoding, tags, cover art and Ogg Vorbis encoding."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100


class AudioError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise AudioError(f"'{tool}' não encontrado. Instale o ffmpeg (ex.: sudo apt install ffmpeg).")
    return path


@dataclass
class AudioInfo:
    duration: float
    tags: dict[str, str] = field(default_factory=dict)
    has_cover: bool = False


def probe(path: Path) -> AudioInfo:
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioError(f"Não foi possível ler o arquivo de áudio: {result.stderr.strip() or path.name}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise AudioError("O arquivo não contém uma faixa de áudio.")
    tags: dict[str, str] = {}
    for source in [data.get("format", {})] + streams:
        for key, value in (source.get("tags") or {}).items():
            tags.setdefault(key.lower(), str(value).strip())
    has_cover = any(s.get("codec_type") == "video" for s in streams)
    return AudioInfo(float(data.get("format", {}).get("duration") or 0.0), tags, has_cover)


def decode(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 stereo, shape (2, samples)."""
    result = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-ac", "2",
         "-ar", str(sample_rate), "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise AudioError(f"Falha ao decodificar o áudio: {result.stderr.decode(errors='replace').strip()}")
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size < sample_rate:
        raise AudioError("O áudio é curto demais.")
    return samples.reshape(-1, 2).T.copy()


def encode_ogg(samples: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE, quality: int = 8,
               pad: float = 0.0) -> None:
    """Write (channels, samples) float audio as Ogg Vorbis, optionally prefixed by ``pad`` seconds of silence."""
    if pad > 0:
        silence = np.zeros((samples.shape[0], int(round(pad * sample_rate))), dtype=np.float32)
        samples = np.concatenate([silence, samples], axis=1)
    data = np.ascontiguousarray(np.clip(samples, -1.0, 1.0).T, dtype=np.float32)
    result = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-f", "f32le", "-ar", str(sample_rate), "-ac", str(samples.shape[0]),
         "-i", "-", "-c:a", "libvorbis", "-q:a", str(quality), str(path)],
        input=data.tobytes(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise AudioError(f"Falha ao gravar {path.name}: {result.stderr.decode(errors='replace').strip()}")


def extract_cover(source: Path, target: Path, size: int = 512) -> bool:
    """Save the embedded cover art (if any) as a PNG no larger than ``size`` pixels."""
    result = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-i", str(source), "-map", "0:v:0", "-frames:v", "1",
         "-vf", f"scale={size}:{size}:force_original_aspect_ratio=decrease", str(target)],
        capture_output=True,
    )
    return result.returncode == 0 and target.exists() and target.stat().st_size > 0


def convert_image(source: Path, target: Path, size: int = 512) -> bool:
    """Convert an uploaded image (jpg/png/webp...) to the PNG cover Clone Hero expects."""
    return extract_cover(source, target, size)


def _wrap(text: str, width: int, max_lines: int = 3) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])[: width - 1] + "…"]
    return lines


def make_cover(title: str, artist: str, target: Path, size: int = 512) -> bool:
    """Draw a plain cover with the title and artist for songs without album art."""
    title_lines = _wrap(title, 16)
    title_size = int(min(56, 460 / (0.56 * max(len(line) for line in title_lines))))
    workdir = target.parent
    title_file, artist_file = workdir / ".cover_title.txt", workdir / ".cover_artist.txt"
    title_file.write_text("\n".join(title_lines), encoding="utf-8")
    artist_file.write_text(" ".join(_wrap(artist, 40, 1)), encoding="utf-8")
    block = len(title_lines) * title_size * 1.2
    draw = (
        f"drawtext=textfile='{title_file.name}':expansion=none:fontcolor=white:fontsize={title_size}"
        f":line_spacing={int(title_size * 0.2)}:x=(w-text_w)/2:y=(h-{block:.0f})/2-20,"
        f"drawtext=textfile='{artist_file.name}':expansion=none:fontcolor=0xffd166:fontsize=28"
        f":x=(w-text_w)/2:y=(h+{block:.0f})/2+12,"
        "drawbox=x=40:y=ih-56:w=iw-80:h=2:color=0xffffff@0.35:t=fill"
    )
    try:
        result = subprocess.run(
            [_require("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
             "-i", f"gradients=s={size}x{size}:c0=0x141e30:c1=0x35577d:x0=0:y0=0:x1={size}:y1={size}:duration=1",
             "-vf", draw, "-frames:v", "1", target.name],
            capture_output=True,
            cwd=workdir,
        )
    finally:
        title_file.unlink(missing_ok=True)
        artist_file.unlink(missing_ok=True)
    return result.returncode == 0 and target.exists()
