"""Beat/downbeat tracking and construction of the chart tempo map.

Beat This! (Foscarin et al., ISMIR 2024) provides frame-wise beat and downbeat
activations. Its own peak picking can be erratic on hard material, so beats are
decoded with a dynamic-programming tracker that enforces a stable tempo (Ellis,
2007) and bars with a small HMM over beat positions. The beat sequence is then
approximated by as few constant-tempo segments as possible, which is what the
.chart tempo map stores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chart import RESOLUTION

log = logging.getLogger(__name__)

BEAT_FPS = 50  # Beat This! works at 22050 Hz with a 441-sample hop
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
MIRROR_REPO = "safe-models/beat-this-final0"
MIRROR_SHA256 = "f758c77ba5d862cad3718a95705964e9af55298b1fb3544c62c9d79184ea0f2e"


class TempoMap:
    """Converts between seconds and ticks using exactly the tempo events written to the chart."""

    def __init__(self, tempos: list[tuple[int, int]], resolution: int = RESOLUTION):
        if not tempos or tempos[0][0] != 0:
            raise ValueError("the tempo map needs an event at tick 0")
        self.tempos = tempos
        self.resolution = resolution
        self._ticks = np.array([tick for tick, _ in tempos], dtype=float)
        self._spt = np.array([60000.0 / milli_bpm / resolution for _, milli_bpm in tempos])
        self._starts = np.concatenate([[0.0], np.cumsum(np.diff(self._ticks) * self._spt[:-1])])

    def tick_to_time(self, tick):
        tick = np.asarray(tick, dtype=float)
        i = np.clip(np.searchsorted(self._ticks, tick, side="right") - 1, 0, None)
        return self._starts[i] + (tick - self._ticks[i]) * self._spt[i]

    def time_to_tick(self, time):
        time = np.asarray(time, dtype=float)
        i = np.clip(np.searchsorted(self._starts, time, side="right") - 1, 0, None)
        return self._ticks[i] + (time - self._starts[i]) / self._spt[i]

    def bpm_at(self, tick) -> np.ndarray:
        i = np.clip(np.searchsorted(self._ticks, np.asarray(tick, dtype=float), side="right") - 1, 0, None)
        return np.array([milli for _, milli in self.tempos])[i] / 1000.0


@dataclass
class Beats:
    times: np.ndarray  # seconds, original (unpadded) audio timeline
    downbeats: np.ndarray  # indices into ``times`` where a bar starts
    source: str


@dataclass
class Timing:
    tempo_map: TempoMap
    time_signatures: list[tuple[int, int]]
    pad: float  # seconds of silence prepended to every audio file
    bar_ticks: np.ndarray  # tick of every bar line
    beat_ticks: np.ndarray = field(default_factory=lambda: np.zeros(0))  # tick of each detected beat

    def to_ticks(self, times) -> np.ndarray:
        """Fractional ticks for times on the original (unpadded) audio timeline."""
        return self.tempo_map.time_to_tick(np.asarray(times, dtype=float) + self.pad)


# --------------------------------------------------------------------------- model loading


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_checkpoint(download: bool) -> Path:
    import torch

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
    if not checkpoint.exists():
        if not download:
            raise FileNotFoundError(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix(".part")
        # torch.hub has no timeout; the JKU server sometimes hangs or answers 503.
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=30) as response, open(partial, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        partial.replace(checkpoint)
    return checkpoint


def _mirror_model(download: bool):
    from beat_this.model.beat_tracker import BeatThis
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import logging as hf_logging
    from safetensors.torch import load_file

    hf_logging.set_verbosity_error()
    weights = Path(hf_hub_download(MIRROR_REPO, "model.safetensors", local_files_only=not download))
    if _sha256(weights) != MIRROR_SHA256:
        raise ValueError("checksum mismatch")
    with open(hf_hub_download(MIRROR_REPO, "config.json", local_files_only=not download)) as handle:
        model = BeatThis(**json.load(handle))
    result = model.load_state_dict(load_file(weights), strict=False)
    # The rotary-embedding frequencies are deterministic buffers shared between layers;
    # safetensors drops the duplicated ones.
    unexpected = result.unexpected_keys + [k for k in result.missing_keys if not k.endswith("rotary_embed.freqs")]
    if unexpected:
        raise ValueError(f"unexpected state dict keys: {unexpected[:5]}")
    return model


def load_beat_model(device: str = "cpu"):
    """Load Beat This! ``final0`` from the official server or its Hugging Face mirror.

    Local copies are tried before any download. Returns None when nothing works,
    and the caller falls back to librosa.
    """
    from beat_this.inference import load_model

    errors = []
    for download in (False, True):
        try:
            return load_model(str(_official_checkpoint(download)), device)
        except Exception as exc:  # noqa: BLE001 - any failure falls through to the next source
            errors.append(f"oficial: {exc}")
        try:
            return _mirror_model(download).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"espelho: {exc}")
    log.warning("Beat This! indisponível (%s); usando librosa", "; ".join(errors[-2:]))
    return None


def beat_activations(model, mono: np.ndarray, sr: int, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise (50 fps) beat and downbeat probabilities from Beat This!."""
    import soxr
    import torch
    from beat_this.inference import split_predict_aggregate
    from beat_this.preprocessing import LogMelSpect

    signal = soxr.resample(mono.astype(np.float32), sr, 22050) if sr != 22050 else mono
    with torch.inference_mode():
        spect = LogMelSpect(device=device)(torch.tensor(signal, dtype=torch.float32, device=device))
        pred = split_predict_aggregate(spect, chunk_size=1500, border_size=6, overlap_mode="keep_first", model=model)
        beat = torch.sigmoid(pred["beat"]).float().cpu().numpy()
        down = torch.sigmoid(pred["downbeat"]).float().cpu().numpy()
    return beat, down


# --------------------------------------------------------------------------- beat decoding


def _peaks(act: np.ndarray, threshold: float, radius: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d

    is_peak = (act >= threshold) & (act == maximum_filter1d(act, 2 * radius + 1))
    idx = np.flatnonzero(is_peak)
    if idx.size:  # plateaus produce adjacent duplicates
        idx = idx[np.concatenate([[True], np.diff(idx) > radius])]
    return idx


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    spectrum = np.fft.rfft(x, 2 * len(x))
    return np.fft.irfft(np.abs(spectrum) ** 2)[: len(x)]


def _refine_peak(values: np.ndarray, i: int) -> float:
    """Parabolic interpolation of a local maximum at index ``i``."""
    if 0 < i < len(values) - 1:
        a, b, c = values[i - 1], values[i], values[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return i + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float(i)


def estimate_period(act: np.ndarray, fps: float, hint: float | None = None) -> float:
    """Dominant beat period in frames. ``hint`` (frames) pins the metrical level."""
    ac = _autocorr(act)
    if hint is not None:
        lo, hi = int(hint * 0.9), int(np.ceil(hint * 1.1)) + 1
        weights = np.ones(hi - lo)
    else:
        lo, hi = int(fps * 60 / 210), int(fps * 60 / 55) + 1
        bpm = 60 * fps / np.arange(lo, hi)
        weights = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.9) ** 2)  # mild preference for ~120 BPM
    hi = min(hi, len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi] * weights[: hi - lo]))
    return _refine_peak(ac, lag)


def local_periods(act: np.ndarray, fps: float, period: float, window_s: float = 12.0) -> np.ndarray:
    """Per-frame beat period that follows tempo changes of up to ±25% around ``period``."""
    from scipy.ndimage import median_filter

    n = len(act)
    win, hop = int(window_s * fps), int(2 * fps)
    lo, hi = int(period * 0.8), int(np.ceil(period * 1.25)) + 1
    if n < 2 * win:
        return np.full(n, period)
    centers, values = [], []
    for start in range(0, n - win + 1, hop):
        ac = _autocorr(act[start : start + win])
        lag = lo + int(np.argmax(ac[lo:hi]))
        centers.append(start + win / 2)
        values.append(_refine_peak(ac, lag))
    values = median_filter(np.array(values), size=5, mode="nearest")
    return np.interp(np.arange(n), centers, values)


def dp_beats(act: np.ndarray, periods: np.ndarray, tightness: float = 100.0) -> np.ndarray:
    """Ellis-style dynamic programming beat tracker; returns fractional beat frames."""
    n = len(act)
    cum = np.zeros(n)
    back = np.full(n, -1)
    for t in range(n):
        p = periods[t]
        lo, hi = max(1, int(round(p * 0.5))), int(round(p * 2.0))
        first = max(0, t - hi)
        last = t - lo
        if last < first:
            cum[t] = act[t]
            continue
        prev = np.arange(first, last + 1)
        scores = cum[prev] - tightness * np.log((t - prev) / p) ** 2
        best = int(np.argmax(scores))
        if scores[best] > 0:
            cum[t] = act[t] + scores[best]
            back[t] = prev[best]
        else:
            cum[t] = act[t]
    # The best path ends at the last real beat: after the music stops, beats only add ~0.
    end = int(np.flatnonzero(cum >= cum.max() - 0.5)[0])
    path = [end]
    while back[path[-1]] >= 0:
        path.append(int(back[path[-1]]))
    frames = np.array(path[::-1])
    return np.array([_refine_peak(act, int(f)) for f in frames])


def _bar_viterbi(p: np.ndarray, meter: int, irregular_cost: float = 8.0) -> tuple[np.ndarray, float]:
    """Positions in the bar for each beat (0 = downbeat), allowing occasional irregular bars."""
    states = meter + 2  # bars may be up to two beats longer than the meter
    eps = 1e-4
    emit = np.empty((len(p), states))
    emit[:, 0] = -np.log(p + eps)
    emit[:, 1:] = -np.log(1 - p + eps)[:, None]
    trans = np.full((states, states), np.inf)
    for s in range(states - 1):
        trans[s, s + 1] = 0.0 if s + 1 < meter else irregular_cost
    for s in range(1, states):
        trans[s, 0] = 0.0 if s >= meter - 1 else irregular_cost
    cost = emit[0] + np.where(np.arange(states) < meter, 0.0, irregular_cost)
    back = np.zeros((len(p), states), dtype=int)
    for i in range(1, len(p)):
        total = cost[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        cost = total[back[i], np.arange(states)] + emit[i]
    path = np.zeros(len(p), dtype=int)
    path[-1] = int(np.argmin(cost))
    for i in range(len(p) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, float(cost.min())


def decode_downbeats(down_at_beats: np.ndarray) -> np.ndarray:
    """Choose 3/4 or 4/4 (preferring 4/4) and return the beat indices that start bars."""
    if len(down_at_beats) < 8:
        return np.arange(0, len(down_at_beats), 4)
    best = None
    for meter, prior in ((4, 0.0), (3, 3.0)):
        path, cost = _bar_viterbi(down_at_beats, meter)
        if best is None or cost + prior < best[0]:
            best = (cost + prior, path)
    return np.flatnonzero(best[1] == 0)


def track_beats(mono: np.ndarray, sr: int, model, device: str = "cpu") -> Beats:
    """Beat times and bar starts for a mono signal on the original timeline.

    ``model`` comes from :func:`load_beat_model`; ``None`` selects the librosa fallback.
    """
    if model is None:
        return _librosa_beats(mono, sr)
    beat, down = beat_activations(model, mono, sr, device)
    act = np.maximum(beat, down)
    peaks = _peaks(act, 0.5, 3)
    hint = float(np.median(np.diff(peaks))) if len(peaks) > 8 else None
    period = estimate_period(act, BEAT_FPS, hint)
    frames = dp_beats(act, local_periods(act, BEAT_FPS, period))
    frames = _trim_silent_ends(frames, act)
    idx = np.clip(np.round(frames).astype(int), 0, len(down) - 1)
    # Downbeat evidence: the strongest response within one frame of the beat.
    down_at = np.max([down[np.clip(idx + d, 0, len(down) - 1)] for d in (-1, 0, 1)], axis=0)
    return Beats(frames / BEAT_FPS, decode_downbeats(down_at), "beat_this")


def align_to_onsets(beats: Beats, onsets: np.ndarray, max_shift: float = 0.04) -> Beats:
    """Shift the beat grid so that rhythmic onsets (kick and snare) sit on it on average.

    Beat This! puts beats ~15 ms after the drum attacks. Notes are snapped to the
    grid, so without this correction the whole chart would lag the audio.
    """
    times, onsets = beats.times, np.asarray(onsets, dtype=float)
    if len(times) < 8:
        return beats
    onsets = onsets[(onsets > times[0]) & (onsets < times[-1])]
    if len(onsets) < 16:
        return beats
    pos = np.interp(onsets, times, np.arange(len(times)))
    period = np.diff(times)[np.clip(np.floor(pos).astype(int), 0, len(times) - 2)]
    offset = (pos * 4 - np.round(pos * 4)) / 4 * period  # seconds from the nearest 16th
    near = np.abs(offset) < 0.06
    if near.sum() < 16:
        return beats
    shift = float(np.clip(np.median(offset[near]), -max_shift, max_shift))
    return Beats(times + shift, beats.downbeats, beats.source)


def _trim_silent_ends(frames: np.ndarray, act: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    idx = np.clip(np.round(frames).astype(int), 0, len(act) - 1)
    strong = np.flatnonzero(act[idx] >= threshold)
    if strong.size < 2:
        return frames
    return frames[strong[0] : strong[-1] + 1]


def _librosa_beats(mono: np.ndarray, sr: int) -> Beats:
    import librosa

    hop = 512
    fps = sr / hop
    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    env = env / (np.percentile(env, 99) + 1e-9)
    period = estimate_period(env, fps)
    frames = dp_beats(np.clip(env, 0, 1), local_periods(env, fps, period), tightness=200.0)
    times = frames / fps
    # Bar phase: the kick (low band) usually lands on beat 1.
    low = librosa.onset.onset_strength(
        S=librosa.power_to_db(librosa.feature.melspectrogram(y=mono, sr=sr, hop_length=hop, fmax=150, n_mels=8))
    )
    at = low[np.clip(np.round(frames).astype(int), 0, len(low) - 1)]
    phase = int(np.argmax([at[k::4].sum() for k in range(4)]))
    return Beats(times, np.arange(phase, len(times), 4), "librosa")


# --------------------------------------------------------------------------- tempo map


def _fits(resid: np.ndarray, drift_tol: float, max_tol: float) -> bool:
    """Beat detection jitters by ~10 ms, so judge the 8-beat moving average (real tempo drift)."""
    if np.max(np.abs(resid)) > max_tol:
        return False
    w = min(8, len(resid))
    return float(np.max(np.abs(np.convolve(resid, np.ones(w) / w, mode="valid")))) <= drift_tol


def _segment_fit(targets: np.ndarray, start: float | None, bpm: float | None = None) -> tuple[float, float]:
    """Least-squares (start time, seconds per beat); ``start``/``bpm`` fix those parameters."""
    rel = np.arange(len(targets), dtype=float)
    if bpm is not None:
        spb = 60.0 / bpm
        return (float(np.mean(targets - rel * spb)) if start is None else start), spb
    if start is None:
        spb, intercept = np.polyfit(rel, targets, 1)
        return float(intercept), float(spb)
    return start, float(np.dot(rel, targets - start) / np.dot(rel, rel))


def _tempo_segments(
    times: np.ndarray, drift_tol: float = 0.015, max_tol: float = 0.05, global_tol: float = 0.025
) -> tuple[float, list[tuple[int, float]]]:
    """Time of beat 0 and ``(first beat index, bpm)`` constant-tempo segments covering the beats."""

    def resid(seg: np.ndarray, fit: tuple[float, float]) -> np.ndarray:
        return fit[0] + np.arange(len(seg)) * fit[1] - seg

    def nice(seg: np.ndarray, fit: tuple[float, float], start: float | None, tol: float) -> tuple[float, float]:
        # Produced music sits on an integer BPM; prefer it whenever it fits as well.
        bpm = 60.0 / fit[1]
        for step in (1.0, 0.5):
            candidate = round(bpm / step) * step
            nice_fit = _segment_fit(seg, start, candidate)
            if _fits(resid(seg, nice_fit), tol, max_tol):
                return nice_fit[0], candidate
        return fit[0], round(bpm, 3)

    fit = _segment_fit(times, None)
    if _fits(resid(times, fit), global_tol, max_tol * 1.5):
        first, bpm = nice(times, fit, None, global_tol)
        return first, [(0, bpm)]

    segments: list[tuple[int, float]] = []
    first = None
    start: float | None = None
    s, n = 0, len(times)
    while s < n - 1:
        e = s + 1
        fit = _segment_fit(times[s : e + 1], start)
        while e + 1 < n:
            candidate = _segment_fit(times[s : e + 2], start)
            if not _fits(resid(times[s : e + 2], candidate), drift_tol, max_tol):
                break
            e, fit = e + 1, candidate
        seg_start, bpm = nice(times[s : e + 1], fit, start, drift_tol)
        if first is None:
            first = start = seg_start
        if not segments or segments[-1][1] != bpm:
            segments.append((s, bpm))
        start += (e - s) * 60.0 / bpm
        s = e
    return float(first), segments


def build_timing(
    beats: Beats,
    duration: float,
    first_note: float | None,
    lead_in: float = 2.5,
    resolution: int = RESOLUTION,
) -> Timing:
    """Tempo map, time signatures and audio padding for the detected beats.

    Silence is prepended so that tick 0 falls exactly on an extrapolated beat, the
    first bar line lands on the first detected downbeat and the first note appears
    at least ``lead_in`` seconds into the song.
    """
    times = np.asarray(beats.times, dtype=float)
    if len(times) < 8:  # no usable pulse: a plain 120 BPM grid
        times = np.arange(0.0, max(duration, 1.0), 0.5)
        downbeats = np.arange(0, len(times), 4)
    else:
        downbeats = np.asarray(beats.downbeats, dtype=int)
        if len(downbeats) == 0:
            downbeats = np.arange(0, len(times), 4)

    first_beat, tempos = _tempo_segments(times)
    period = 60.0 / tempos[0][1]
    first_bar = int(downbeats[0])
    lengths = np.diff(downbeats)
    meter0 = int(np.bincount(lengths).argmax()) if len(lengths) else 4
    earliest = min(first_beat, first_note) if first_note is not None else first_beat
    need = max(first_beat / period, (lead_in + first_beat - earliest) / period)
    virtual = int(np.ceil(need - 1e-9))
    virtual += (-(virtual + first_bar)) % meter0  # first detected downbeat on a bar line
    pad = virtual * period - first_beat

    tempo_events = [(0 if i == 0 else (virtual + beat) * resolution, int(round(bpm * 1000))) for i, (beat, bpm) in enumerate(tempos)]
    tempo_map = TempoMap(tempo_events, resolution)

    signatures = [(0, meter0)]
    # Bar lines in the extrapolated lead-in, as (negative) indices of real beats.
    bar_beats = [b - virtual for b in range(0, first_bar + virtual, meter0)]
    for i, beat in enumerate(downbeats):
        length = int(downbeats[i + 1] - beat) if i + 1 < len(downbeats) else signatures[-1][1]
        if length != signatures[-1][1]:
            signatures.append(((virtual + int(beat)) * resolution, length))
        bar_beats.append(int(beat))
    # Keep bar lines going after the last downbeat until the end of the audio.
    end_tick = float(tempo_map.time_to_tick(duration + pad))
    last_len = signatures[-1][1]
    while (virtual + bar_beats[-1] + last_len) * resolution < end_tick + resolution * last_len:
        bar_beats.append(bar_beats[-1] + last_len)
    bar_ticks = (np.array(bar_beats) + virtual) * resolution
    beat_ticks = (np.arange(len(times)) + virtual) * resolution
    return Timing(tempo_map, signatures, float(pad), bar_ticks.astype(int), beat_ticks.astype(int))
