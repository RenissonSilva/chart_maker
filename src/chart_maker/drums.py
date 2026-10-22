"""Expert pro-drums track from the separated drum stem.

Hits come from ADTOF (Zehren et al., 2021/2023), a drum transcription model
trained on crowdsourced rhythm-game charts, which outputs five classes: kick,
snare, toms, hi-hat and cymbals (crash and ride together). The rest is mapping
to Clone Hero lanes and playability rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import chart as ch
from .quantize import quantize
from .tempo import Timing

ADTOF_FPS = 100
KINDS = ("kick", "snare", "tom", "hihat", "cymbal")  # ADTOF output order (GM 35, 38, 47, 42, 49)
THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)  # tuned by the ADTOF authors

# Which hand notes survive when more than two land on the same tick.
HAND_PRIORITY = {"crash": 0, "cymbal": 0, "snare": 1, "ride": 2, "tom": 3, "hihat": 4}
CYMBAL_LANE = {"hihat": ch.DRUM_YELLOW, "ride": ch.DRUM_BLUE, "crash": ch.DRUM_GREEN, "cymbal": ch.DRUM_GREEN}
DOUBLE_KICK_INTERVAL = 0.17  # seconds; faster kicks alternate with Expert+ (2x) kicks

_model = None


@dataclass
class DrumHit:
    time: float
    kind: str  # kick | snare | tom | hihat | crash | ride
    strength: float
    tom_lane: int = ch.DRUM_BLUE


def _adtof_model():
    global _model
    if _model is None:
        import contextlib
        import io

        import adtof_pytorch as ad

        model = ad.create_frame_rnn_model(ad.calculate_n_bins())
        with contextlib.redirect_stdout(io.StringIO()):  # the loader prints the weights path
            _model = ad.load_pytorch_weights(model, ad.get_default_weights_path(), strict=True).eval()
    return _model


def activations(mono: np.ndarray, sr: int) -> np.ndarray:
    """ADTOF frame activations (100 fps, 5 classes) for a mono 44.1 kHz signal."""
    import torch
    from adtof_pytorch.audio import create_adtof_processor

    if sr != 44100:
        raise ValueError("ADTOF expects 44.1 kHz audio")
    processor = create_adtof_processor()
    spec = processor.apply_filterbank(processor.compute_stft(mono.astype(np.float32)))
    x = torch.from_numpy(np.ascontiguousarray(spec.T[None, :, :, None]))
    with torch.inference_mode():
        return _adtof_model()(x)[0].numpy()


def pick_hits(act: np.ndarray) -> list[DrumHit]:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor

    hits = []
    for k, kind in enumerate(KINDS):
        picker = NotePeakPickingProcessor(threshold=THRESHOLDS[k], pre_avg=0.1, post_avg=0.01, pre_max=0.02,
                                          post_max=0.01, combine=0.02, fps=ADTOF_FPS)
        for t, _ in picker.process(act[:, k]):
            frame = min(int(round(t * ADTOF_FPS)), len(act) - 1)
            hits.append(DrumHit(float(t), kind, float(act[frame, k])))
    return sorted(hits, key=lambda h: h.time)


def assign_tom_lanes(hits: list[DrumHit], mono: np.ndarray, sr: int) -> None:
    """Split toms into high/mid/floor (yellow/blue/green) by the pitch of each hit."""
    import librosa

    toms = [h for h in hits if h.kind == "tom"]
    if not toms:
        return
    n_fft, hop = 4096, 441
    spec = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= 60) & (freqs <= 500)
    pitches = []
    for hit in toms:
        f = int(round(hit.time * sr / hop))
        after = spec[band, f + 1 : f + 8].mean(axis=1) if f + 8 < spec.shape[1] else spec[band, -7:].mean(axis=1)
        before = spec[band, max(0, f - 3) : max(1, f - 1)].mean(axis=1)
        gain = np.clip(after - before, 0, None)
        pitches.append(np.log2(freqs[band][int(np.argmax(gain))]) * 12 if gain.max() > 0 else np.nan)
    pitches = np.array(pitches)
    valid = ~np.isnan(pitches)
    if valid.sum() < 2:
        return
    spread = np.percentile(pitches[valid], 90) - np.percentile(pitches[valid], 10)
    k = 3 if spread > 7 else 2 if spread > 2.5 else 1
    lanes = {3: [ch.DRUM_YELLOW, ch.DRUM_BLUE, ch.DRUM_GREEN], 2: [ch.DRUM_BLUE, ch.DRUM_GREEN], 1: [ch.DRUM_BLUE]}[k]
    centers = np.percentile(pitches[valid], np.linspace(85, 15, k))  # high to low
    for _ in range(20):  # 1-D k-means
        groups = np.argmin(np.abs(pitches[valid][:, None] - centers[None, :]), axis=1)
        centers = np.array([pitches[valid][groups == g].mean() if np.any(groups == g) else centers[g] for g in range(k)])
    order = np.argsort(-centers)  # highest pitch first
    rank = {int(g): i for i, g in enumerate(order)}
    median = float(np.median(pitches[valid]))
    for hit, p in zip(toms, pitches):
        g = int(np.argmin(np.abs(centers - (median if np.isnan(p) else p))))
        hit.tom_lane = lanes[rank[g]]


def split_cymbals(hits: list[DrumHit], beat_period: float) -> None:
    """ADTOF merges crash and ride: steady runs of cymbal hits are a ride pattern, the rest are crashes."""
    cymbals = [h for h in hits if h.kind == "cymbal"]
    i = 0
    while i < len(cymbals):
        j = i
        while j + 1 < len(cymbals) and cymbals[j + 1].time - cymbals[j].time <= 1.05 * beat_period:
            j += 1
        run = cymbals[i : j + 1]
        gaps = np.diff([h.time for h in run])
        is_ride = len(run) >= 6 and np.median(gaps) <= 0.8 * beat_period or len(run) >= 10
        for h in run:
            h.kind = "ride" if is_ride else "crash"
        i = j + 1


def _hand_lanes(kinds: dict[str, DrumHit]) -> list[tuple[int, bool]]:
    """(lane, is_cymbal) for the hand notes on one tick, obeying two hands and color conflicts."""
    ranked = sorted((k for k in kinds if k != "kick"), key=lambda k: HAND_PRIORITY[k])[:2]
    out: list[tuple[int, bool]] = []
    for kind in ranked:
        if kind == "snare":
            out.append((ch.DRUM_RED, False))
        elif kind in CYMBAL_LANE:
            out.append((CYMBAL_LANE[kind], True))
    for kind in ranked:
        if kind == "tom":
            used = {lane for lane, _ in out}
            preferred = kinds["tom"].tom_lane
            options = [preferred] + [lane for lane in (ch.DRUM_GREEN, ch.DRUM_BLUE, ch.DRUM_YELLOW) if lane != preferred]
            lane = next((lane for lane in options if lane not in used), None)
            if lane is not None:
                out.append((lane, False))
    return out


def build_track(hits: list[DrumHit], timing: Timing, name: str = "ExpertDrums") -> ch.Track:
    if not hits:
        return ch.Track(name)
    times = np.array([h.time for h in hits])
    res = timing.tempo_map.resolution
    raw_ticks = timing.to_ticks(times)
    beat_seconds = 60.0 / timing.tempo_map.bpm_at(raw_ticks)
    ticks = quantize(raw_ticks, res, beat_seconds)

    by_tick: dict[int, dict[str, DrumHit]] = {}
    for tick, hit in zip(ticks, hits):
        slot = by_tick.setdefault(int(tick), {})
        if hit.kind not in slot or hit.strength > slot[hit.kind].strength:
            slot[hit.kind] = hit

    # Fast kick runs: every other kick becomes an Expert+ (2x kick) note.
    kick_ticks = sorted(t for t, kinds in by_tick.items() if "kick" in kinds)
    kick_times = timing.tempo_map.tick_to_time(np.array(kick_ticks, dtype=float))
    double = set()
    run = []
    for tick, t in zip(kick_ticks, kick_times):
        if run and t - run[-1][1] < DOUBLE_KICK_INTERVAL:
            run.append((tick, t))
            continue
        if len(run) >= 3:
            double.update(tk for tk, _ in run[1::2])
        run = [(tick, t)]
    if len(run) >= 3:
        double.update(tk for tk, _ in run[1::2])

    notes = []
    for tick in sorted(by_tick):
        kinds = by_tick[tick]
        lanes, flags = [], []
        if "kick" in kinds:
            lanes.append(ch.DOUBLE_KICK if tick in double else ch.KICK)
        for lane, cymbal in _hand_lanes(kinds):
            lanes.append(lane)
            if cymbal:
                flags.append(ch.CYMBAL_FLAG[lane])
        if lanes:
            notes.append(ch.Note(tick, tuple(sorted(lanes)), 0, tuple(sorted(flags))))
    return ch.Track(name, notes)
