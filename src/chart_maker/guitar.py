"""Expert 5-fret guitar track from the separated guitar stem.

Notes are found with spectral-flux onset detection. Each note gets a pitch from
harmonic summation over a constant-Q spectrum, and a chord size from how much
harmonic energy is left after removing the strongest harmonic series. Frets are
chosen by dynamic programming so that the chart follows the melodic contour:
same pitch → same fret, higher pitch → further right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import chart as ch
from .quantize import quantize
from .tempo import Timing

SR = 22050
HOP = 256
FMIN_MIDI = 40  # E2, lowest note of a standard-tuned guitar
BINS_PER_SEMITONE = 3
N_BINS = 12 * BINS_PER_SEMITONE * 6
N_CANDIDATES = 12 * BINS_PER_SEMITONE * 4 + 1  # f0 search range E2..E6
HARMONICS = 8
HARMONIC_OFFSETS = np.round(12 * BINS_PER_SEMITONE * np.log2(np.arange(1, HARMONICS + 1))).astype(int)
HARMONIC_WEIGHTS = 0.85 ** np.arange(HARMONICS)

# Chord detection compares spectral-peak energy per pitch class with the root's class,
# above a floor measured on the semitone and tritone (which no single note, power chord or
# triad excites, so they only carry bleed and noise). Within 8 harmonics a single note
# reaches the fifth only through harmonics 3 and 6 (contrast ~0.35 even with heavy
# distortion) and the thirds barely at all; real chords sit well above that.
FIFTH_CONTRAST = 0.5
THIRD_CONTRAST = 0.55
FULL_CHORD_CONTRAST = 0.7  # both fifth and third this strong -> 3-note chord (sustained notes only)
MIN_SUSTAIN_BEATS = 0.75


@dataclass
class GuitarNote:
    time: float
    end: float
    pitch: float  # MIDI note number
    chord: int  # 1 = single note, 2-3 = chord size
    strength: float
    ratios: np.ndarray = field(default_factory=lambda: np.zeros(12), repr=False)


SOURCES = ("auto", "guitar", "other", "piano", "guitar+other", "guitar+piano")


def choose_source(stems: dict[str, np.ndarray], mode: str = "auto") -> tuple[np.ndarray, str]:
    """Pick the (stereo) stem to chart and say which one was used.

    htdemucs_6s often leaves distorted guitars (and synth leads in songs without
    guitar) in "other", and songs without guitar are charted from their lead
    instrument, so in auto mode the loudest of "other"/"piano" is added when the
    guitar stem is much weaker.
    """
    if mode not in SOURCES:
        raise ValueError(f"unknown guitar source {mode!r}; use one of {SOURCES}")

    def level(name: str) -> float:
        return float(10 * np.log10(np.mean(stems[name].astype(np.float64) ** 2) + 1e-12))

    if mode == "auto":
        backup = max(("other", "piano"), key=level)
        mode = "guitar" if level("guitar") >= level(backup) - 6.0 else f"guitar+{backup}"
    if "+" in mode:
        first, second = mode.split("+")
        return stems[first] + stems[second], mode
    return stems[mode], mode


def _salience(x: np.ndarray) -> np.ndarray:
    widened = np.maximum(x, np.maximum(np.roll(x, 1), np.roll(x, -1)))  # tolerate slight mistuning
    widened[0] = x[0]
    sal = np.zeros(N_CANDIDATES)
    for off, w in zip(HARMONIC_OFFSETS, HARMONIC_WEIGHTS):
        idx = np.arange(N_CANDIDATES) + off
        ok = idx < len(x)
        sal[ok] += w * widened[idx[ok]]
    return sal


def pitch_class_ratios(x: np.ndarray, k1: int) -> np.ndarray:
    """Peak energy of each pitch class relative to the class of bin ``k1`` (12 values, index = interval)."""
    hi = min(len(x) - 1, k1 + HARMONIC_OFFSETS[-1] + 1)
    seg = np.concatenate([[0.0], x[k1:hi], [0.0]])
    inner = seg[1:-1]
    is_peak = (inner >= seg[:-2]) & (inner > seg[2:]) & (inner > 0.1 * inner.max())
    pc = np.zeros(12)
    for b in np.flatnonzero(is_peak):
        pc[int(round(b / BINS_PER_SEMITONE)) % 12] += inner[b]
    return pc / (pc[0] + 1e-9)


def analyze_pitch(spectrum: np.ndarray) -> tuple[float, np.ndarray]:
    """(MIDI pitch, pitch-class ratios) from one averaged constant-Q magnitude spectrum."""
    x = np.sqrt(np.maximum(spectrum, 0.0))
    sal = _salience(x)
    k1 = int(np.argmax(sal))
    if sal[k1] <= 0:
        return float("nan"), np.zeros(12)
    pitch = FMIN_MIDI + _parabolic(sal, k1) / BINS_PER_SEMITONE
    return float(pitch), pitch_class_ratios(x, k1)


def chord_size(r: np.ndarray) -> int:
    floor = (r[1] + r[6]) / 2
    fifth = max(r[7], r[5]) - floor  # r[5]: the detected root was really the fifth of a power chord
    third = max(r[3], r[4]) - floor
    if fifth > FULL_CHORD_CONTRAST and third > FULL_CHORD_CONTRAST:
        return 3
    return 2 if fifth > FIFTH_CONTRAST or third > THIRD_CONTRAST else 1


def _parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return i + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float(i)


def _onset_envelope(y: np.ndarray, cqt: np.ndarray) -> np.ndarray:
    """Spectral flux on a mel spectrogram (pick attacks) plus on the constant-Q spectrum.

    Mel bands are wider than a semitone in the low register, so legato pitch changes
    (hammer-ons, slides) barely register there; with three bins per semitone the CQT
    flux catches them. Each half is scaled by its own 99th percentile.
    """
    import librosa
    from scipy.ndimage import maximum_filter

    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128,
                                                             fmin=60, fmax=8000))
    mel_flux = librosa.onset.onset_strength(S=mel, sr=SR, lag=2, max_size=3)
    # Above ~200 Hz the constant-Q filters are short enough for precise onset times.
    level = librosa.amplitude_to_db(cqt[12 * BINS_PER_SEMITONE + 10 :], ref=np.max, top_db=80)
    ref = maximum_filter(level, size=(3, 1))
    cqt_flux = np.concatenate([[0.0, 0.0], np.maximum(0.0, level[:, 2:] - ref[:, :-2]).mean(axis=0)])
    n = min(len(mel_flux), len(cqt_flux))
    return 0.5 * (mel_flux[:n] / (np.percentile(mel_flux, 99) + 1e-9) + cqt_flux[:n] / (np.percentile(cqt_flux, 99) + 1e-9))


def detect_notes(mono: np.ndarray, sr: int) -> list[GuitarNote]:
    import librosa

    y = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else mono.astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    floor = np.percentile(rms_db, 95) - 30.0
    if np.percentile(rms_db, 95) < -60:  # silent stem
        return []

    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(FMIN_MIDI), n_bins=N_BINS,
                             bins_per_octave=12 * BINS_PER_SEMITONE, filter_scale=0.5))
    n = cqt.shape[1]
    env = _onset_envelope(y, cqt)[:n]
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.12, wait=4)
    peaks = np.array([p for p in peaks if rms_db[min(p + 2, len(rms_db) - 1)] > floor], dtype=int)
    if peaks.size == 0:
        return []

    notes = []
    for i, f in enumerate(peaks):
        nxt = peaks[i + 1] if i + 1 < len(peaks) else n
        a, b = min(f + 2, n - 1), max(min(f + 20, nxt - 1), min(f + 3, n))
        current = cqt[:, a:b].mean(axis=1)
        # Only what was just played: the tail of the previous note would look like a second voice.
        novel = np.maximum(current - 0.8 * cqt[:, max(0, f - 3) : max(1, f)].mean(axis=1), 0.0)
        pitch, ratios = analyze_pitch(novel if novel.sum() > 0.3 * current.sum() else current)
        if np.isnan(pitch):
            continue
        # The note rings until its level falls 12 dB below its attack, or the next onset.
        seg = rms_db[f:nxt]
        peak = seg[: max(1, min(len(seg), 9))].max()
        below = np.flatnonzero(seg[3:] < peak - 12.0)
        end_frame = f + 3 + int(below[0]) if below.size else nxt
        notes.append(GuitarNote(
            time=float(librosa.frames_to_time(f, sr=SR, hop_length=HOP)),
            end=float(librosa.frames_to_time(end_frame, sr=SR, hop_length=HOP)),
            pitch=pitch,
            chord=chord_size(ratios),
            strength=float(env[f]),
            ratios=ratios,
        ))
    _fix_octave_errors(notes)
    return notes


def _fix_octave_errors(notes: list[GuitarNote]) -> None:
    """A note an octave away from two agreeing neighbours is usually a harmonic picked by mistake."""
    for i in range(1, len(notes) - 1):
        prev, cur, nxt = notes[i - 1].pitch, notes[i].pitch, notes[i + 1].pitch
        if abs(prev - nxt) <= 2:
            for shift in (-12, 12):
                if abs(cur + shift - (prev + nxt) / 2) <= 2 and abs(cur - (prev + nxt) / 2) >= 10:
                    notes[i].pitch = cur + shift
                    break


def _smooth_chords(notes: list[GuitarNote], times: np.ndarray) -> list[int]:
    """Flip a lone chord among single notes (or the reverse) when both close neighbours agree."""
    sizes = [n.chord for n in notes]
    out = list(sizes)
    for i in range(1, len(notes) - 1):
        before, after = sizes[i - 1] > 1, sizes[i + 1] > 1
        if before == after != (sizes[i] > 1) and times[i + 1] - times[i - 1] < 0.8:
            out[i] = 2 if before else 1
    return out


def assign_frets(pitches: np.ndarray, sizes: list[int], times: np.ndarray) -> list[int]:
    """Lowest fret of each note/chord, following the pitch contour (Viterbi over 5 frets)."""
    n = len(pitches)
    if n == 0:
        return []
    frets = np.arange(5)
    unary = np.zeros((n, 5))
    for i in range(n):
        window = pitches[np.abs(times - times[i]) <= 4.0]
        lo, hi = np.percentile(window, 10), np.percentile(window, 90)
        if hi - lo < 5:
            mid = (hi + lo) / 2
            lo, hi = mid - 2.5, mid + 2.5
        top = 5 - sizes[i]
        target = np.clip((pitches[i] - lo) / (hi - lo) * top, 0, top)
        unary[i] = 0.4 * np.abs(frets - target) + np.where(frets > top, np.inf, 0.0)
        unary[i, 4] += 0.15  # orange is the least comfortable lane

    cost = unary[0].copy()
    back = np.zeros((n, 5), dtype=int)
    for i in range(1, n):
        dp = np.round(pitches[i]) - np.round(pitches[i - 1])
        db = frets[None, :] - frets[:, None]  # from (rows) -> to (cols)
        if abs(dp) < 0.5:
            trans = 2.0 * np.abs(db)
        else:
            # Charters compress intervals: a step moves one fret, a fifth two, an octave three.
            want = 1 if abs(dp) <= 3 else 2 if abs(dp) <= 8 else 3
            same_dir = np.sign(db) == np.sign(dp)
            trans = np.where(db == 0, 1.0, np.where(same_dir, 0.3 * np.abs(np.abs(db) - want), 2.5 + 0.2 * np.abs(db)))
        if times[i] - times[i - 1] > 1.5:  # new phrase: position matters more than continuity
            trans = trans * 0.2
        total = cost[:, None] + trans
        back[i] = np.argmin(total, axis=0)
        cost = total[back[i], frets] + unary[i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path.tolist()


def build_track(notes: list[GuitarNote], timing: Timing, name: str = "ExpertSingle") -> ch.Track:
    if not notes:
        return ch.Track(name)
    res = timing.tempo_map.resolution
    raw = timing.to_ticks([n.time for n in notes])
    ticks = quantize(raw, res, 60.0 / timing.tempo_map.bpm_at(raw))

    best: dict[int, GuitarNote] = {}
    for tick, note in zip(ticks, notes):
        if tick not in best or note.strength > best[tick].strength:
            best[int(tick)] = note
    order = sorted(best)
    kept = [best[t] for t in order]
    times = np.array([n.time for n in kept])

    sixteenth = res // 4
    lengths = []
    for i, (tick, note) in enumerate(zip(order, kept)):
        end = int(round(float(timing.to_ticks(note.end)) / sixteenth) * sixteenth)
        if i + 1 < len(order):
            end = min(end, order[i + 1] - sixteenth)  # leave a gap to re-strum
        lengths.append(end - tick if end - tick >= MIN_SUSTAIN_BEATS * res else 0)

    # Three-note chords only for held strums; quick ones are tiring and usually power chords.
    sizes = [2 if s == 3 and length < res else s for s, length in zip(_smooth_chords(kept, times), lengths)]
    frets = assign_frets(np.array([n.pitch for n in kept]), sizes, times)
    notes_out = [ch.Note(tick, tuple(range(frets[i], frets[i] + sizes[i])), lengths[i]) for i, tick in enumerate(order)]
    return ch.Track(name, notes_out)
