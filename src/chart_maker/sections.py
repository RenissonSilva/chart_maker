"""Practice-mode sections from a structural segmentation of the mix."""

from __future__ import annotations

import numpy as np

MIN_BARS = 4  # shortest section
SECTION_SECONDS = 18.0  # typical section length, bounds how many boundaries are kept


def _bar_features(mono: np.ndarray, sr: int, bar_times: np.ndarray) -> np.ndarray:
    """Per-bar chroma (harmony), MFCC (timbre) and loudness, standardized: (features, bars)."""
    import librosa

    hop = 512
    y = librosa.resample(mono, orig_sr=sr, target_sr=22050) if sr != 22050 else mono
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=22050, hop_length=hop, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=hop)
    n = chroma.shape[1]
    frames = np.clip(librosa.time_to_frames(bar_times, sr=22050, hop_length=hop), 0, n - 1)
    ends = np.append(frames[1:], n)
    feats = np.array([
        np.concatenate([chroma[:, a:max(b, a + 1)].mean(axis=1),
                        mfcc[1:, a:max(b, a + 1)].mean(axis=1) / 50.0,
                        [np.log1p(rms[0, a:max(b, a + 1)].mean() * 100)]])
        for a, b in zip(frames, ends)
    ]).T
    return (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-9)


def novelty(feats: np.ndarray, half: int = 4) -> np.ndarray:
    """Foote novelty: a Gaussian-tapered checkerboard kernel slid along the self-similarity diagonal."""
    unit = feats / (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-9)
    ssm = unit.T @ unit
    n = ssm.shape[0]
    idx = np.arange(-half, half)
    taper = np.exp(-0.5 * ((idx + 0.5) / (half * 0.6)) ** 2)
    sign = np.where(idx < 0, -1.0, 1.0)
    kernel = np.outer(sign * taper, sign * taper)
    padded = np.pad(ssm, half, mode="edge")
    return np.array([np.sum(kernel * padded[i : i + 2 * half, i : i + 2 * half]) for i in range(n)])


def boundaries(curve: np.ndarray, max_sections: int) -> list[int]:
    """Strongest novelty peaks at least MIN_BARS apart (bar 0 always starts a section)."""
    n = len(curve)
    peaks = [i for i in range(MIN_BARS, n - MIN_BARS + 1)
             if curve[i] >= curve[max(0, i - 2) : i + 3].max() and curve[i] > 0]
    chosen: list[int] = []
    for i in sorted(peaks, key=lambda i: -curve[i]):
        if len(chosen) >= max_sections - 1:
            break
        if all(abs(i - j) >= MIN_BARS for j in chosen):
            chosen.append(i)
    return [0] + sorted(chosen)


def detect_sections(mono: np.ndarray, sr: int, bar_times: np.ndarray, duration: float) -> list[tuple[int, str]]:
    """Return ``(bar index, name)`` for each section, with boundaries on bar lines.

    Segments that sound alike share a letter ("Parte A", "Parte B", "Parte A (2)", ...).
    """
    valid = np.flatnonzero((bar_times >= 0) & (bar_times < duration - 1.0))
    if len(valid) < 2 * MIN_BARS:
        return [(int(valid[0]) if len(valid) else 0, "Intro")]
    feats = _bar_features(mono, sr, bar_times[valid])
    starts = boundaries(novelty(feats), max(2, int(round(duration / SECTION_SECONDS))))

    bounds = starts + [len(valid)]
    centroids = [feats[:, a:b].mean(axis=1) for a, b in zip(bounds[:-1], bounds[1:])]
    letters: list[int] = []
    seen: list[np.ndarray] = []
    for c in centroids:
        sims = [float(np.dot(c, o) / (np.linalg.norm(c) * np.linalg.norm(o) + 1e-9)) for o in seen]
        if sims and max(sims) > 0.8:
            letters.append(int(np.argmax(sims)))
        else:
            seen.append(c)
            letters.append(len(seen) - 1)

    names = []
    counts: dict[int, int] = {}
    alphabet: dict[int, str] = {}  # letters in order of first appearance after the intro
    for i, letter in enumerate(letters):
        if i == 0:
            names.append("Intro")
        elif i == len(letters) - 1 and bounds[i + 1] - bounds[i] <= 8:
            names.append("Final")
        else:
            label = "Parte " + alphabet.setdefault(letter, chr(ord("A") + len(alphabet) % 26))
            counts[letter] = counts.get(letter, 0) + 1
            names.append(label if counts[letter] == 1 else f"{label} ({counts[letter]})")
    return [(int(valid[b]), name) for b, name in zip(starts, names)]
