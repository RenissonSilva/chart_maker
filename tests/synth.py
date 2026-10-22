"""Small synthetic signals for tests: guitar-like tones and a rock drum loop."""

import numpy as np

SR = 44100


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(midis, dur=0.4, sr=SR, drive=0.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for m in midis:
        f = midi_hz(m)
        for h in range(1, 30):
            if f * h > sr / 2.2:
                break
            y += np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28)) / h
    y *= np.exp(-t * 2.0) * np.minimum(1.0, t / 0.003)
    if drive:
        y = np.tanh(drive * y / np.abs(y).max())
    return (y / np.abs(y).max()).astype(np.float32)


def melody(events, total, sr=SR) -> np.ndarray:
    """``events``: (start seconds, [midi notes], duration)."""
    out = np.zeros(int(total * sr), dtype=np.float32)
    for i, (start, notes, dur) in enumerate(events):
        y = tone(notes, dur, sr, drive=3.0, seed=i)
        s = int(start * sr)
        out[s : s + y.size] += 0.5 * y[: max(0, out.size - s)]
    return out


def drum_loop(bpm=120.0, bars=8, sr=SR) -> np.ndarray:
    beat = 60.0 / bpm
    n = int((bars * 4 * beat + 1) * sr)
    y = np.zeros(n)
    rng = np.random.default_rng(0)
    t = np.arange(int(0.4 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)) * t) * np.exp(-t * 8)
    snare = (rng.standard_normal(t.size) * 0.6 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 18)
    hat = np.diff(rng.standard_normal(t.size + 1)) * np.exp(-t * 60) * 0.3
    for b in range(bars * 4):
        s = int(b * beat * sr)
        y[s : s + kick.size] += kick if b % 2 == 0 else snare
        for e in range(2):
            s2 = int((b + e / 2) * beat * sr)
            y[s2 : s2 + hat.size] += hat
    return (y / np.abs(y).max() * 0.8).astype(np.float32)
