"""Snap detected note positions onto a musical grid (16ths, triplets, ...)."""

from __future__ import annotations

import numpy as np

# Subdivisions of a beat and the per-note cost (in beats) of choosing them over plain 16ths.
GRIDS = (4, 3, 6, 8)
GRID_COST = {4: 0.0, 3: 0.03, 6: 0.06, 8: 0.06}
SWITCH_COST = 0.05  # discourages flipping between straight and triplet feel every beat


def quantize(ticks: np.ndarray, resolution: int, beat_seconds: np.ndarray | None = None) -> np.ndarray:
    """Return integer ticks snapped to the best grid of each beat.

    ``beat_seconds`` (per note) disables 32nd notes where a beat is shorter than
    0.45 s, since they would be unplayably fast there.
    """
    ticks = np.asarray(ticks, dtype=float)
    if ticks.size == 0:
        return ticks.astype(int)
    pos = ticks / resolution
    beat = np.floor(pos).astype(int)
    frac = pos - beat
    beats = np.unique(beat)
    index = {b: i for i, b in enumerate(beats)}
    cost = np.zeros((len(beats), len(GRIDS)))
    for g, d in enumerate(GRIDS):
        err = np.abs(frac - np.round(frac * d) / d) + GRID_COST[d]
        if d == 8 and beat_seconds is not None:
            err = np.where(np.asarray(beat_seconds) < 0.45, np.inf, err)
        np.add.at(cost[:, g], [index[b] for b in beat], err)

    # Viterbi over beats that contain notes; gaps of empty beats reset the penalty.
    choice = np.zeros(len(beats), dtype=int)
    acc = cost[0].copy()
    back = np.zeros((len(beats), len(GRIDS)), dtype=int)
    switch = SWITCH_COST * (1 - np.eye(len(GRIDS)))
    for i in range(1, len(beats)):
        trans = acc[:, None] + switch * (beats[i] - beats[i - 1] <= 2)
        back[i] = np.argmin(trans, axis=0)
        acc = trans[back[i], np.arange(len(GRIDS))] + cost[i]
    choice[-1] = int(np.argmin(acc))
    for i in range(len(beats) - 1, 0, -1):
        choice[i - 1] = back[i, choice[i]]

    d = np.array([GRIDS[choice[index[b]]] for b in beat])
    snapped = beat + np.round(frac * d) / d
    return np.round(snapped * resolution).astype(int)
