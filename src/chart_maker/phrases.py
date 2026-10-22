"""Star Power phrases and drum activation fills."""

from __future__ import annotations

import numpy as np

from . import chart as ch


def _bar_index(bar_ticks: np.ndarray, tick: int) -> int:
    return int(np.searchsorted(bar_ticks, tick, side="right") - 1)


def star_power(
    notes: list[ch.Note],
    bar_ticks: np.ndarray,
    bar_seconds: float,
    spacing_seconds: float = 24.0,
    avoid: list[ch.Phrase] | None = None,
) -> list[ch.Phrase]:
    """One or two bars of Star Power roughly every ``spacing_seconds``, on the densest nearby bar."""
    if not notes:
        return []
    note_ticks = np.array([n.tick for n in notes])
    counts = np.bincount([max(0, _bar_index(bar_ticks, t)) for t in note_ticks], minlength=len(bar_ticks))
    blocked = np.zeros(len(bar_ticks), dtype=bool)
    for phrase in avoid or []:
        lo, hi = _bar_index(bar_ticks, phrase.tick), _bar_index(bar_ticks, phrase.tick + phrase.length)
        blocked[max(0, lo - 1) : hi + 2] = True

    step = max(4, int(round(spacing_seconds / bar_seconds)))
    first_bar = _bar_index(bar_ticks, int(note_ticks[0]))
    phrases = []
    target = first_bar + step // 2
    while target < len(bar_ticks):
        window = [b for b in range(max(first_bar, target - step // 3), min(len(bar_ticks) - 1, target + step // 3 + 1))
                  if counts[b] >= 3 and not blocked[b]]
        if window:
            bar = max(window, key=lambda b: (counts[b], -abs(b - target)))
            length = 2 if counts[bar] < 6 and bar + 1 < len(bar_ticks) - 1 and not blocked[bar + 1] else 1
            inside = note_ticks[(note_ticks >= bar_ticks[bar]) & (note_ticks < bar_ticks[min(bar + length, len(bar_ticks) - 1)])]
            if len(inside):
                phrases.append(ch.Phrase(int(inside[0]), ch.STAR_POWER, int(inside[-1] - inside[0]) + 1))
                blocked[bar : bar + length + 1] = True
            target = bar + step
        else:
            target += step // 2
    return phrases


def drum_fills(
    notes: list[ch.Note],
    bar_ticks: np.ndarray,
    section_ticks: list[int],
    min_gap_bars: int = 8,
) -> list[ch.Phrase]:
    """Activation fills: the bar before a strong downbeat, preferring crashes and section starts.

    The fill phrase includes its last tick, so it ends exactly on the activation note.
    """
    by_tick = {n.tick: n for n in notes}
    if not notes:
        return []
    first = _bar_index(bar_ticks, notes[0].tick)
    crash_flag = ch.CYMBAL_FLAG[ch.DRUM_GREEN]
    note_ticks = np.array([n.tick for n in notes])
    candidates = []
    for b in range(first + 2, len(bar_ticks)):
        note = by_tick.get(int(bar_ticks[b]))
        if note is None:
            continue
        played = np.count_nonzero((note_ticks >= bar_ticks[b - 1]) & (note_ticks < bar_ticks[b]))
        if played < 2:  # a fill needs something being played before the activation note
            continue
        score = (
            2.0 * (int(bar_ticks[b]) in section_ticks)
            + 1.0 * (crash_flag in note.flags)
            + 0.5 * any(f in note.flags for f in ch.CYMBAL_FLAG.values())
            + 0.25 * (ch.KICK in note.lanes)
        )
        candidates.append((b, score))

    chosen: list[int] = []
    for b, _score in sorted(candidates, key=lambda c: (-c[1], c[0])):
        if all(abs(b - other) >= min_gap_bars for other in chosen):
            chosen.append(b)
    return [ch.Phrase(int(bar_ticks[b - 1]), ch.DRUM_FILL, int(bar_ticks[b] - bar_ticks[b - 1])) for b in sorted(chosen)]
