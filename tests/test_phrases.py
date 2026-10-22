import numpy as np

from chart_maker import chart as ch
from chart_maker import phrases

BAR = 768


def bars(n=60):
    return np.arange(n) * BAR


def test_star_power_covers_notes_and_repeats():
    notes = [ch.Note(t, (0,)) for t in range(BAR * 2, BAR * 58, 96)]
    sp = phrases.star_power(notes, bars(), bar_seconds=2.0, spacing_seconds=20.0)
    assert len(sp) >= 4
    ticks = np.array([n.tick for n in notes])
    for p in sp:
        inside = ticks[(ticks >= p.tick) & (ticks < p.tick + p.length)]
        assert len(inside) >= 3 and inside[0] == p.tick


def test_drum_fills_end_on_a_note_and_avoid_star_power():
    notes = []
    for b in range(2, 58):
        notes.append(ch.Note(b * BAR, (ch.KICK, ch.DRUM_GREEN), 0, (68,) if b % 8 == 0 else ()))
        notes += [ch.Note(b * BAR + k * 192, (ch.DRUM_RED,)) for k in (1, 2, 3)]
    fills = phrases.drum_fills(notes, bars(), section_ticks=[16 * BAR])
    note_ticks = {n.tick for n in notes}
    assert fills
    for f in fills:
        assert f.length == BAR and f.tick + f.length in note_ticks
    assert any(f.tick + f.length == 16 * BAR for f in fills)  # section start is preferred
    sp = phrases.star_power(notes, bars(), bar_seconds=2.0, avoid=fills)
    for p in sp:
        for f in fills:
            assert p.tick + p.length <= f.tick or p.tick > f.tick + f.length
