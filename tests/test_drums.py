import numpy as np

from chart_maker import chart as ch
from chart_maker import drums, tempo
from chart_maker.drums import DrumHit


def timing_120(n_beats=200):
    beats = tempo.Beats(np.arange(n_beats) * 0.5, np.arange(0, n_beats, 4), "test")
    return tempo.build_timing(beats, duration=n_beats * 0.5, first_note=0.0, lead_in=2.0)


def hand_lanes(*kinds, tom_lane=ch.DRUM_BLUE):
    return drums._hand_lanes({k: DrumHit(0.0, k, 1.0, tom_lane) for k in kinds})


def test_two_hands_keep_the_most_important_notes():
    lanes = hand_lanes("hihat", "snare", "crash")
    assert sorted(lanes) == [(ch.DRUM_RED, False), (ch.DRUM_GREEN, True)]


def test_tom_moves_off_a_cymbal_of_the_same_color():
    lanes = hand_lanes("hihat", "tom", tom_lane=ch.DRUM_YELLOW)
    assert (ch.DRUM_YELLOW, True) in lanes
    tom = [lane for lane, cymbal in lanes if not cymbal]
    assert tom and tom[0] != ch.DRUM_YELLOW


def test_split_cymbals_ride_pattern_vs_crash():
    ride = [DrumHit(10 + 0.25 * i, "cymbal", 1.0) for i in range(12)]
    crash = [DrumHit(1.0, "cymbal", 1.0), DrumHit(4.0, "cymbal", 1.0)]
    hits = crash + ride
    drums.split_cymbals(hits, beat_period=0.5)
    assert all(h.kind == "ride" for h in ride)
    assert all(h.kind == "crash" for h in crash)


def test_build_track_double_kick_and_cymbal_flags():
    hits = [DrumHit(5.0 + 0.125 * i, "kick", 1.0) for i in range(8)]  # 16ths at 120 BPM: 8 kicks/s
    hits += [DrumHit(5.0, "hihat", 1.0), DrumHit(5.5, "snare", 1.0)]
    track = drums.build_track(sorted(hits, key=lambda h: h.time), timing_120())
    kicks = [n for n in track.notes if ch.KICK in n.lanes or ch.DOUBLE_KICK in n.lanes]
    assert len(kicks) == 8
    assert sum(ch.DOUBLE_KICK in n.lanes for n in kicks) == 4  # every other kick is Expert+
    first = track.notes[0]
    assert ch.DRUM_YELLOW in first.lanes and ch.CYMBAL_FLAG[ch.DRUM_YELLOW] in first.flags
