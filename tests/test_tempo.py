import numpy as np
import pytest

from chart_maker import tempo


def test_tempo_map_round_trip():
    tm = tempo.TempoMap([(0, 120000), (768, 60000)])
    assert tm.tick_to_time(192) == pytest.approx(0.5)
    assert tm.tick_to_time(768) == pytest.approx(2.0)
    assert tm.tick_to_time(960) == pytest.approx(3.0)  # one beat at 60 BPM
    ticks = np.array([0, 100, 767, 768, 2000])
    assert tm.time_to_tick(tm.tick_to_time(ticks)) == pytest.approx(ticks)


def make_beats(bpm=120.0, n=96, first=0.3, jitter=0.004, seed=0):
    rng = np.random.default_rng(seed)
    times = first + np.arange(n) * 60.0 / bpm + rng.normal(0, jitter, n)
    return tempo.Beats(times, np.arange(1, n, 4), "test")  # pickup beat before the first bar


def test_constant_tempo_snaps_to_integer_bpm():
    beats = make_beats(bpm=120.0)
    timing = tempo.build_timing(beats, duration=50.0, first_note=0.3, lead_in=2.5)
    assert timing.tempo_map.tempos == [(0, 120000)]
    assert timing.time_signatures == [(0, 4)]
    # The first detected downbeat (beat index 1) sits exactly on a bar line.
    assert timing.beat_ticks[1] % (4 * 192) == 0
    # The first note is pushed to at least the lead-in by padding the audio.
    assert 0.3 + timing.pad >= 2.5
    # Detected beats land on integer beats of the map.
    residual = timing.to_ticks(beats.times) - timing.beat_ticks
    assert np.abs(residual).max() < 192 * 0.05


def test_tempo_change_creates_segments():
    first = np.arange(40) * 0.5
    second = first[-1] + np.arange(1, 41) * 60.0 / 150.0
    beats = tempo.Beats(np.concatenate([first, second]) + 1.0, np.arange(0, 80, 4), "test")
    timing = tempo.build_timing(beats, duration=40.0, first_note=1.0)
    bpms = [bpm / 1000 for _, bpm in timing.tempo_map.tempos]
    assert bpms[0] == 120 and bpms[-1] == 150
    residual = timing.to_ticks(beats.times) - timing.beat_ticks
    assert np.abs(residual).max() < 192 * 0.1


def test_irregular_bar_emits_time_signature():
    beats = make_beats(n=40, first=0.5, jitter=0.0)
    beats.downbeats = np.array([0, 4, 8, 11, 15, 19, 23, 27, 31, 35])  # one 3-beat bar
    timing = tempo.build_timing(beats, duration=25.0, first_note=0.5)
    nums = [num for _, num in timing.time_signatures]
    assert nums == [4, 3, 4]


def test_dp_beats_recovers_missing_and_spurious_peaks():
    period = 25  # frames at 50 fps: 120 BPM
    act = np.zeros(1000)
    act[np.arange(10, 1000, period)] = 1.0
    act[[260, 261]] = 0.0  # a missing beat
    act[[480 + 7]] = 0.9  # a spurious peak between beats
    frames = tempo.dp_beats(act, np.full(len(act), float(period)))
    assert np.allclose(np.diff(frames), period, atol=1.0)


def test_downbeat_decoding_prefers_consistent_bars():
    p = np.tile([0.9, 0.05, 0.05, 0.05], 16)
    p[21] = 0.8  # a single misleading downbeat activation
    assert list(tempo.decode_downbeats(p)) == list(range(0, 64, 4))
    waltz = np.tile([0.9, 0.05, 0.05], 20)
    assert list(tempo.decode_downbeats(waltz)) == list(range(0, 60, 3))
