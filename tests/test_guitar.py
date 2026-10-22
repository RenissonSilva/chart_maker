import librosa
import numpy as np
import pytest

from chart_maker import guitar

from .synth import SR, melody, tone


def spectrum_of(y):
    y22 = librosa.resample(np.concatenate([np.zeros(2205, dtype=np.float32), y]), orig_sr=SR, target_sr=guitar.SR)
    cqt = np.abs(librosa.cqt(y22, sr=guitar.SR, hop_length=guitar.HOP, fmin=librosa.midi_to_hz(guitar.FMIN_MIDI),
                             n_bins=guitar.N_BINS, bins_per_octave=12 * guitar.BINS_PER_SEMITONE, filter_scale=0.5))
    f = int(0.1 * guitar.SR / guitar.HOP)
    return cqt[:, f + 2 : f + 20].mean(axis=1)


@pytest.mark.parametrize("root", [45, 52, 57])
@pytest.mark.parametrize("drive", [0.0, 3.0])
def test_single_notes_vs_power_chords(root, drive):
    pitch, ratios = guitar.analyze_pitch(spectrum_of(tone([root], drive=drive)))
    assert abs(pitch - root) < 0.5
    assert guitar.chord_size(ratios) == 1
    _, ratios = guitar.analyze_pitch(spectrum_of(tone([root, root + 7], drive=drive)))
    assert guitar.chord_size(ratios) >= 2


def test_frets_follow_the_melodic_contour():
    pitches = np.array([40, 42, 43, 45, 47, 48, 50, 52, 52, 50, 48, 47], dtype=float)
    times = np.arange(len(pitches)) * 0.25
    frets = guitar.assign_frets(pitches, [1] * len(pitches), times)
    for i in range(1, len(pitches)):
        step = np.sign(pitches[i] - pitches[i - 1])
        assert np.sign(frets[i] - frets[i - 1]) in (step, 0)
    assert frets[7] == frets[8]  # repeated pitch, same fret
    assert len(set(frets)) >= 4  # uses the highway


def test_detect_notes_on_a_synthetic_riff():
    events = [(0.5 + 0.3 * i, [m], 0.28) for i, m in enumerate([45, 48, 50, 52, 50, 48, 45, 43])]
    events.append((3.2, [40, 47], 0.8))  # a held power chord at the end
    notes = guitar.detect_notes(melody(events, 4.5), SR)
    assert len(notes) == len(events)
    for note, (start, midis, _) in zip(notes, events):
        assert abs(note.time - start) < 0.03
        assert min(abs(note.pitch - midis[0] + 12 * k) for k in (-1, 0, 1)) < 0.6
    assert notes[-1].chord >= 2
    assert all(n.chord == 1 for n in notes[:-1])
