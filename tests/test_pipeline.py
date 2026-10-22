"""End-to-end run on a synthetic song. Slow (runs Demucs on CPU) and needs the models,
which are downloaded on first use: ``uv run pytest -m slow``."""

import numpy as np
import pytest
import soundfile as sf

from chart_maker import chart as ch
from chart_maker import pipeline

from .synth import SR, drum_loop, melody


@pytest.mark.slow
def test_generate_synthetic_song(tmp_path):
    bars, bpm = 12, 120.0
    drums = drum_loop(bpm, bars)
    beat = 60.0 / bpm
    # Demucs files this additive synth tone under "piano" (in the bass register it would be
    # "bass"), which the automatic guitar source picks up when the guitar stem is silent.
    riff = [52, 52, 55, 57, 52, 52, 59, 57]
    events = [(b * beat + 0.5 * beat * k, [riff[k] , riff[k] + 7], 0.45 * beat)
              for b in range(0, bars * 4, 4) for k in range(8)]
    guitar = melody(events, len(drums) / SR)[: len(drums)]
    mix = np.stack([0.6 * drums + 0.5 * guitar] * 2, axis=1)
    source = tmp_path / "Synthetic Song.wav"
    sf.write(source, mix, SR)

    options = pipeline.Options(artist="Teste", cache_dir=None)
    result = pipeline.generate(source, tmp_path / "out", options)

    folder = result.folder
    assert {p.name for p in folder.iterdir()} >= {"notes.chart", "song.ini", "song.ogg", "album.png"}
    pipeline.validate_chart(folder / "notes.chart")
    assert abs(result.bpm - bpm) < 1.0 or abs(result.bpm - 2 * bpm) < 2.0
    assert result.drum_notes > 50 and result.guitar_notes > 30
    sections = ch.parse_chart((folder / "notes.chart").read_text(encoding="utf-8"))
    drum_events = [(t, v[0]) for t, k, v in ch.track_events(sections, "ExpertDrums") if k == "N"]
    # ADTOF is trained on real kits and ignores this synthetic snare, but the kick (beats 1
    # and 3) and the 8th-note hi-hat must come out on the grid.
    kicks = [t for t, lane in drum_events if lane == ch.KICK]
    assert len(kicks) >= 20 and sum(t % 384 == 0 for t in kicks) >= 0.9 * len(kicks)
    assert sum(lane == ch.CYMBAL_FLAG[ch.DRUM_YELLOW] for _, lane in drum_events) > 60
    assert "name = Synthetic Song" in (folder / "song.ini").read_text(encoding="utf-8")
