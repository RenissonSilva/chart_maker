from pathlib import Path

import pytest

from chart_maker import chart as ch
from chart_maker.pipeline import validate_chart


def sample_chart() -> ch.Chart:
    guitar = ch.Track("ExpertSingle", [ch.Note(768, (0, 1), 192), ch.Note(960, (2,))], [ch.Phrase(768, ch.STAR_POWER, 193)])
    drums = ch.Track(
        "ExpertDrums",
        [ch.Note(768, (ch.KICK, ch.DRUM_YELLOW), 0, (66,)), ch.Note(864, (ch.DRUM_RED,)), ch.Note(960, (ch.DOUBLE_KICK,))],
        [ch.Phrase(0, ch.DRUM_FILL, 768)],
    )
    return ch.Chart(
        metadata={"name": 'Song "Quoted"', "artist": "Ártista", "year": "2024", "charter": "test"},
        tempos=[(0, 120000), (1536, 140500)],
        time_signatures=[(0, 4), (3072, 3)],
        sections=[(0, "Intro")],
        tracks=[guitar, drums],
    )


def test_render_structure_and_order():
    text = ch.render_chart(sample_chart())
    sections = ch.parse_chart(text)
    assert list(sections) == ["Song", "SyncTrack", "Events", "ExpertSingle", "ExpertDrums"]
    song = dict(sections["Song"])
    assert song["Resolution"] == "192"
    assert song["Name"] == "\"Song 'Quoted'\""  # no escapes exist in .chart
    assert song["Year"] == '", 2024"'
    sync = ch.track_events(sections, "SyncTrack")
    assert sync[0] == (0, "TS", [4]) and sync[1] == (0, "B", [120000])
    drums = ch.track_events(sections, "ExpertDrums")
    assert drums[0] == (0, "S", [64, 768])
    assert [(t, v) for t, k, v in drums if t == 768] == [(768, [0, 0]), (768, [2, 0]), (768, [66, 0])]
    assert "\r\n" in text and not text.startswith("﻿")


def test_parse_rejects_out_of_order_events():
    text = "[Song]\n{\n  Resolution = 192\n}\n[ExpertSingle]\n{\n  10 = N 0 0\n  5 = N 1 0\n}\n"
    with pytest.raises(ValueError, match="out of order"):
        ch.track_events(ch.parse_chart(text), "ExpertSingle")


def test_validate_written_chart(tmp_path: Path):
    path = tmp_path / "notes.chart"
    ch.write_chart(sample_chart(), path)
    validate_chart(path)


def test_validate_catches_three_hands(tmp_path: Path):
    bad = sample_chart()
    bad.tracks[1].notes.append(ch.Note(1200, (ch.DRUM_RED, ch.DRUM_YELLOW, ch.DRUM_BLUE)))
    path = tmp_path / "notes.chart"
    ch.write_chart(bad, path)
    with pytest.raises(ValueError, match="two hand notes"):
        validate_chart(path)
