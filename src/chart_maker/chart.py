"""Clone Hero ``.chart`` data model, writer and a small parser used for validation.

Format reference: https://github.com/TheNathannator/GuitarGame_ChartFormats
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

RESOLUTION = 192  # ticks per quarter note (Moonscraper's default)

# 5-fret guitar note numbers
GREEN, RED, YELLOW, BLUE, ORANGE = range(5)
FORCE_FLAG, TAP_FLAG, OPEN_NOTE = 5, 6, 7

# Drum note numbers (4-lane pro)
KICK, DRUM_RED, DRUM_YELLOW, DRUM_BLUE, DRUM_GREEN = range(5)
DOUBLE_KICK = 32
CYMBAL_FLAG = {DRUM_YELLOW: 66, DRUM_BLUE: 67, DRUM_GREEN: 68}

# Special phrases
STAR_POWER = 2
DRUM_FILL = 64


@dataclass(frozen=True)
class Note:
    """All note events that start on one tick: lane numbers plus per-tick modifiers."""

    tick: int
    lanes: tuple[int, ...]
    length: int = 0
    flags: tuple[int, ...] = ()


@dataclass(frozen=True)
class Phrase:
    tick: int
    kind: int
    length: int


@dataclass
class Track:
    name: str  # e.g. "ExpertSingle", "ExpertDrums"
    notes: list[Note] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)


@dataclass
class Chart:
    metadata: dict[str, str | int | float]
    tempos: list[tuple[int, int]]  # (tick, BPM * 1000)
    time_signatures: list[tuple[int, int]]  # (tick, numerator); denominator is always 4
    sections: list[tuple[int, str]] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    resolution: int = RESOLUTION


def _quote(value: object) -> str:
    # .chart has no escape characters, so quotes and line breaks cannot survive.
    text = str(value).replace('"', "'").replace("\r", " ").replace("\n", " ")
    return f'"{text}"'


def render_chart(chart: Chart) -> str:
    meta = chart.metadata
    lines = ["[Song]", "{"]
    song_fields = [
        ("Name", _quote(meta.get("name", ""))),
        ("Artist", _quote(meta.get("artist", ""))),
        ("Charter", _quote(meta.get("charter", ""))),
        ("Album", _quote(meta.get("album", ""))),
        ("Year", _quote(f", {meta['year']}" if meta.get("year") else "")),
        ("Offset", "0"),
        ("Resolution", str(chart.resolution)),
        ("Player2", "bass"),
        ("Difficulty", str(meta.get("difficulty", 0))),
        ("PreviewStart", f"{float(meta.get('preview_start', 0.0)):.3f}"),
        ("PreviewEnd", "0"),
        ("Genre", _quote(meta.get("genre", ""))),
        ("MediaType", _quote("cd")),
        ("MusicStream", _quote(meta.get("music_stream", "song.ogg"))),
    ]
    lines += [f"  {key} = {value}" for key, value in song_fields]
    lines.append("}")

    sync: list[tuple[int, int, str]] = []
    sync += [(tick, 0, f"TS {num}") for tick, num in chart.time_signatures]
    sync += [(tick, 1, f"B {bpm}") for tick, bpm in chart.tempos]
    lines += ["[SyncTrack]", "{"]
    lines += [f"  {tick} = {text}" for tick, _, text in sorted(sync)]
    lines.append("}")

    lines += ["[Events]", "{"]
    lines += [f"  {tick} = E {_quote('section ' + name)}" for tick, name in sorted(chart.sections)]
    lines.append("}")

    for track in chart.tracks:
        events: list[tuple[int, int, int, str]] = []
        for note in track.notes:
            for lane in note.lanes:
                events.append((note.tick, 0, lane, f"N {lane} {note.length}"))
            for flag in note.flags:
                events.append((note.tick, 0, flag, f"N {flag} 0"))
        for phrase in track.phrases:
            events.append((phrase.tick, 1, phrase.kind, f"S {phrase.kind} {phrase.length}"))
        lines += [f"[{track.name}]", "{"]
        lines += [f"  {tick} = {text}" for tick, _, _, text in sorted(events)]
        lines.append("}")
    return "\r\n".join(lines) + "\r\n"


def write_chart(chart: Chart, path: Path) -> None:
    # UTF-8 without BOM: a BOM in front of "[Song]" breaks naive parsers.
    path.write_text(render_chart(chart), encoding="utf-8", newline="")


_LINE = re.compile(r"^\s*(\S+)\s*=\s*(.*?)\s*$")


def parse_chart(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a .chart into ``{section: [(key, value), ...]}`` without interpreting values."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    name = None
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if name in sections:
                raise ValueError(f"duplicate section [{name}]")
            continue
        if line == "{":
            if name is None:
                raise ValueError("'{' without a section header")
            current = sections.setdefault(name, [])
            continue
        if line == "}":
            current, name = None, None
            continue
        match = _LINE.match(line)
        if current is None or match is None:
            raise ValueError(f"unexpected line: {raw!r}")
        current.append((match.group(1), match.group(2)))
    return sections


def track_events(sections: dict[str, list[tuple[str, str]]], name: str) -> list[tuple[int, str, list[int]]]:
    """Return ``(tick, type, [values])`` for a track/sync section, validating tick order."""
    out = []
    last = -1
    for key, value in sections.get(name, []):
        tick = int(key)
        if tick < last:
            raise ValueError(f"[{name}] events out of order at tick {tick}")
        last = tick
        kind, *rest = value.split()
        out.append((tick, kind, [int(v) for v in rest] if kind in ("N", "S", "B", "TS") else []))
    return out
