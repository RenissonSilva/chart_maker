"""song.ini writer (Clone Hero reads metadata from here before the .chart [Song] section)."""

from __future__ import annotations

from pathlib import Path

NO_PART = -1


def render_song_ini(fields: dict[str, object]) -> str:
    lines = ["[song]"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        lines.append(f"{key} = {text}")
    return "\r\n".join(lines) + "\r\n"


def write_song_ini(path: Path, fields: dict[str, object]) -> None:
    path.write_text(render_song_ini(fields), encoding="utf-8", newline="")


def song_ini_fields(
    *,
    name: str,
    artist: str,
    album: str,
    genre: str,
    year: str,
    charter: str,
    song_length_ms: int,
    preview_start_ms: int,
    diff_guitar: int,
    diff_drums: int,
) -> dict[str, object]:
    return {
        "name": name,
        "artist": artist,
        "album": album,
        "genre": genre,
        "year": year,
        "charter": charter,
        "song_length": song_length_ms,
        "preview_start_time": preview_start_ms,
        "diff_band": max(diff_guitar, diff_drums),
        "diff_guitar": diff_guitar,
        "diff_drums": diff_drums,
        "diff_drums_real": diff_drums,
        "diff_bass": NO_PART,
        "diff_rhythm": NO_PART,
        "diff_guitar_coop": NO_PART,
        "diff_keys": NO_PART,
        "diff_vocals": NO_PART,
        "pro_drums": True,
        "delay": 0,
        "loading_phrase": "Chart gerado automaticamente pelo chart_maker.",
    }
