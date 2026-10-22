"""End to end: an audio file in, a Clone Hero song folder (notes.chart, song.ini, song.ogg) out."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, drums, guitar, phrases, sections, separation, songini, tempo
from . import chart as ch


SR = audio.SAMPLE_RATE
GUITAR_TIERS = (1.5, 2.5, 3.5, 5.0, 6.5, 8.0)  # busy-passage notes per second -> Clone Hero intensity 0..6
DRUM_TIERS = (2.0, 3.0, 4.5, 6.0, 7.5, 9.0)
MIN_TRACK_NOTES = 16

ProgressFn = Callable[[float, str], None]


def default_cache_dir() -> Path:
    if os.environ.get("CHART_MAKER_CACHE"):
        return Path(os.environ["CHART_MAKER_CACHE"])
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "chart_maker" / "stems"


@dataclass
class Options:
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    year: str | None = None
    charter: str = "chart_maker"
    cover: Path | None = None  # image to use as album art (otherwise the embedded cover, if any)
    source_name: str | None = None  # original file name, for the fallback title
    device: str = "cpu"
    guitar_source: str = "auto"
    export_stems: bool = False  # separate guitar/drums audio so misses mute the instrument
    lead_in: float = 2.5
    cache_dir: Path | None = field(default_factory=default_cache_dir)


@dataclass
class Result:
    folder: Path
    name: str
    artist: str
    duration: float
    bpm: float
    tempo_changes: int
    guitar_notes: int
    guitar_chords: int
    drum_notes: int
    diff_guitar: int
    diff_drums: int
    guitar_source: str
    beat_source: str
    sections: list[str]
    seconds: float
    warnings: list[str]
    preview: dict = field(repr=False, default_factory=dict)


def safe_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip().rstrip(". ")
    return text[:120] or "Musica"


def _metadata(source: Path, info: audio.AudioInfo, options: Options) -> dict[str, str]:
    tags = info.tags
    year_match = re.search(r"\d{4}", tags.get("date", "") or tags.get("year", ""))
    fallback = Path(options.source_name).stem if options.source_name else source.stem
    return {
        "name": options.title or tags.get("title") or fallback,
        "artist": options.artist or tags.get("artist") or tags.get("album_artist") or "Desconhecido",
        "album": options.album or tags.get("album", ""),
        "genre": options.genre or tags.get("genre") or "Rock",
        "year": options.year or (year_match.group(0) if year_match else ""),
    }


def _difficulty(track: ch.Track, timing: tempo.Timing, tiers: tuple[float, ...]) -> int:
    if len(track.notes) < 2:
        return songini.NO_PART
    times = timing.tempo_map.tick_to_time(np.array([n.tick for n in track.notes], dtype=float))
    counts = np.bincount((times // 5).astype(int))
    busy = np.percentile(counts[counts > 0], 75) / 5.0
    return int(np.searchsorted(tiers, busy))


def _preview_start_ms(mono: np.ndarray, timing: tempo.Timing, duration: float) -> int:
    """Start of the loudest 30 seconds, on a bar line."""
    hop = SR // 2
    frames = len(mono) // hop
    window = 60  # half-second frames
    if frames <= window:
        return 0
    rms = np.sqrt((mono[: frames * hop].astype(np.float64) ** 2).reshape(frames, hop).mean(axis=1))
    energy = np.convolve(rms, np.ones(window), mode="valid")
    start = float(np.argmax(energy[: max(1, len(energy) - 1)])) * 0.5 + timing.pad
    bar_times = timing.tempo_map.tick_to_time(timing.bar_ticks)
    bar = bar_times[np.argmin(np.abs(bar_times - start))]
    return int(max(0.0, min(bar, duration + timing.pad - 30)) * 1000)


def _preview(tracks: list[ch.Track], timing: tempo.Timing, section_events: list[tuple[int, str]]) -> dict:
    """Note times (seconds on the padded timeline) for the in-browser highway preview."""
    tm = timing.tempo_map

    def seconds(tick: float) -> float:
        return round(float(tm.tick_to_time(tick)), 4)

    out: dict = {}
    for track in tracks:
        key = "guitar" if track.name == "ExpertSingle" else "drums"
        out[key] = {
            "notes": [{"t": seconds(n.tick), "d": round(seconds(n.tick + n.length) - seconds(n.tick), 4) if n.length else 0,
                       "l": list(n.lanes), "f": list(n.flags)} for n in track.notes],
            "sp": [[seconds(p.tick), seconds(p.tick + p.length)] for p in track.phrases if p.kind == ch.STAR_POWER],
            "fills": [[seconds(p.tick), seconds(p.tick + p.length)] for p in track.phrases if p.kind == ch.DRUM_FILL],
        }
    out["beats"] = [round(float(t), 4) for t in tm.tick_to_time(np.arange(0, timing.bar_ticks[-1], tm.resolution))]
    out["bars"] = [round(float(t), 4) for t in tm.tick_to_time(timing.bar_ticks)]
    out["sections"] = [[seconds(tick), name] for tick, name in section_events]
    return out


def validate_chart(path: Path) -> None:
    """Re-read the written chart and check the invariants Clone Hero relies on."""
    sections_ = ch.parse_chart(path.read_text(encoding="utf-8"))
    for required in ("Song", "SyncTrack"):
        if required not in sections_:
            raise ValueError(f"[{required}] missing")
    sync = ch.track_events(sections_, "SyncTrack")
    if not any(t == 0 and k == "B" for t, k, _ in sync) or not any(t == 0 and k == "TS" for t, k, _ in sync):
        raise ValueError("tempo and time signature must start at tick 0")
    for name, allowed in (("ExpertSingle", {0, 1, 2, 3, 4}), ("ExpertDrums", {0, 1, 2, 3, 4, 32, 66, 67, 68})):
        seen: set[tuple[int, int]] = set()
        lanes_at: dict[int, set[int]] = {}
        for tick, kind, values in ch.track_events(sections_, name):
            if kind == "N":
                if values[0] not in allowed or values[1] < 0:
                    raise ValueError(f"[{name}] invalid note {values} at {tick}")
                if (tick, values[0]) in seen:
                    raise ValueError(f"[{name}] duplicate note {values[0]} at {tick}")
                seen.add((tick, values[0]))
                lanes_at.setdefault(tick, set()).add(values[0])
        if name == "ExpertDrums":
            for tick, lanes in lanes_at.items():
                for flag, lane in ((66, 2), (67, 3), (68, 4)):
                    if flag in lanes and lane not in lanes:
                        raise ValueError(f"[{name}] cymbal marker without a pad at {tick}")
                if len(lanes & {1, 2, 3, 4}) > 2:
                    raise ValueError(f"[{name}] more than two hand notes at {tick}")


def generate(
    source: Path,
    output_root: Path,
    options: Options | None = None,
    progress: ProgressFn | None = None,
) -> Result:
    options = options or Options()
    report: ProgressFn = progress or (lambda fraction, message: None)
    started = time.time()
    warnings: list[str] = []

    report(0.0, "Lendo o áudio")
    info = audio.probe(source)
    mix = audio.decode(source)
    duration = mix.shape[1] / SR
    if duration > 15 * 60:
        raise audio.AudioError("Músicas com mais de 15 minutos não são suportadas.")
    mono = mix.mean(axis=0)
    meta = _metadata(source, info, options)

    stage = "Separando os instrumentos (Demucs)"
    report(0.02, stage)
    stems = separation.separate_cached(mix, options.cache_dir, options.device, lambda f: report(0.02 + 0.63 * f, stage))

    report(0.66, "Detectando andamento e compassos")
    beat_model = tempo.load_beat_model(options.device)
    if beat_model is None:
        warnings.append("Modelo Beat This! indisponível: andamento estimado com librosa (menos preciso).")
    beats = tempo.track_beats(mono, SR, beat_model, options.device)

    report(0.72, "Transcrevendo a bateria")
    drum_mono = stems["drums"].mean(axis=0)
    hits = drums.pick_hits(drums.activations(drum_mono, SR))
    beats = tempo.align_to_onsets(beats, [h.time for h in hits if h.kind in ("kick", "snare")])
    drums.assign_tom_lanes(hits, drum_mono, SR)
    beat_period = float(np.median(np.diff(beats.times))) if len(beats.times) > 1 else 0.5
    drums.split_cymbals(hits, beat_period)

    report(0.80, "Transcrevendo a guitarra")
    guitar_audio, guitar_label = guitar.choose_source(stems, options.guitar_source)
    guitar_notes = guitar.detect_notes(guitar_audio.mean(axis=0), SR)

    report(0.88, "Montando o chart")
    firsts = [x[0].time for x in (hits, guitar_notes) if x]
    timing = tempo.build_timing(beats, duration, min(firsts) if firsts else None, options.lead_in)
    guitar_track = guitar.build_track(guitar_notes, timing)
    drum_track = drums.build_track(hits, timing)

    bar_ticks = timing.bar_ticks
    bar_times = timing.tempo_map.tick_to_time(bar_ticks) - timing.pad
    section_list = sections.detect_sections(mono, SR, bar_times, duration)
    section_events = [(int(bar_ticks[b]), name) for b, name in section_list]
    bar_seconds = float(np.median(np.diff(bar_times))) if len(bar_times) > 1 else 2.0
    fills = phrases.drum_fills(drum_track.notes, bar_ticks, [t for t, _ in section_events])
    drum_track.phrases = sorted(
        fills + phrases.star_power(drum_track.notes, bar_ticks, bar_seconds, avoid=fills), key=lambda p: p.tick
    )
    guitar_track.phrases = phrases.star_power(guitar_track.notes, bar_ticks, bar_seconds)

    # A handful of notes means the instrument is absent and they are separation artifacts.
    if len(guitar_track.notes) < MIN_TRACK_NOTES:
        guitar_track.notes = []
        warnings.append("Quase nenhuma nota de guitarra encontrada; o chart só terá bateria.")
    if len(drum_track.notes) < MIN_TRACK_NOTES:
        drum_track.notes = []
        warnings.append("Quase nenhuma batida de bateria encontrada; o chart só terá guitarra.")
    tracks = [t for t in (guitar_track, drum_track) if t.notes]
    if not tracks:
        raise RuntimeError("Não foi possível detectar notas de guitarra nem de bateria nesta música.")
    diff_guitar = _difficulty(guitar_track, timing, GUITAR_TIERS)
    diff_drums = _difficulty(drum_track, timing, DRUM_TIERS)
    preview_ms = _preview_start_ms(mono, timing, duration)

    report(0.92, "Gravando os arquivos")
    folder = output_root / safe_name(f"{meta['artist']} - {meta['name']}")
    folder.mkdir(parents=True, exist_ok=True)
    stem_files = {"drums.ogg": stems["drums"], "guitar.ogg": guitar_audio}
    if options.export_stems:
        for name, data in stem_files.items():
            audio.encode_ogg(data, folder / name, pad=timing.pad, quality=6)
        audio.encode_ogg(mix - stems["drums"] - guitar_audio, folder / "song.ogg", pad=timing.pad)
    else:
        audio.encode_ogg(mix, folder / "song.ogg", pad=timing.pad)
        for name in stem_files:  # stale stems from an earlier run would play on top of the mix
            (folder / name).unlink(missing_ok=True)

    cover = folder / "album.png"
    has_cover = bool(options.cover) and audio.convert_image(options.cover, cover)
    has_cover = has_cover or (info.has_cover and audio.extract_cover(source, cover))
    if not has_cover:
        audio.make_cover(meta["name"], meta["artist"], cover)

    chart = ch.Chart(
        metadata={**meta, "charter": options.charter, "preview_start": preview_ms / 1000.0,
                  "difficulty": max(diff_guitar, diff_drums, 0)},
        tempos=timing.tempo_map.tempos,
        time_signatures=timing.time_signatures,
        sections=section_events,
        tracks=tracks,
    )
    ch.write_chart(chart, folder / "notes.chart")
    validate_chart(folder / "notes.chart")
    songini.write_song_ini(folder / "song.ini", songini.song_ini_fields(
        **meta, charter=options.charter, song_length_ms=int((duration + timing.pad) * 1000),
        preview_start_ms=preview_ms, diff_guitar=diff_guitar, diff_drums=diff_drums,
    ))
    report(1.0, "Pronto")

    tempos = timing.tempo_map.tempos
    return Result(
        folder=folder,
        name=meta["name"],
        artist=meta["artist"],
        duration=duration,
        bpm=round(tempos[0][1] / 1000.0, 2),
        tempo_changes=len(tempos) - 1,
        guitar_notes=len(guitar_track.notes),
        guitar_chords=sum(1 for n in guitar_track.notes if len(n.lanes) > 1),
        drum_notes=len(drum_track.notes),
        diff_guitar=diff_guitar,
        diff_drums=diff_drums,
        guitar_source=guitar_label,
        beat_source=beats.source,
        sections=[name for _, name in section_list],
        seconds=round(time.time() - started, 1),
        warnings=warnings,
        preview=_preview(tracks, timing, section_events),
    )
