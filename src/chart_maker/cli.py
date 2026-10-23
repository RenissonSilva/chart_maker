"""Command line: ``chart-maker generate musica.mp3`` or ``chart-maker serve``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from .guitar import SOURCES


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Read ``KEY=valor`` lines from ``.env`` into the environment; variables already set take precedence."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # no escape processing: Windows paths keep their backslashes
        os.environ.setdefault(key, value)


def _output_default() -> Path:
    return Path(os.environ.get("CHART_MAKER_OUTPUT", "output"))


def _songs_dir_default() -> Path | None:
    value = os.environ.get("CHART_MAKER_SONGS_DIR")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        print(f"Aviso: CHART_MAKER_SONGS_DIR não existe ({path}), usando {_output_default()}", file=sys.stderr)
        return None
    return path


def _progress(fraction: float, message: str) -> None:
    sys.stderr.write(f"\r  [{fraction * 100:5.1f}%] {message:<45}")
    sys.stderr.flush()


def _generate(args: argparse.Namespace) -> int:
    from .pipeline import Options, generate

    status = 0
    for path in args.files:
        if not path.is_file():
            print(f"Arquivo não encontrado: {path}", file=sys.stderr)
            status = 1
            continue
        options = Options(
            title=args.title, artist=args.artist, album=args.album, genre=args.genre, year=args.year,
            cover=args.cover, device=args.device, guitar_source=args.guitar_source,
            export_stems=args.stems, cache_dir=None if args.no_cache else Options().cache_dir,
        )
        print(f"Gerando chart: {path.name}", file=sys.stderr)
        try:
            result = generate(path, args.output, options, _progress)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next file
            sys.stderr.write("\n")
            print(f"Erro em {path.name}: {exc}", file=sys.stderr)
            status = 1
            continue
        sys.stderr.write("\n")
        print(f"  Pasta: {result.folder}")
        print(f"  {result.artist} - {result.name} | {result.duration:.0f}s | {result.bpm} BPM"
              f"{f' (+{result.tempo_changes} mudanças)' if result.tempo_changes else ''}")
        print(f"  Guitarra: {result.guitar_notes} notas ({result.guitar_chords} acordes, fonte: {result.guitar_source}),"
              f" dificuldade {result.diff_guitar}")
        print(f"  Bateria: {result.drum_notes} notas, dificuldade {result.diff_drums}")
        print(f"  Seções: {', '.join(result.sections)}")
        for warning in result.warnings:
            print(f"  Aviso: {warning}")
        print(f"  Tempo de processamento: {result.seconds:.0f}s")
    return status


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web.app import create_app

    app = create_app(output_dir=args.output, songs_dir=args.songs_dir, device=args.device)
    print(f"Abra http://{args.host}:{args.port} no navegador", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chart-maker", description="Gera charts de Clone Hero (guitarra e bateria no Expert).")
    sub = parser.add_subparsers(dest="command", required=True)
    songs_dir = _songs_dir_default()

    gen = sub.add_parser("generate", aliases=["gerar"], help="gera o chart de uma ou mais músicas")
    gen.add_argument("files", nargs="+", type=Path, help="arquivos de áudio (mp3, ogg, flac, wav, m4a...)")
    gen.add_argument("-o", "--output", type=Path, default=songs_dir or _output_default(),
                     help="pasta onde a pasta da música será criada (padrão: CHART_MAKER_SONGS_DIR do .env, senão ./output)")
    gen.add_argument("--title", help="nome da música (padrão: tag do arquivo)")
    gen.add_argument("--artist", help="artista (padrão: tag do arquivo)")
    gen.add_argument("--album")
    gen.add_argument("--genre")
    gen.add_argument("--year")
    gen.add_argument("--cover", type=Path, help="imagem da capa (padrão: capa embutida no arquivo)")
    gen.add_argument("--guitar-source", choices=SOURCES, default="auto",
                     help="stem usado para a guitarra (auto escolhe pelo volume)")
    gen.add_argument("--stems", action="store_true",
                     help="grava guitarra e bateria em arquivos separados (o instrumento some quando você erra)")
    gen.add_argument("--no-cache", action="store_true", help="não reutiliza nem guarda a separação de instrumentos")
    gen.add_argument("--device", default="cpu", help="cpu ou cuda")
    gen.set_defaults(func=_generate)

    srv = sub.add_parser("serve", aliases=["servidor"], help="abre a interface web para enviar músicas")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("-o", "--output", type=Path, default=_output_default(), help="onde guardar os charts gerados")
    srv.add_argument("--songs-dir", type=Path, default=songs_dir,
                     help="pasta de músicas do jogo, para instalar com um clique (padrão: CHART_MAKER_SONGS_DIR do .env)")
    srv.add_argument("--device", default="cpu")
    srv.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    _load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
