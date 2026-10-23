"""Local web UI: upload a song, follow the progress, download or install the chart."""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import pipeline
from ..guitar import SOURCES

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 300 * 1024 * 1024
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".opus", ".flac", ".wav", ".m4a", ".aac", ".wma", ".webm", ".mp4", ".mka", ".aiff", ".aif"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
JOB_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class Job:
    id: str
    filename: str
    created: float
    options: dict
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    stage: str = "Na fila"
    result: dict | None = None
    error: str | None = None
    installed_to: str | None = None
    folder: str | None = None
    extra: dict = field(default_factory=dict)


class JobStore:
    """Jobs live in memory and in ``<output>/.jobs/<id>/job.json`` so they survive restarts."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        for path in sorted(root.glob("*/job.json")):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, TypeError):
                continue
            if job.status in ("queued", "running"):  # interrupted by a restart
                job.status, job.error, job.stage = "error", "O servidor foi reiniciado durante o processamento.", "Interrompido"
            self.jobs[job.id] = job

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            path = self.dir(job.id) / "job.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)

    def get(self, job_id: str) -> Job:
        if not JOB_ID.match(job_id) or job_id not in self.jobs:
            raise HTTPException(404, "Tarefa não encontrada")
        return self.jobs[job_id]


def _install(folder: Path, songs_dir: Path) -> Path:
    target = songs_dir / folder.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(folder, target)
    return target


def _worker(store: JobStore, tasks: queue.Queue, output_dir: Path, songs_dir: Path | None, device: str) -> None:
    while True:
        job_id = tasks.get()
        job = store.jobs[job_id]
        job_dir = store.dir(job_id)
        job.status, job.stage = "running", "Iniciando"
        store.save(job)
        last_save = 0.0

        def progress(fraction: float, message: str, job: Job = job) -> None:
            nonlocal last_save
            job.progress, job.stage = round(fraction, 4), message
            if time.time() - last_save > 0.5 or fraction >= 1.0:
                last_save = time.time()
                store.save(job)

        opts = job.options
        cover = job_dir / opts["cover"] if opts.get("cover") else None
        options = pipeline.Options(
            title=opts.get("title") or None,
            artist=opts.get("artist") or None,
            album=opts.get("album") or None,
            genre=opts.get("genre") or None,
            year=opts.get("year") or None,
            cover=cover,
            source_name=job.filename,
            device=device,
            guitar_source=opts.get("guitar_source", "auto"),
            export_stems=bool(opts.get("stems")),
        )
        try:
            result = pipeline.generate(job_dir / opts["upload"], output_dir, options, progress)
            (job_dir / "preview.json").write_text(json.dumps(result.preview), encoding="utf-8")
            summary = asdict(result)
            summary.pop("preview", None)
            summary["folder"] = str(result.folder)
            job.result, job.folder = summary, str(result.folder)
            if opts.get("install") and songs_dir is not None:
                job.installed_to = str(_install(result.folder, songs_dir))
            job.status, job.progress, job.stage = "done", 1.0, "Pronto"
        except Exception as exc:  # noqa: BLE001 - shown to the user
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
            job.status, job.error, job.stage = "error", str(exc) or exc.__class__.__name__, "Erro"
        finally:
            (job_dir / opts["upload"]).unlink(missing_ok=True)  # the chart folder has its own audio copy
            store.save(job)
            tasks.task_done()


def create_app(output_dir: Path, songs_dir: Path | None = None, device: str = "cpu") -> FastAPI:
    output_dir = Path(output_dir).expanduser().resolve()
    songs_dir = Path(songs_dir).expanduser().resolve() if songs_dir else None
    store = JobStore(output_dir / ".jobs")
    tasks: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_worker, args=(store, tasks, output_dir, songs_dir, device), daemon=True).start()

    app = FastAPI(title="chart_maker")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config() -> dict:
        return {"output_dir": str(output_dir), "songs_dir": str(songs_dir) if songs_dir else None,
                "guitar_sources": list(SOURCES)}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        jobs = sorted(store.jobs.values(), key=lambda j: j.created, reverse=True)
        return [asdict(j) for j in jobs]

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        cover: UploadFile | None = File(None),
        title: str = Form(""),
        artist: str = Form(""),
        album: str = Form(""),
        genre: str = Form(""),
        year: str = Form(""),
        guitar_source: str = Form("auto"),
        stems: bool = Form(False),
        install: bool = Form(False),
    ) -> dict:
        name = Path(file.filename or "musica").name
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            raise HTTPException(400, f"Formato não suportado ({ext or 'sem extensão'}). Use mp3, ogg, flac, wav, m4a...")
        if guitar_source not in SOURCES:
            raise HTTPException(400, "Fonte de guitarra inválida")
        if install and songs_dir is None:
            raise HTTPException(400, "A pasta de músicas do jogo não foi configurada (CHART_MAKER_SONGS_DIR no .env ou --songs-dir).")
        job_id = uuid.uuid4().hex
        job_dir = store.dir(job_id)
        job_dir.mkdir(parents=True)
        options = {"title": title.strip(), "artist": artist.strip(), "album": album.strip(), "genre": genre.strip(),
                   "year": year.strip(), "guitar_source": guitar_source, "stems": stems, "install": install,
                   "upload": "input" + ext}
        await _save_upload(file, job_dir / options["upload"])
        if cover is not None and cover.filename:
            cover_ext = Path(cover.filename).suffix.lower()
            if cover_ext not in IMAGE_EXTENSIONS:
                shutil.rmtree(job_dir)
                raise HTTPException(400, "A capa deve ser uma imagem (png, jpg, webp).")
            options["cover"] = "cover" + cover_ext
            await _save_upload(cover, job_dir / options["cover"])
        job = Job(id=job_id, filename=name, created=time.time(), options=options)
        store.save(job)
        tasks.put(job_id)
        return {"id": job_id}

    async def _save_upload(upload: UploadFile, target: Path) -> None:
        size = 0
        with open(target, "wb") as out:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    out.close()
                    shutil.rmtree(target.parent, ignore_errors=True)
                    raise HTTPException(413, "Arquivo grande demais (máximo 300 MB).")
                out.write(chunk)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return asdict(store.get(job_id))

    def _folder(job: Job) -> Path:
        if job.status != "done" or not job.folder or not Path(job.folder).is_dir():
            raise HTTPException(409, "O chart ainda não está pronto (ou a pasta foi removida).")
        return Path(job.folder)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        job = store.get(job_id)
        folder = _folder(job)
        archive = store.dir(job_id) / f"{folder.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    zf.write(path, f"{folder.name}/{path.name}")
        return FileResponse(archive, media_type="application/zip", filename=archive.name)

    @app.post("/api/jobs/{job_id}/install")
    def install_job(job_id: str) -> dict:
        job = store.get(job_id)
        if songs_dir is None:
            raise HTTPException(400, "A pasta de músicas do jogo não foi configurada (CHART_MAKER_SONGS_DIR no .env ou --songs-dir).")
        job.installed_to = str(_install(_folder(job), songs_dir))
        store.save(job)
        return {"installed_to": job.installed_to}

    @app.get("/api/jobs/{job_id}/preview")
    def preview(job_id: str) -> JSONResponse:
        path = store.dir(store.get(job_id).id) / "preview.json"
        if not path.exists():
            raise HTTPException(404, "Pré-visualização indisponível")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/jobs/{job_id}/audio")
    def song_audio(job_id: str) -> FileResponse:
        path = _folder(store.get(job_id)) / "song.ogg"
        return FileResponse(path, media_type="audio/ogg")

    @app.get("/api/jobs/{job_id}/cover")
    def song_cover(job_id: str) -> FileResponse:
        path = _folder(store.get(job_id)) / "album.png"
        if not path.exists():
            raise HTTPException(404, "Sem capa")
        return FileResponse(path, media_type="image/png")

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job.status in ("queued", "running"):
            raise HTTPException(409, "Aguarde o processamento terminar.")
        with store.lock:
            store.jobs.pop(job.id, None)
        shutil.rmtree(store.dir(job.id), ignore_errors=True)
        return {"deleted": job.id}

    return app
