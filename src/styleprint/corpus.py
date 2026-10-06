"""Reference corpus: the recordings of a target speaker that style is learned from.

On-disk layout, one directory per speaker:

    corpus/<speaker_id>/
        speaker.json      who the speaker is
        manifest.jsonl    one Recording per line
        raw/              originals exactly as ingested
        audio/            normalized 16 kHz mono 16-bit WAV, ready for speech-to-text

The manifest is the source of truth. Audio directories can be rebuilt from
`raw/` (or re-downloaded from `source`) and are not meant to be versioned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE = 16_000

# What kind of speech event a recording is. Style varies a lot across these,
# so coverage is reported per genre.
GENRES = ("speech", "lecture", "interview", "podcast", "conversation", "debate", "monologue", "other")

# How prepared the speech is. Spontaneous speech carries the most stylistic
# signal (fillers, self-repairs, discourse markers); scripted speech may
# reflect a speechwriter rather than the speaker.
REGISTERS = ("scripted", "prepared", "spontaneous")


class CorpusError(Exception):
    pass


@dataclass
class Speaker:
    id: str
    name: str
    description: str = ""
    languages: list[str] = field(default_factory=list)  # empty until set or detected
    created_at: str = ""


@dataclass
class Recording:
    id: str
    title: str
    source: str  # original path or URL
    genre: str
    register: str
    raw_path: str  # relative to the speaker directory
    audio_path: str  # relative to the speaker directory
    sha256: str  # of the raw file
    duration_s: float
    multi_speaker: bool = False  # other voices present; needs diarization before analysis
    date: str | None = None  # when it was recorded (ISO date), if known
    rights: str = ""  # basis for using it, e.g. "own recording", "consent on file", "public, research use"
    notes: str = ""
    added_at: str = ""


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:60].rstrip("-") or "recording"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise CorpusError(f"`{tool}` not found on PATH")
    return exe


def ffmpeg_binary() -> str:
    return os.environ.get("FFMPEG_BINARY") or _require("ffmpeg")


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def normalize_audio(src: Path, dst: Path) -> None:
    """Convert any audio/video file to 16 kHz mono 16-bit PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-y", "-v", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CorpusError(f"ffmpeg failed on {src}: {result.stderr.strip()}")


def download(url: str, dest_dir: Path) -> tuple[Path, str]:
    """Fetch the best audio stream for a URL with yt-dlp; returns (file path, video title)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [_require("yt-dlp"), "--no-playlist", "-f", "bestaudio/best", "-o", "%(id)s.%(ext)s",
         "--no-simulate", "--print", "title", "--print", "after_move:filepath", "-P", str(dest_dir), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CorpusError(f"yt-dlp failed on {url}: {result.stderr.strip()}")
    lines = result.stdout.strip().splitlines()
    path = Path(lines[-1])
    return path, (lines[0] if len(lines) > 1 else path.stem)


class Corpus:
    def __init__(self, root: Path | str = "corpus"):
        self.root = Path(root)

    # --- speakers -----------------------------------------------------------

    def speaker_dir(self, speaker_id: str) -> Path:
        return self.root / speaker_id

    def init_speaker(self, name: str, speaker_id: str | None = None, description: str = "",
                     languages: list[str] | None = None) -> Speaker:
        speaker = Speaker(id=speaker_id or slugify(name), name=name, description=description,
                          languages=languages or [], created_at=_now())
        d = self.speaker_dir(speaker.id)
        if (d / "speaker.json").exists():
            raise CorpusError(f"speaker '{speaker.id}' already exists")
        for sub in ("raw", "audio"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        (d / "manifest.jsonl").touch()
        (d / "speaker.json").write_text(json.dumps(asdict(speaker), indent=2) + "\n")
        return speaker

    def speaker(self, speaker_id: str) -> Speaker:
        path = self.speaker_dir(speaker_id) / "speaker.json"
        if not path.exists():
            raise CorpusError(f"no speaker '{speaker_id}' in {self.root}")
        return Speaker(**json.loads(path.read_text()))

    def resolve(self, name_or_id: str) -> Speaker:
        """Find a speaker by id, display name (any case) or the slug of either."""
        wanted = {name_or_id, slugify(name_or_id), slugify(name_or_id).replace("-", "")}
        for sp in self.speakers():
            if sp.id in wanted or sp.name.lower() == name_or_id.lower() or slugify(sp.name) in wanted:
                return sp
        known = ", ".join(f"{sp.name} ({sp.id})" for sp in self.speakers()) or "none yet"
        raise CorpusError(f"no style named '{name_or_id}' (known: {known})")

    def update_speaker(self, speaker: Speaker) -> None:
        (self.speaker_dir(speaker.id) / "speaker.json").write_text(json.dumps(asdict(speaker), indent=2) + "\n")

    def speakers(self) -> list[Speaker]:
        if not self.root.exists():
            return []
        return [self.speaker(p.parent.name) for p in sorted(self.root.glob("*/speaker.json"))]

    # --- recordings ---------------------------------------------------------

    def recordings(self, speaker_id: str) -> list[Recording]:
        self.speaker(speaker_id)
        known = {f.name for f in fields(Recording)}
        lines = (self.speaker_dir(speaker_id) / "manifest.jsonl").read_text().splitlines()
        return [Recording(**{k: v for k, v in json.loads(line).items() if k in known})
                for line in lines if line.strip()]

    def _write_manifest(self, speaker_id: str, recs: list[Recording]) -> None:
        path = self.speaker_dir(speaker_id) / "manifest.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in recs))
        tmp.replace(path)

    def add(self, speaker_id: str, source: str, *, title: str | None = None, genre: str = "other",
            register: str = "spontaneous", multi_speaker: bool = False, date: str | None = None,
            rights: str = "", notes: str = "", recording_id: str | None = None) -> Recording:
        """Ingest a local file or URL: keep the original, write a normalized WAV, record metadata."""
        if genre not in GENRES:
            raise CorpusError(f"genre must be one of {', '.join(GENRES)}")
        if register not in REGISTERS:
            raise CorpusError(f"register must be one of {', '.join(REGISTERS)}")

        d = self.speaker_dir(speaker_id)
        recs = self.recordings(speaker_id)
        is_url = re.match(r"^https?://", source) is not None

        if is_url:
            for r in recs:
                if r.source == source:
                    raise CorpusError(f"already in corpus as '{r.id}'")

        with tempfile.TemporaryDirectory() as tmp:
            if is_url:
                fetched, url_title = download(source, Path(tmp))
                title = title or url_title
            else:
                fetched = Path(source).expanduser()
                if not fetched.is_file():
                    raise CorpusError(f"no such file: {source}")

            digest = _sha256(fetched)
            for r in recs:
                if r.sha256 == digest:
                    raise CorpusError(f"already in corpus as '{r.id}'")

            title = title or fetched.stem
            rid = recording_id or slugify(title)
            taken = {r.id for r in recs}
            if recording_id and rid in taken:
                raise CorpusError(f"recording id '{rid}' already exists")
            base, n = rid, 2
            while rid in taken:
                rid, n = f"{base}-{n}", n + 1

            raw = d / "raw" / f"{rid}{fetched.suffix.lower()}"
            audio = d / "audio" / f"{rid}.wav"
            normalize_audio(fetched, audio)
            try:
                shutil.copy2(fetched, raw)
                duration = wav_duration(audio)
            except Exception:
                audio.unlink(missing_ok=True)
                raw.unlink(missing_ok=True)
                raise

        rec = Recording(
            id=rid, title=title, source=source if is_url else str(fetched.resolve()),
            genre=genre, register=register, raw_path=str(raw.relative_to(d)),
            audio_path=str(audio.relative_to(d)), sha256=digest, duration_s=round(duration, 2),
            multi_speaker=multi_speaker, date=date, rights=rights, notes=notes, added_at=_now(),
        )
        self._write_manifest(speaker_id, recs + [rec])
        return rec

    def remove(self, speaker_id: str, recording_id: str) -> Recording:
        recs = self.recordings(speaker_id)
        match = [r for r in recs if r.id == recording_id]
        if not match:
            raise CorpusError(f"no recording '{recording_id}'")
        rec = match[0]
        d = self.speaker_dir(speaker_id)
        (d / rec.raw_path).unlink(missing_ok=True)
        (d / rec.audio_path).unlink(missing_ok=True)
        self._write_manifest(speaker_id, [r for r in recs if r.id != recording_id])
        return rec

    # --- reporting ----------------------------------------------------------

    def stats(self, speaker_id: str) -> dict:
        recs = self.recordings(speaker_id)
        by_genre: dict[str, float] = defaultdict(float)
        by_register: dict[str, float] = defaultdict(float)
        for r in recs:
            by_genre[r.genre] += r.duration_s
            by_register[r.register] += r.duration_s
        return {
            "recordings": len(recs),
            "total_s": round(sum(r.duration_s for r in recs), 2),
            "multi_speaker": sum(r.multi_speaker for r in recs),
            "by_genre": dict(by_genre),
            "by_register": dict(by_register),
        }

    def validate(self, speaker_id: str, check_hashes: bool = False) -> list[str]:
        """Return a list of problems; empty means the corpus is consistent."""
        d = self.speaker_dir(speaker_id)
        problems: list[str] = []
        seen_ids: set[str] = set()
        seen_hashes: dict[str, str] = {}
        for r in self.recordings(speaker_id):
            if r.id in seen_ids:
                problems.append(f"{r.id}: duplicate id")
            seen_ids.add(r.id)
            if r.sha256 in seen_hashes:
                problems.append(f"{r.id}: same content as {seen_hashes[r.sha256]}")
            seen_hashes[r.sha256] = r.id
            raw, audio = d / r.raw_path, d / r.audio_path
            if not raw.exists():
                problems.append(f"{r.id}: missing raw file {r.raw_path}")
            elif check_hashes and _sha256(raw) != r.sha256:
                problems.append(f"{r.id}: raw file hash mismatch")
            if not audio.exists():
                problems.append(f"{r.id}: missing normalized audio {r.audio_path}")
        referenced = {Path(r.raw_path).name for r in self.recordings(speaker_id)}
        for f in (d / "raw").glob("*"):
            if f.name not in referenced:
                problems.append(f"untracked raw file {f.name}")
        return problems
