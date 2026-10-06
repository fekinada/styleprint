"""The learning pipeline as reusable steps: import → transcribe → analyze → rhetoric → style card.

Each step is idempotent and skips work already done, so `styleprint learn` can be
re-run after adding recordings and only pays for what's new.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from styleprint.corpus import Corpus, CorpusError, Speaker

Log = Callable[[str], None]

MEDIA_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf",
                    ".amr", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp"}


def expand_sources(sources: list[str], log: Log | None = None) -> list[str]:
    """Folders become the audio/video files inside them; files and URLs pass through."""
    out: list[str] = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_dir():
            found = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file())
            if not found and log:
                log(f"  ! no audio or video files in {src} (looked for {', '.join(sorted(MEDIA_EXTENSIONS))})")
            out += found
        else:
            out.append(src)
    return out


def ensure_speaker(c: Corpus, name: str, language: str | None, log: Log) -> Speaker:
    try:
        sp = c.resolve(name)
    except CorpusError:
        sp = c.init_speaker(name, languages=[language] if language else [])
        how = f"language {language}" if language else "language will be detected from the audio"
        log(f"new style '{sp.name}' ({sp.id}), {how}")
        return sp
    if language and (not sp.languages or sp.languages[0] != language):
        sp.languages = [language] + [x for x in sp.languages if x != language]
        c.update_speaker(sp)
        log(f"language set to {language}")
    return sp


def ingest(c: Corpus, sp: Speaker, sources: list[str], log: Log, **meta) -> tuple[int, int]:
    added = failed = 0
    for src in expand_sources(sources, log):
        try:
            rec = c.add(sp.id, src, **meta)
            log(f"  + {rec.id} ({rec.duration_s / 60:.1f} min)")
            added += 1
        except CorpusError as e:
            if "already in corpus" in str(e):
                log(f"  = {Path(src).name}: {e}")
            else:
                log(f"  ! {src}: {e}")
                failed += 1
    return added, failed


# Below this, Whisper's guess is too unsure to build a style on.
MIN_DETECT_CONFIDENCE = 0.6
# A recording this clearly in another language is probably a mistake.
MISMATCH_CONFIDENCE = 0.8


def transcribe(c: Corpus, sp: Speaker, log: Log, transcriber=None) -> int:
    """Transcribe new recordings, detecting the style's language first if it isn't set."""
    from styleprint.transcribe import Transcriber

    t = transcriber or Transcriber(c)
    pending = [r for r in c.recordings(sp.id) if not t.is_current(sp.id, r)]
    if not pending:
        return 0

    checked = set()
    if not sp.languages:
        ranked = t.detect(sp.id, pending[0])
        lang, conf = ranked[0]
        if conf < MIN_DETECT_CONFIDENCE:
            guesses = ", ".join(f"{l} {p:.0%}" for l, p in ranked[:3])
            raise CorpusError(f"couldn't tell the language of {pending[0].id} ({guesses}): pass --language")
        sp.languages = [lang]
        c.update_speaker(sp)
        log(f"  detected language: {lang} ({conf:.0%}, from {pending[0].id})")
        checked.add(pending[0].id)

    for r in pending:
        if r.id in checked:
            continue
        lang, conf = t.detect(sp.id, r)[0]
        if lang != sp.languages[0] and conf >= MISMATCH_CONFIDENCE:
            log(f"  ! {r.id} sounds like '{lang}' ({conf:.0%}), not '{sp.languages[0]}': "
                f"transcribing as '{sp.languages[0]}' anyway; remove it if it's the wrong recording")

    done = 0
    for r in t.run(sp.id):
        if r.status == "done":
            log(f"  + {r.recording.id}")
            done += 1
    return done


def analyze(c: Corpus, sp: Speaker, log: Log, pause_s: float = 0.5) -> dict:
    from styleprint.linguistic import analyze as run_analysis
    from styleprint.report import render
    from styleprint.segment import segment_speaker

    from styleprint.lexicons import LEXICONS

    if not sp.languages:
        raise CorpusError(f"no language for '{sp.name}' yet: transcribe first, or set --language")
    if sp.languages[0] not in LEXICONS:
        raise CorpusError(f"'{sp.name}' speaks '{sp.languages[0]}', but analysis supports only: "
                          f"{', '.join(LEXICONS)}. Transcripts are kept; the style can't be built yet.")
    utts = segment_speaker(c, sp.id, pause_s=pause_s)
    if not utts:
        raise CorpusError(f"no transcripts for '{sp.name}' yet: add recordings first")
    rec_ids = {u.recording_id for u in utts}
    duration = sum(r.duration_s for r in c.recordings(sp.id) if r.id in rec_ids)
    profile = run_analysis(utts, sp.languages[0], speaker=sp.name, duration_s=duration)
    out = c.speaker_dir(sp.id) / "profile"
    out.mkdir(exist_ok=True)
    (out / "linguistic.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1) + "\n")
    (out / "linguistic.md").write_text(render(profile))
    log(f"  {len(utts)} utterances, {profile['corpus']['words']:,} words → profile/linguistic.md")
    return profile


def rhetoric(c: Corpus, sp: Speaker, log: Log, **llm_overrides) -> dict:
    from styleprint.llm import LLM, LLMConfig
    from styleprint.rhetoric import RhetoricAnnotator, aggregate, chunk_utterances, render
    from styleprint.segment import load_utterances

    d = c.speaker_dir(sp.id)
    if not (d / "utterances.jsonl").exists():
        raise CorpusError("no utterances yet: the analysis step must run first")
    utts = load_utterances(d / "utterances.jsonl")
    cfg = LLMConfig.load(**llm_overrides)
    llm = LLM(cfg)
    model = llm.model
    n = len(chunk_utterances(utts))
    log(f"  {model} (thinking {'on' if cfg.thinking else 'off'}), {n} passages")

    annotator = RhetoricAnnotator(llm.json, d / "profile" / "rhetoric_cache", model, cfg.thinking)
    results = []
    for i, r in enumerate(annotator.run(utts, sp.languages[0]), 1):
        results.append(r)
        if not r.cached:
            log(f"  [{i}/{n}] {r.chunk.recording_id} #{r.chunk.number}: {len(r.items)} found ({r.seconds:.0f}s)")
    result = aggregate(results, sum(len(u.text.split()) for u in utts), model)
    (d / "profile" / "rhetoric.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    (d / "profile" / "rhetoric.md").write_text(render(result, sp.name))
    cached = sum(r.cached for r in results)
    log(f"  {result['annotations']} annotations ({cached}/{n} passages cached) → profile/rhetoric.md")
    return result


def load_profile(c: Corpus, sp: Speaker) -> tuple[dict, dict | None]:
    d = c.speaker_dir(sp.id) / "profile"
    if not (d / "linguistic.json").exists():
        raise CorpusError(f"'{sp.name}' has no style yet: run `styleprint learn \"{sp.name}\" <files>`")
    rhet = json.loads((d / "rhetoric.json").read_text()) if (d / "rhetoric.json").exists() else None
    return json.loads((d / "linguistic.json").read_text()), rhet


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stylecard(c: Corpus, sp: Speaker, log: Log, force: bool = False) -> str:
    """Build the card, unless it has hand edits (then keep it, unless forced).

    The hash of the last generated card is kept next to it: if the file still matches,
    nobody edited it and it's safe to regenerate from the updated profile.
    """
    from styleprint.stylecard import build

    path = c.speaker_dir(sp.id) / "profile" / "stylecard.md"
    stamp = path.with_suffix(".md.sha256")
    ling, rhet = load_profile(c, sp)
    card = build(ling, rhet, sp.name)
    if path.exists() and not force:
        current = path.read_text()
        # No stamp (card predates stamping): unedited only if it matches what we'd build now.
        expected = stamp.read_text().strip() if stamp.exists() else _sha(card)
        if _sha(current) != expected:
            log("  style card has hand edits: kept as-is (use --rebuild-card to regenerate it)")
            return current
    path.write_text(card)
    stamp.write_text(_sha(card) + "\n")
    log("  → profile/stylecard.md")
    return card


def status(c: Corpus, sp: Speaker) -> dict:
    d = c.speaker_dir(sp.id)
    recs = c.recordings(sp.id)
    transcribed = sum((d / "transcripts" / f"{r.id}.json").exists() for r in recs)
    profile = d / "profile" / "linguistic.json"
    words = json.loads(profile.read_text())["corpus"]["words"] if profile.exists() else 0
    ready = profile.exists() and (d / "profile" / "stylecard.md").exists()
    # Learned before the latest recording was added?
    stale = ready and recs and transcribed < len(recs)
    return {
        "recordings": len(recs),
        "minutes": round(sum(r.duration_s for r in recs) / 60, 1),
        "transcribed": transcribed,
        "words": words,
        "rhetoric": (d / "profile" / "rhetoric.json").exists(),
        "ready": ready,
        "stale": bool(stale),
        "generations": len(list((d / "generations").glob("*.txt"))) if (d / "generations").exists() else 0,
    }
