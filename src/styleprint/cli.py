from __future__ import annotations

import argparse
import sys
from datetime import datetime

from styleprint.corpus import GENRES, REGISTERS, Corpus, CorpusError
from styleprint.llm import LLMError
from styleprint.transcribe import DEFAULT_MODEL, Transcriber


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def cmd_init(c: Corpus, a: argparse.Namespace) -> None:
    sp = c.init_speaker(a.name, a.id, a.description, a.languages)
    print(f"created speaker '{sp.id}' at {c.speaker_dir(sp.id)}")


def cmd_add(c: Corpus, a: argparse.Namespace) -> int:
    failed = 0
    for src in a.sources:
        try:
            rec = c.add(a.speaker, src, title=a.title if len(a.sources) == 1 else None, genre=a.genre,
                        register=a.register, multi_speaker=a.multi_speaker, date=a.date,
                        rights=a.rights, notes=a.notes)
            print(f"+ {rec.id}  {_hms(rec.duration_s)}  {rec.genre}/{rec.register}")
        except (CorpusError, LLMError) as e:
            print(f"! {src}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def cmd_rm(c: Corpus, a: argparse.Namespace) -> None:
    rec = c.remove(a.speaker, a.recording)
    print(f"- {rec.id}")


def cmd_list(c: Corpus, a: argparse.Namespace) -> None:
    if a.speaker is None:
        for sp in c.speakers():
            st = c.stats(sp.id)
            print(f"{sp.id:24} {sp.name:30} {st['recordings']:4} recs  {_hms(st['total_s'])}")
        return
    for r in c.recordings(a.speaker):
        flag = " [multi]" if r.multi_speaker else ""
        print(f"{r.id:40} {_hms(r.duration_s)}  {r.genre:12} {r.register:11} {r.date or '':10}{flag}")


def cmd_stats(c: Corpus, a: argparse.Namespace) -> None:
    st = c.stats(a.speaker)
    print(f"{st['recordings']} recordings, {_hms(st['total_s'])} total, "
          f"{st['multi_speaker']} need diarization")
    for label, key in (("genre", "by_genre"), ("register", "by_register")):
        print(f"\nby {label}:")
        for k, v in sorted(st[key].items(), key=lambda kv: -kv[1]):
            share = v / st["total_s"] if st["total_s"] else 0
            print(f"  {k:14} {_hms(v)}  {share:5.0%}")


def cmd_validate(c: Corpus, a: argparse.Namespace) -> int:
    problems = c.validate(a.speaker, check_hashes=a.hashes)
    for p in problems:
        print(p)
    if not problems:
        print("ok")
    return 1 if problems else 0


def cmd_transcribe(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl

    sp = c.resolve(a.speaker)
    t = Transcriber(c, model=a.model)
    if not sp.languages:  # detect the language, then transcribe everything new
        pl.transcribe(c, sp, _log, transcriber=t)
        return
    for r in t.run(sp.id, a.recordings or None, force=a.force, prompt=a.prompt):
        mark = "=" if r.status == "skipped" else "+"
        print(f"{mark} {r.recording.id}  {_hms(r.recording.duration_s)}  {r.status}", flush=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _llm_overrides(a: argparse.Namespace) -> dict:
    return {"url": getattr(a, "url", None), "model": getattr(a, "model", None), "thinking": getattr(a, "thinking", None)}


def cmd_learn(c: Corpus, a: argparse.Namespace) -> int:
    from styleprint import pipeline as pl

    sp = pl.ensure_speaker(c, a.name, a.language, _log)
    failed = 0
    if a.sources:
        _log("1/5 importing")
        _, failed = pl.ingest(c, sp, a.sources, _log, genre=a.genre, register=a.register,
                              multi_speaker=a.multi_speaker)
    if not c.recordings(sp.id):
        raise CorpusError(f"'{sp.name}' has no recordings: pass audio/video files or a folder")
    _log("2/5 transcribing")
    if not pl.transcribe(c, sp, _log):
        _log("  up to date")
    _log("3/5 analyzing")
    pl.analyze(c, sp, _log)
    _log("4/5 rhetoric")
    if a.no_rhetoric:
        _log("  skipped (--no-rhetoric)")
    else:
        try:
            pl.rhetoric(c, sp, _log, **_llm_overrides(a))
        except LLMError as e:
            _log(f"  skipped, LLM unavailable: {e}")
            _log("  (the style works without it; re-run `learn` later to add rhetoric)")
    _log("5/5 style card")
    pl.stylecard(c, sp, _log, force=a.rebuild_card)
    st = pl.status(c, sp)
    _log(f"\nready: '{sp.name}' learned from {st['recordings']} recordings ({st['minutes']} min, {st['words']:,} words)")
    _log(f'try:   styleprint generate "{sp.name}" "your topic"')
    return 1 if failed else 0


def cmd_styles(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl

    speakers = c.speakers()
    if not speakers:
        print('no styles yet: styleprint learn "Name" <audio files or folder>')
        return
    rows = []
    for sp in speakers:
        st = pl.status(c, sp)
        state = ("needs learn" if not st["ready"] else "new audio, re-run learn" if st["stale"]
                 else "ready" if st["rhetoric"] else "ready (no rhetoric)")
        rows.append((sp.name, sp.id, sp.languages[0] if sp.languages else "?", f"{st['recordings']}", f"{st['minutes']}",
                     f"{st['words']:,}", str(st["generations"]), state))
    head = ("NAME", "ID", "LANG", "RECS", "MIN", "WORDS", "GENERATED", "STATUS")
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip())


def cmd_analyze(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl

    pl.analyze(c, c.resolve(a.speaker), _log, pause_s=a.pause)


def cmd_rhetoric(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl

    pl.rhetoric(c, c.resolve(a.speaker), _log, **_llm_overrides(a))


def cmd_stylecard(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl

    pl.stylecard(c, c.resolve(a.speaker), _log, force=a.force)


def _print_issues(issues) -> None:
    if not issues:
        print("  no style issues found")
    for i in issues:
        print(f"  - {i.metric}: expected {i.expected}, got {i.got}")


def cmd_generate(c: Corpus, a: argparse.Namespace) -> None:
    from styleprint import pipeline as pl
    from styleprint.corpus import slugify
    from styleprint.generate import generate, select_passages, to_json
    from styleprint.llm import LLM, LLMConfig
    from styleprint.segment import load_utterances

    sp = c.resolve(a.speaker)
    d = c.speaker_dir(sp.id)
    target, rhet = pl.load_profile(c, sp)
    card_path = d / "profile" / "stylecard.md"
    card = card_path.read_text() if card_path.exists() else pl.stylecard(c, sp, _log)
    passages = select_passages(load_utterances(d / "utterances.jsonl"), rhet)

    llm = LLM(LLMConfig.load(**_llm_overrides(a)))
    print(f"{sp.name} · {llm.model}: draft + up to {a.revisions} revision(s), ~{a.words} words", flush=True)

    def show(i, v):
        print(f"\n[{'draft' if i == 0 else f'revision {i}'}] style issues: {len(v.issues)} (score {v.score})", flush=True)
        _print_issues(v.issues)

    gen = generate(lambda m, t: llm.chat(m, t), llm.model, sp.name, sp.languages[0], card, passages, target,
                   a.instruction, words=a.words, revisions=a.revisions, temperature=a.temperature,
                   on_version=show)
    out = d / "generations"
    out.mkdir(exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(a.instruction)[:40]}"
    (out / f"{stem}.json").write_text(to_json(gen))
    (out / f"{stem}.txt").write_text(gen.text + "\n")
    label = "draft" if gen.best == 0 else f"revision {gen.best}"
    print(f"\n=== best: {label} ===\n\n{gen.text}\n\nwrote {out / stem}.txt (+ .json with prompt and all versions)")


def cmd_compare(c: Corpus, a: argparse.Namespace) -> int:
    from pathlib import Path

    from styleprint import pipeline as pl
    from styleprint.generate import report, score

    sp = c.resolve(a.speaker)
    target, _ = pl.load_profile(c, sp)
    draft, issues = report(target, Path(a.file).read_text(), sp.languages[0])
    print(f"{draft['corpus']['words']} words, {draft['corpus']['utterances']} utterances; "
          f"style issues: {len(issues)} (score {score(issues)})")
    _print_issues(issues)
    for i in issues:
        print(f"    → {i.feedback}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="styleprint",
        description="Learn a speaker's style from recordings, then write new text in it.",
        epilog='Typical use:  styleprint learn "Name" recordings/   then   styleprint generate "Name" "topic"',
    )
    p.add_argument("--root", default="corpus", help="corpus directory (default: ./corpus)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_flags(s: argparse.ArgumentParser) -> None:
        s.add_argument("--url", help="OpenAI-compatible server (default: styleprint.toml [llm] url)")
        s.add_argument("--model", help="model id (default: whatever the server serves)")
        s.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None,
                       help="let Qwen3-style models reason before answering (slower)")

    # --- main commands
    s = sub.add_parser("learn", help="learn a style from recordings (create, import, transcribe, analyze)")
    s.add_argument("name", help='style name, e.g. "Philippe Etchebest" (created if new)')
    s.add_argument("sources", nargs="*", help="audio/video files, folders or URLs (none: just re-learn)")
    s.add_argument("--language", help="language code of the recordings, e.g. fr or en (default: detected)")
    s.add_argument("--genre", choices=GENRES, default="other")
    s.add_argument("--register", choices=REGISTERS, default="spontaneous")
    s.add_argument("--multi-speaker", action="store_true", help="other voices are present")
    s.add_argument("--no-rhetoric", action="store_true", help="skip the LLM rhetoric step")
    s.add_argument("--rebuild-card", action="store_true", help="overwrite a hand-edited style card")
    llm_flags(s)
    s.set_defaults(func=cmd_learn)

    s = sub.add_parser("generate", help="write a spoken script in a learned style")
    s.add_argument("speaker", help="style name or id (see `styleprint styles`)")
    s.add_argument("instruction", help="topic or instruction, e.g. 'explique comment bien dormir'")
    s.add_argument("--words", type=int, default=250)
    s.add_argument("--revisions", type=int, default=1, help="critic-guided rewrites (0 to skip)")
    s.add_argument("--temperature", type=float, default=0.8)
    llm_flags(s)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("styles", help="list learned styles")
    s.set_defaults(func=cmd_styles)

    s = sub.add_parser("compare", help="score a text file against a style")
    s.add_argument("speaker")
    s.add_argument("file")
    s.set_defaults(func=cmd_compare)

    # --- individual steps (what `learn` runs), for debugging and fine control
    corpus = sub.add_parser("corpus", help="manage the reference corpus").add_subparsers(dest="action", required=True)

    s = corpus.add_parser("init", help="register a target speaker")
    s.add_argument("name")
    s.add_argument("--id", help="speaker id (default: slug of name)")
    s.add_argument("--description", default="")
    s.add_argument("--languages", nargs="+", default=None, help="default: detected on transcription")
    s.set_defaults(func=cmd_init)

    s = corpus.add_parser("add", help="ingest audio/video files or URLs")
    s.add_argument("speaker")
    s.add_argument("sources", nargs="+", help="local paths or http(s) URLs (URLs need yt-dlp)")
    s.add_argument("--title", help="only used when adding a single source")
    s.add_argument("--genre", choices=GENRES, default="other")
    s.add_argument("--register", choices=REGISTERS, default="spontaneous")
    s.add_argument("--multi-speaker", action="store_true", help="other voices are present")
    s.add_argument("--date", help="recording date, YYYY-MM-DD")
    s.add_argument("--rights", default="", help="basis for use, e.g. 'own recording'")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_add)

    s = corpus.add_parser("rm", help="remove a recording")
    s.add_argument("speaker")
    s.add_argument("recording")
    s.set_defaults(func=cmd_rm)

    s = corpus.add_parser("list", help="list speakers, or a speaker's recordings")
    s.add_argument("speaker", nargs="?")
    s.set_defaults(func=cmd_list)

    s = corpus.add_parser("stats", help="duration coverage by genre and register")
    s.add_argument("speaker")
    s.set_defaults(func=cmd_stats)

    s = corpus.add_parser("validate", help="check manifest against files on disk")
    s.add_argument("speaker")
    s.add_argument("--hashes", action="store_true", help="re-hash raw files")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("transcribe", help="speech-to-text with word timestamps")
    s.add_argument("speaker")
    s.add_argument("recordings", nargs="*", help="recording ids (default: all)")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--prompt", help="override the default verbatim prompt for the speaker's language")
    s.add_argument("--force", action="store_true", help="redo transcripts that are already current")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("analyze", help="segment transcripts and build the linguistic style profile")
    s.add_argument("speaker")
    s.add_argument("--pause", type=float, default=0.5, help="pause (s) that ends an utterance")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("rhetoric", help="annotate rhetorical devices with an LLM (resumable)")
    s.add_argument("speaker")
    llm_flags(s)
    s.set_defaults(func=cmd_rhetoric)

    s = sub.add_parser("stylecard", help="rebuild the style card used for generation")
    s.add_argument("speaker")
    s.add_argument("--force", action="store_true", help="overwrite a hand-edited card")
    s.set_defaults(func=cmd_stylecard)

    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(Corpus(a.root), a) or 0
    except (CorpusError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
