"""Rhetoric: LLM annotation of rhetorical devices, with verbatim evidence.

Rhetorical devices (analogies, anecdotes, reframing...) need judgment, so a model
annotates the transcript in chunks. Every annotation must quote the transcript
verbatim; quotes that aren't found in the passage are discarded, which filters
out invented evidence. Results are cached per chunk, so a slow model can be
interrupted and resumed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from styleprint.segment import Utterance

PROMPT_VERSION = 1
CHUNK_WORDS = 350
CONTEXT_UTTERANCES = 3

DEVICES = {
    "rhetorical_question": "a question that does not seek an answer, or that the speaker answers himself "
                           "(\"Why does this matter? Because…\")",
    "contrast": "an explicit opposition between two things, options or situations",
    "analogy": "a comparison, metaphor or image borrowed from another domain",
    "anecdote": "a personal story or a narrated past event",
    "emphatic_repetition": "a word or phrase deliberately repeated for emphasis",
    "list": "an enumeration of three or more items, or a three-part rhythm",
    "reframing": "correcting or redefining what the audience expects (\"not X, but Y\", \"the real X\")",
    "provocative_opening": "a hook meant to grab attention at the start of the video or of a section",
    "counterargument": "anticipating or answering an objection the audience might have",
    "humor": "a joke, teasing or self-deprecation",
    "authority": "invoking their own experience, expertise or status to support a point",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": 15,  # without bounds, constrained decoding can loop on new items
            "items": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "enum": list(DEVICES)},
                    "utterances": {"type": "array", "items": {"type": "integer"}, "maxItems": 8},
                    "quote": {"type": "string", "maxLength": 300},
                    "why": {"type": "string", "maxLength": 250},
                },
                "required": ["device", "utterances", "quote", "why"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = """You are annotating the rhetorical style of a speaker from a transcript of spoken {language_name}.
The transcript comes from speech-to-text, so punctuation is unreliable and utterances may be fragments.

Find clear instances of these devices:
{devices}

Rules:
- Only annotate the numbered lines under PASSAGE. Lines under CONTEXT are there for understanding only.
- "quote" must be copied EXACTLY from the passage (same words, same spelling), and be as short as possible
  while still showing the device.
- "utterances" lists the line numbers the instance spans.
- "why": one short sentence, in English.
- Skip anything doubtful. Annotating nothing is a valid answer.

CONTEXT:
{context}

PASSAGE:
{passage}

Answer with JSON: {{"items": [...]}}"""

LANGUAGE_NAMES = {"fr": "French", "en": "English"}

# llm(prompt, schema) -> parsed JSON
LLMFn = Callable[[str, dict], dict]


@dataclass
class Chunk:
    recording_id: str
    number: int
    context: list[Utterance]
    utterances: list[Utterance]

    @property
    def key(self) -> str:
        text = "\n".join(u.text for u in self.utterances)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def chunk_utterances(utts: list[Utterance], max_words: int = CHUNK_WORDS) -> list[Chunk]:
    by_rec: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        by_rec[u.recording_id].append(u)
    chunks = []
    for rid, rec_utts in by_rec.items():
        cur: list[Utterance] = []
        n = 0
        start = 0
        for i, u in enumerate(rec_utts):
            cur.append(u)
            n += len(u.text.split())
            if n >= max_words or i == len(rec_utts) - 1:
                ctx = rec_utts[max(0, start - CONTEXT_UTTERANCES):start]
                chunks.append(Chunk(rid, len([c for c in chunks if c.recording_id == rid]), ctx, cur))
                start = i + 1
                cur, n = [], 0
    return chunks


def build_prompt(chunk: Chunk, language: str) -> str:
    return PROMPT.format(
        language_name=LANGUAGE_NAMES.get(language, language),
        devices="\n".join(f"- {k}: {v}" for k, v in DEVICES.items()),
        context="\n".join(f"  {u.text}" for u in chunk.context) or "  (start of recording)",
        passage="\n".join(f"[{u.index}] {u.text}" for u in chunk.utterances),
    )


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^\w']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_in(quote: str, passage: str) -> bool:
    """True if the quote appears in the passage; "…" may elide text between parts that appear in order."""
    parts = [_norm(x) for x in re.split(r"\.\.\.|…|\[\.\.\.\]", quote)]
    parts = [x for x in parts if x]
    pos = 0
    for part in parts:
        pos = passage.find(part, pos)
        if pos < 0:
            return False
        pos += len(part)
    return bool(parts)


def verify(items: list[dict], chunk: Chunk) -> tuple[list[dict], list[dict]]:
    """Keep annotations whose quote really appears in the passage; fix their line numbers."""
    by_index = {u.index: u for u in chunk.utterances}
    passage = " ".join(_norm(u.text) for u in chunk.utterances)
    kept, rejected = [], []
    for it in items:
        q = _norm(it.get("quote", ""))
        if not q or not quote_in(it.get("quote", ""), passage) or it.get("device") not in DEVICES:
            rejected.append(it)
            continue
        lines = [i for i in it.get("utterances", []) if i in by_index]
        hits = [u.index for u in chunk.utterances if _norm(u.text) and (_norm(u.text) in q or q in _norm(u.text))]
        lines = sorted(set(lines) | set(hits)) or hits
        first = by_index[lines[0]] if lines else None
        kept.append({
            "device": it["device"], "quote": it["quote"].strip(), "why": it.get("why", "").strip(),
            "recording": chunk.recording_id, "utterances": lines, "t": first.start if first else None,
        })
    return kept, rejected


@dataclass
class ChunkResult:
    chunk: Chunk
    items: list[dict]
    rejected: int
    cached: bool
    seconds: float


class RhetoricAnnotator:
    def __init__(self, llm: LLMFn, cache_dir: Path, model_id: str, thinking: bool):
        self.llm = llm
        self.cache_dir = cache_dir
        self.tag = hashlib.sha256(f"{PROMPT_VERSION}|{model_id}|{thinking}".encode()).hexdigest()[:8]

    def _cache_path(self, chunk: Chunk) -> Path:
        return self.cache_dir / f"{chunk.recording_id}__{chunk.number:03d}__{chunk.key}__{self.tag}.json"

    def run(self, utts: list[Utterance], language: str) -> Iterator[ChunkResult]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunk_utterances(utts):
            path = self._cache_path(chunk)
            if path.exists():
                data = json.loads(path.read_text())
                yield ChunkResult(chunk, data["items"], data["rejected"], True, 0.0)
                continue
            t = time.time()
            raw = self.llm(build_prompt(chunk, language), SCHEMA)
            items, rejected = verify(raw.get("items", []), chunk)
            path.write_text(json.dumps({"items": items, "rejected": len(rejected), "rejected_items": rejected},
                                       ensure_ascii=False, indent=1) + "\n")
            yield ChunkResult(chunk, items, len(rejected), False, time.time() - t)


def aggregate(results: list[ChunkResult], n_words: int, model: str) -> dict:
    items = [it for r in results for it in r.items]
    n_recs = len({r.chunk.recording_id for r in results})
    by_device: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_device[it["device"]].append(it)
    devices = []
    for name, its in sorted(by_device.items(), key=lambda kv: -len(kv[1])):
        recs = Counter(i["recording"] for i in its)
        queues: dict[str, list[dict]] = defaultdict(list)
        for i in sorted(its, key=lambda i: i["t"] or 0):
            queues[i["recording"]].append(i)
        ex: list[dict] = []
        while len(ex) < 5 and any(queues.values()):  # round-robin across recordings
            for q in queues.values():
                if q and len(ex) < 5:
                    i = q.pop(0)
                    ex.append({k: i[k] for k in ("quote", "why", "recording", "t")})
        devices.append({"device": name, "definition": DEVICES[name], "count": len(its),
                        "per_1k": round(1000 * len(its) / n_words, 2) if n_words else 0,
                        "recordings": len(recs), "examples": ex})
    return {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": len(results),
        "recordings": n_recs,
        "words": n_words,
        "annotations": len(items),
        "rejected_unverifiable": sum(r.rejected for r in results),
        "devices": devices,
        "absent": [d for d in DEVICES if d not in by_device],
        "all": items,
    }


def render(r: dict, speaker: str) -> str:
    L = [f"# Rhetorical profile: {speaker}", "",
         f"Annotated by `{r['model']}` over {r['chunks']} passages ({r['words']:,} words, {r['recordings']} recordings). "
         f"{r['annotations']} annotations kept; {r['rejected_unverifiable']} discarded because their quote "
         f"was not found in the transcript.", "",
         "These are model judgments, not measurements. Read the examples before trusting a count.", "",
         "| device | count | per 1k words | in recordings |", "|---|---|---|---|"]
    L += [f"| {d['device']} | {d['count']} | {d['per_1k']} | {d['recordings']}/{r['recordings']} |" for d in r["devices"]]
    if r["absent"]:
        L += ["", f"Not found: {', '.join(r['absent'])}"]
    for d in r["devices"]:
        L += ["", f"## {d['device']}", "", f"_{d['definition']}_", ""]
        L += [f"- “{e['quote']}” — {e['why']} _({e['recording']})_" for e in d["examples"]]
    return "\n".join(L) + "\n"
