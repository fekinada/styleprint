"""Clean transcripts and split them into utterances.

Speech has no reliable sentences, and Whisper's punctuation is inconsistent, so an
utterance ends at sentence-final punctuation *or* at a pause longer than
PAUSE_S between words. Utterances are the unit for all syntax and discourse metrics.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from styleprint.corpus import Corpus

PAUSE_S = 0.5
# Unpunctuated runs with no long pause would otherwise grow without bound;
# once an utterance reaches LONG_RUN words, a shorter pause also ends it.
LONG_RUN = 20
SHORT_PAUSE_S = 0.2
FINAL_PUNCT = (".", "?", "!", "…")

# Segments Whisper invents over silence or music, rather than speech.
NO_SPEECH_PROB = 0.8
MIN_LOGPROB = -1.0


@dataclass
class Utterance:
    recording_id: str
    index: int
    start: float | None
    end: float | None
    text: str
    split: str  # "punct", "pause" or "end": why the utterance ended

    @property
    def ends_with(self) -> str:
        return self.text[-1] if self.text and self.text[-1] in "?!." else ""


def _is_hallucination(seg: dict) -> bool:
    nsp, lp = seg.get("no_speech_prob"), seg.get("avg_logprob")
    return nsp is not None and lp is not None and nsp > NO_SPEECH_PROB and lp < MIN_LOGPROB


def _join(words: list[str], french_spacing: bool = True) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+(?=['’-])|(?<=['’])\s+", "", text)  # Whisper splits c'est into c + 'est
    text = re.sub(r"\s+([,.;:…])", r"\1", text)  # no space before these, even in French
    if french_spacing:
        text = re.sub(r"\s*([?!])", r" \1", text)  # French: space before ? and !
    else:
        text = re.sub(r"\s+([?!])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _french_spacing(language: str | None) -> bool:
    from styleprint.lexicons import LEXICONS

    return LEXICONS.get(language or "", {}).get("french_spacing", language == "fr")


def utterances_from_transcript(transcript: dict, pause_s: float = PAUSE_S) -> list[Utterance]:
    rid = transcript["recording_id"]
    fs = _french_spacing(transcript.get("language"))
    words = [w for seg in transcript["segments"] if not _is_hallucination(seg) for w in seg["words"]
             if w["word"] and w["word"] != "..."]
    out: list[Utterance] = []
    cur: list[dict] = []

    def flush(reason: str) -> None:
        if cur:
            text = _join([w["word"] for w in cur], fs)
            if re.search(r"\w", text):
                out.append(Utterance(rid, len(out), cur[0]["start"], cur[-1]["end"], text, reason))
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w["word"].endswith(FINAL_PUNCT):
            flush("punct")
        elif nxt is not None and nxt["start"] is not None and w["end"] is not None:
            gap = nxt["start"] - w["end"]
            if gap > pause_s or (len(cur) >= LONG_RUN and gap > SHORT_PAUSE_S):
                flush("pause")
    flush("end")
    return out


def utterances_from_text(text: str, recording_id: str = "text", language: str | None = None) -> list[Utterance]:
    """Split plain text (e.g. generated output) into utterances, for comparison with a profile."""
    parts = re.split(r"(?<=[.?!…])\s+|\n+", text)
    fs = _french_spacing(language)
    return [Utterance(recording_id, i, None, None, _join(p.split(), fs), "punct")
            for i, p in enumerate(x for x in parts if re.search(r"\w", x))]


def segment_speaker(corpus: Corpus, speaker_id: str, pause_s: float = PAUSE_S) -> list[Utterance]:
    """Segment every transcribed recording and write utterances.jsonl for the speaker."""
    d = corpus.speaker_dir(speaker_id)
    utts: list[Utterance] = []
    for rec in corpus.recordings(speaker_id):
        path = d / "transcripts" / f"{rec.id}.json"
        if path.exists():
            utts.extend(utterances_from_transcript(json.loads(path.read_text()), pause_s))
    out = d / "utterances.jsonl"
    out.write_text("".join(json.dumps(asdict(u), ensure_ascii=False) + "\n" for u in utts))
    return utts


def load_utterances(path: Path) -> list[Utterance]:
    return [Utterance(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
