"""Style-conditioned generation: style card + real example passages → draft → critic → revision.

The prompt carries the style card (rules) and a few real passages (to imitate).
The critic runs the same linguistic analysis on the draft and compares it with
the speaker's profile; the gaps become concrete feedback for a revision. The
version with the fewest style issues wins.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from styleprint.linguistic import analyze
from styleprint.segment import Utterance, utterances_from_text

PASSAGE_WORDS = 130
N_PASSAGES = 4

# What a passage shows, from the rhetoric annotations inside it.
FUNCTION_LABELS = {
    "counterargument": "answering the listener before they object",
    "anecdote": "a personal memory",
    "reframing": "correcting the usual idea",
    "humor": "teasing",
    "authority": "backing advice with their own practice",
    "rhetorical_question": "asking and answering their own question",
    "contrast": "weighing two options",
    "analogy": "an everyday comparison",
}

# chat(messages, temperature) -> text
ChatFn = Callable[[list[dict], float], str]


# --- example passages --------------------------------------------------------


@dataclass
class Passage:
    recording_id: str
    utterances: list[Utterance]
    devices: list[str] = field(default_factory=list)
    role: str = ""

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    @property
    def indices(self) -> set[int]:
        return {u.index for u in self.utterances}


def _windows(utts: list[Utterance], words: int) -> list[list[Utterance]]:
    out, cur, n = [], [], 0
    for u in utts:
        cur.append(u)
        n += len(u.text.split())
        if n >= words:
            out.append(cur)
            cur, n = [], 0
    if cur and (n >= words / 2 or not out):
        out.append(cur)
    elif cur:
        out[-1].extend(cur)
    return out


def select_passages(utts: list[Utterance], rhetoric: dict | None, n: int = N_PASSAGES,
                    words: int = PASSAGE_WORDS) -> list[Passage]:
    """Pick passages by what they do: an opening, a closing, and the richest in rhetorical habits."""
    by_rec: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        by_rec[u.recording_id].append(u)
    annotations = (rhetoric or {}).get("all", [])

    candidates: list[Passage] = []
    for rid, rec_utts in by_rec.items():
        wins = _windows(rec_utts, words)
        for i, w in enumerate(wins):
            p = Passage(rid, w)
            p.devices = sorted({a["device"] for a in annotations
                                if a["recording"] == rid and set(a["utterances"]) & p.indices
                                and a["device"] in FUNCTION_LABELS})
            p.role = "opening" if i == 0 else "closing" if i == len(wins) - 1 else "middle"
            candidates.append(p)

    picked: list[Passage] = []
    used_recs: set[str] = set()

    def take(pool: list[Passage]) -> None:
        fresh = [p for p in pool if p.recording_id not in used_recs] or pool
        if fresh:
            p = fresh[0]
            picked.append(p)
            used_recs.add(p.recording_id)
            candidates.remove(p)

    take(sorted((p for p in candidates if p.role == "opening"), key=lambda p: -len(p.devices)))
    while len(picked) < n - 1 and any(p.role == "middle" for p in candidates):
        covered = {d for p in picked for d in p.devices}
        take(sorted((p for p in candidates if p.role == "middle"),
                    key=lambda p: (-len(set(p.devices) - covered), -len(p.devices))))
    take(sorted((p for p in candidates if p.role == "closing"), key=lambda p: -len(p.devices)))
    return picked[:n]


def describe(p: Passage) -> str:
    parts = {"opening": ["start of a video"], "closing": ["end of a video"], "middle": []}[p.role]
    parts += [FUNCTION_LABELS[d] for d in p.devices]
    return ", ".join(parts) or "explaining as they go"


# --- critic ------------------------------------------------------------------


@dataclass
class Issue:
    metric: str
    expected: str
    got: str
    feedback: str  # instruction for the reviser
    weight: float = 1.0


def _marker_rates(profile: dict) -> dict[str, dict]:
    return {m["marker"]: m for cat in profile["discourse"]["markers"].values() for m in cat}


def critique(target: dict, draft: dict) -> list[Issue]:
    """Compare a draft's profile with the speaker's; return what to fix."""
    from styleprint.lexicons import LEXICONS

    text = LEXICONS[target["language"]]["text"]
    issues: list[Issue] = []
    words = draft["corpus"]["words"]
    tu, du = target["syntax"]["utterance_length"], draft["syntax"]["utterance_length"]

    if du["median"] > 1.4 * tu["median"]:
        issues.append(Issue("utterance_length", f"median {tu['median']:.0f} words", f"median {du['median']:.0f}",
                            "Utterances are too long and written-sounding. Cut them into short spoken bits, "
                            f"chained with commas and {text['chain_words']}.", 2))
    elif du["median"] < 0.6 * tu["median"]:
        issues.append(Issue("utterance_length", f"median {tu['median']:.0f} words", f"median {du['median']:.0f}",
                            f"Utterances are too choppy; let some run on, joined with commas and {text['chain_words']}."))

    # Rhythm: a uniform staccato can have the right median and still sound nothing like the speaker.
    if tu["long_share"] >= 0.15 and du["long_share"] < max(0.05, tu["long_share"] - 0.2):
        issues.append(Issue("long_runs", f"{tu['long_share']:.0%} of utterances 13+ words", f"{du['long_share']:.0%}",
                            "Too choppy and even. The speaker alternates short bursts with long runs: make about a "
                            "quarter of the utterances long (15–30 words), thoughts chained with commas and "
                            f"{text['chain_words']}, without a full stop.", 2))
    elif du["long_share"] > tu["long_share"] + 0.25:
        issues.append(Issue("long_runs", f"{tu['long_share']:.0%} of utterances 13+ words", f"{du['long_share']:.0%}",
                            f"Too many long run-on sentences ({du['long_share']:.0%} of utterances, the speaker: "
                            f"{tu['long_share']:.0%}). Break the long ones up.", 2))
    if du["cv"] < min(0.45, 0.6 * tu["cv"]):
        issues.append(Issue("rhythm_variation", f"CV {tu['cv']}", f"CV {du['cv']}",
                            "Every utterance is about the same length. Vary it: one-word reactions next to long run-ons."))

    tf, df = target["syntax"]["fragments"]["share"], draft["syntax"]["fragments"]["share"]
    if df < tf - 0.12:
        issues.append(Issue("fragments", f"{tf:.0%}", f"{df:.0%}",
                            f"Too tidy: add a few verbless bits (quick reactions, afterthoughts, {text['fragment_example']})."))

    tn, dn = target["register"]["negation"], draft["register"]["negation"]
    if dn["negations"] >= 2 and tn["share_full"] is not None:
        label = f"{tn['share_full']:.0%} full"
        if dn["share_full"] > tn["share_full"] + 0.25:
            issues.append(Issue("negation", label, f"{dn['share_full']:.0%}", text["negation_too_formal"]))
        elif dn["share_full"] < tn["share_full"] - 0.35:
            issues.append(Issue("negation", label, f"{dn['share_full']:.0%}", text["negation_too_casual"]))

    # Strong preferences (on over nous, yeah over yes...): flag the avoided form when clearly above their rate.
    t_words = target["corpus"]["words"]
    for pt, pd in zip(target["register"]["preferences"], draft["register"]["preferences"]):
        if pt["use_count"] < 5 or pt["use_count"] < 5 * max(pt["avoid_count"], 1):
            continue
        t_rate = pt["avoid_count"] / t_words
        got = pd["avoid_count"]
        if (got >= 1) if t_rate == 0 else (got > max(3 * t_rate * words, 2)):
            expected = "never" if t_rate == 0 else f"~{t_rate * words:.1f}×"
            issues.append(Issue(f"prefer:{pt['use']}", expected, f"« {pt['avoid']} » {got}×",
                                f"Say « {pt['use']} », not « {pt['avoid']} ».", 2))

    if target["register"]["address"]["per_1k"] >= 10 and draft["register"]["address"]["count"] == 0 and words >= 100:
        issues.append(Issue("address", "talks to the listener", "never",
                            "Talk to the listener directly, the way the speaker does.", 2))

    # Markers the speaker never uses but the draft does; and their frequent markers: missing or overused.
    from styleprint.stylecard import MIN_WORDS_FOR_ABSENCE

    absent = {m for cat in target["discourse"]["absent_markers"].values() for m in cat} \
        if target["corpus"]["words"] >= MIN_WORDS_FOR_ABSENCE else set()
    rates_t, rates_d = _marker_rates(target), _marker_rates(draft)
    for m in sorted(absent & set(rates_d)):
        issues.append(Issue(f"marker:{m}", "never", f"{rates_d[m]['count']}×",
                            f"The speaker never says « {m} »: remove it.", 1.5))
    n_recs = target["corpus"]["recordings"]
    for m, r in rates_t.items():
        if r["count"] < 10 or r["recordings"] < 3:
            continue
        expected = r["per_1k"] * words / 1000
        got = rates_d.get(m, {}).get("count", 0)
        core = r["recordings"] == n_recs and r["per_1k"] >= 4  # in every recording, frequent
        if core and expected >= 2 and got == 0:
            issues.append(Issue(f"marker:{m}", f"~{expected:.0f}×", "0×",
                                f"Missing « {m} »: the speaker would say it about {expected:.0f} times in a text this long."))
        elif got >= max(4, 3 * expected):
            issues.append(Issue(f"marker:{m}", f"~{expected:.0f}×", f"{got}×",
                                f"Overused « {m} » ({got}×, the speaker would use it ~{max(expected, 1):.0f}×): "
                                "it reads like a caricature.", 1.5))
    return issues


def score(issues: list[Issue]) -> float:
    return round(sum(i.weight for i in issues), 2)


# --- generation --------------------------------------------------------------

SYSTEM = """You write in the voice of {speaker}, reproducing how they speak, in {language_name}.

Output a spoken script: exactly what they would say out loud, written like a transcript of their speech, in
paragraphs. Punctuate the way they breathe: a full stop where they pause, commas where they chain thoughts
without stopping. No title, no stage directions, no markdown, no quotation marks around the text, no translation.

{card}

# Real passages of {speaker} speaking

These are transcripts of the speaker. Imitate the rhythm, the words that glue their speech together and the
way they address the listener. Do NOT reuse the content or subject matter of these passages: your topic is
different.

{passages}"""

USER = """Topic / instruction: {instruction}

Length: about {words} words. Write only the script."""

REVISE = """Here is your draft:

{draft}

A style check against the speaker's real speech found these problems:
{feedback}

Rewrite the whole script fixing them, keeping the same content and length. Write only the script."""

LANGUAGE_NAMES = {"fr": "French", "en": "English"}


@dataclass
class Version:
    text: str
    issues: list[Issue]
    score: float


@dataclass
class Generation:
    instruction: str
    model: str
    system: str
    versions: list[Version]
    best: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def text(self) -> str:
        return self.versions[self.best].text

    def to_dict(self) -> dict:
        return {**asdict(self), "text": self.text}


def clean(text: str) -> str:
    text = re.sub(r"(?m)^\s*(#+ .*|\*\*.*\*\*|[-*] )", "", text)  # headings, bold titles, bullets
    text = re.sub(r"(?m)^\s*[\[(].*?[\])]\s*$", "", text)  # [stage directions]
    text = text.strip().strip("«»\"“”")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_system(speaker: str, language: str, card: str, passages: list[Passage]) -> str:
    blocks = "\n\n".join(f"## Passage {i} ({describe(p)})\n\n" + "\n".join(u.text for u in p.utterances)
                         for i, p in enumerate(passages, 1))
    return SYSTEM.format(speaker=speaker, language_name=LANGUAGE_NAMES.get(language, language),
                         card=card, passages=blocks)


def evaluate(text: str, target: dict, language: str) -> Version:
    draft = analyze(utterances_from_text(text, language=language), language)
    issues = critique(target, draft)
    return Version(text, issues, score(issues))


def generate(chat: ChatFn, model: str, speaker: str, language: str, card: str, passages: list[Passage],
             target: dict, instruction: str, words: int = 250, revisions: int = 1, temperature: float = 0.8,
             on_version: Callable[[int, Version], None] | None = None) -> Generation:
    system = build_system(speaker, language, card, passages)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": USER.format(instruction=instruction, words=words)}]
    versions = [evaluate(clean(chat(messages, temperature)), target, language)]
    if on_version:
        on_version(0, versions[0])
    for i in range(revisions):
        last = versions[-1]
        if not last.issues:
            break
        feedback = "\n".join(f"- {x.feedback}" for x in last.issues)
        msgs = messages + [{"role": "user", "content": REVISE.format(draft=last.text, feedback=feedback)}]
        versions.append(evaluate(clean(chat(msgs, temperature * 0.75)), target, language))
        if on_version:
            on_version(i + 1, versions[-1])
    best = min(range(len(versions)), key=lambda i: (versions[i].score, -i))
    return Generation(instruction, model, system, versions, best)


def report(target: dict, text: str, language: str) -> tuple[dict, list[Issue]]:
    """Profile any text and critique it against the speaker: for `styleprint compare`."""
    draft = analyze(utterances_from_text(text, language=language), language)
    return draft, critique(target, draft)


def to_json(gen: Generation) -> str:
    return json.dumps(gen.to_dict(), ensure_ascii=False, indent=1) + "\n"
