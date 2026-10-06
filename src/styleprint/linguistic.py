"""Linguistic style profile: lexical, syntactic, discourse and register features.

Works on utterances, so the same code profiles the reference corpus and, later,
generated text (via segment.utterances_from_text) for comparison.

Every rate is per 1,000 words; word-level rates are compared with general usage
(wordfreq) so the profile shows what is *distinctive*, not just frequent. Each
feature keeps verbatim examples, because generation imitates examples far better
than numbers.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache

from styleprint.lexicons import LEXICONS
from styleprint.segment import Utterance

N_EXAMPLES = 4
MATTR_WINDOW = 100
STYLE_POS = {"ADV", "INTJ", "PRON", "CCONJ", "SCONJ", "ADP", "DET", "AUX", "PART"}
LONG_UTTERANCE = 13  # words
LENGTH_BINS = [(1, 3), (4, 7), (8, 12), (13, 20), (21, 35), (36, None)]


@cache
def load_nlp(model: str):
    import spacy

    return spacy.load(model, disable=["ner"])


def _word_frequency(word: str, lang: str) -> float:
    from wordfreq import word_frequency

    return word_frequency(word, lang)


@dataclass
class Parsed:
    utt: Utterance
    doc: object  # spacy Doc
    norm: str  # lowercased text with punctuation, for marker regexes
    words: list  # word tokens (no punctuation)


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


def _is_word(tok) -> bool:
    return any(ch.isalpha() for ch in tok.text)


def _lw(tok) -> str:
    return _norm(tok.text).lstrip("-")  # spaCy splits envoyez-moi into envoyez + -moi


def join_tokens(tokens: list[str]) -> str:
    """Re-join word tokens: no space after a French elision (l'huile) or before an English clitic (it's, don't)."""
    text = " ".join(tokens)
    return re.sub(r"(?<=['’]) | (?=['’]\w|n't\b)", "", text)


def per_1k(count: int, n: int) -> float:
    return round(1000 * count / n, 2) if n else 0.0


def examples(utts: list[Utterance], k: int = N_EXAMPLES) -> list[dict]:
    """Pick up to k examples, spread across recordings, preferring readable lengths."""
    by_rec: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        by_rec[u.recording_id].append(u)
    for rid in by_rec:
        by_rec[rid].sort(key=lambda u: (not 4 <= len(u.text.split()) <= 30, u.index))
    picked: list[Utterance] = []
    seen: set[str] = set()
    while len(picked) < k and any(by_rec.values()):
        for rid in list(by_rec):
            if by_rec[rid] and len(picked) < k:
                u = by_rec[rid].pop(0)
                if u.text not in seen:
                    seen.add(u.text)
                    picked.append(u)
    return [{"text": u.text, "recording": u.recording_id, "t": u.start} for u in picked]


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float | None:
    """Moving-average type-token ratio: lexical diversity that doesn't shrink with corpus size."""
    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 3) if tokens else None
    counts = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, len(tokens)):
        out, inn = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inn] += 1
        total += len(counts)
    return round(total / (len(tokens) - window + 1) / window, 3)


def is_imperative(tok, prev, lex: dict) -> bool:
    """Commands addressed to the listener, by a per-language rule.

    spaCy's Mood=Imp tag misses most imperatives in transcribed speech, so these are rule-based:
    - fr_endings: clause-initial -ez/-ons verb form with no subject before it (regardez, coupez, allons)
    - en_base: clause-initial base-form verb with no subject (look, grab your knife, please sit down)
    """
    clause_start = prev is None or prev.is_punct or _lw(prev) in lex["imperative_after"]
    if not clause_start:
        return False
    t = _lw(tok)
    if lex["imperative"] == "fr_endings":
        if t in lex["imperative_stop"] or len(t) < 4 or not t.endswith(lex["imperative_endings"]):
            return False
        return not (t.endswith("ons") and tok.pos_ not in ("VERB", "AUX"))  # oignons, maisons...
    if lex["imperative"] == "en_base":
        return (tok.tag_ == "VB" and tok.pos_ in ("VERB", "AUX")
                and not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in tok.children))
    return False


def _finite(tok) -> bool:
    return "Fin" in tok.morph.get("VerbForm")


def finite_clauses(p: Parsed, lex: dict) -> list:
    """Clause heads: verbs (or copula predicates) that are finite themselves or via an auxiliary."""
    heads = []
    prev = None
    for tok in p.doc:
        is_head = not tok.dep_.startswith("aux") and tok.dep_ != "cop"
        fin = is_head and (_finite(tok) or any(
            (c.dep_.startswith("aux") or c.dep_ == "cop") and _finite(c) for c in tok.children))
        if fin or (_is_word(tok) and is_imperative(tok, prev, lex)):
            heads.append(tok)
        if not tok.is_space:
            prev = tok
    return heads


# Verbs whose compound tenses use être without being passive: "je suis tombé".
ETRE_VERBS = {"aller", "venir", "arriver", "partir", "sortir", "entrer", "rentrer", "rester", "tomber", "naître",
              "mourir", "devenir", "revenir", "retourner", "monter", "descendre", "passer", "apparaître"}
REFLEXIVE = {"me", "m'", "te", "t'", "se", "s'"}


def _is_passive(tok, lex: dict) -> bool:
    if lex["passive"] == "en":
        return tok.pos_ == "VERB" and any(c.dep_ == "auxpass" for c in tok.children)
    # fr: être + past participle, excluding être-verbs, adjectives and reflexives ("vous vous êtes lavé").
    if tok.pos_ != "VERB" or "Part" not in tok.morph.get("VerbForm") or tok.lemma_ in ETRE_VERBS:
        return False
    if not any(c.dep_ in ("aux:pass", "aux:tense") and c.lemma_ == "être" for c in tok.children):
        return False
    before = [_lw(t) for t in tok.doc[max(0, tok.i - 4):tok.i]]
    reflexive = any(t in REFLEXIVE for t in before) or before.count("vous") >= 2 or before.count("nous") >= 2
    return not reflexive


def imperatives(p: Parsed, lex: dict) -> list[str]:
    out, prev = [], None
    for tok in p.doc:
        if _is_word(tok) and is_imperative(tok, prev, lex):
            out.append(_lw(tok))
        if not tok.is_space:
            prev = tok
    return out


# --- feature groups ---------------------------------------------------------


def lexical(parsed: list[Parsed], lang: str, lex: dict) -> dict:
    words = [tok for p in parsed for tok in p.words]
    forms = [_lw(t) for t in words]
    n = len(forms)

    top_lemmas = {}
    for pos in ("VERB", "NOUN", "ADJ", "ADV"):
        c = Counter(t.lemma_.lower() for t in words if t.pos_ == pos)
        top_lemmas[pos] = [{"lemma": w, "count": k, "per_1k": per_1k(k, n)} for w, k in c.most_common(15)]

    pos_of: dict[str, Counter] = defaultdict(Counter)
    recs_of: dict[str, set] = defaultdict(set)
    for p in parsed:
        for t in p.words:
            pos_of[_lw(t)][t.pos_] += 1
            recs_of[_lw(t)].add(p.utt.recording_id)
    n_recs = len({p.utt.recording_id for p in parsed})
    min_recs = 2 if n_recs >= 2 else 1

    counts = Counter(forms)
    rows = []
    for word, c in counts.items():
        if c < 3 or len(word) < 2 or not word.replace("'", "").isalpha():
            continue
        base = _word_frequency(word, lang)
        rate = c / n
        rows.append({
            "word": word, "count": c, "per_1k": per_1k(c, n),
            "baseline_per_1k": round(1000 * base, 3),
            "log2_ratio": round(math.log2(rate / base), 2) if base else None,
            "recordings": len(recs_of[word]),
            "pos": pos_of[word].most_common(1)[0][0],
        })

    def bucket(row):
        if row["pos"] in STYLE_POS:
            return "function_discourse"
        return "evaluative" if row["pos"] == "ADJ" else "content_topic"

    distinctive: dict[str, list] = {"function_discourse": [], "evaluative": [], "content_topic": []}
    ranked = sorted((r for r in rows if r["log2_ratio"] is not None and r["recordings"] >= min_recs),
                    key=lambda r: -r["log2_ratio"])
    for r in ranked:
        b = distinctive[bucket(r)]
        if len(b) < 20 and r["log2_ratio"] > 1:
            b.append(r)

    underused = sorted(
        (r for r in rows if r["log2_ratio"] is not None and r["baseline_per_1k"] >= 1 and r["log2_ratio"] < -1
         and r["word"].isalpha()),
        key=lambda r: r["log2_ratio"],
    )[:10]

    pronouns = []
    for pr in lex["pronouns"]:
        c = counts[pr]
        base = _word_frequency(pr.rstrip("'"), lang)
        pronouns.append({"pronoun": pr, "count": c, "per_1k": per_1k(c, n),
                         "baseline_per_1k": round(1000 * base, 2) if base else None})

    return {
        "words": n,
        "types": len(counts),
        "mattr": mattr(forms),
        "mattr_window": MATTR_WINDOW,
        "top_lemmas": top_lemmas,
        "distinctive": distinctive,
        "underused": underused,
        "pronouns": pronouns,
    }


def syntax(parsed: list[Parsed], lex: dict) -> dict:
    lens = [len(p.words) for p in parsed]
    clauses = [finite_clauses(p, lex) for p in parsed]
    subord = [sum(1 for t in p.doc if t.pos_ == "SCONJ" or "Rel" in t.morph.get("PronType")) for p in parsed]
    n_utt = len(parsed)
    n_words = sum(lens)

    hist = []
    for lo, hi in LENGTH_BINS:
        k = sum(1 for x in lens if x >= lo and (hi is None or x <= hi))
        hist.append({"words": f"{lo}+" if hi is None else f"{lo}-{hi}", "count": k, "share": round(k / n_utt, 3)})

    fragments = [p.utt for p, c in zip(parsed, clauses) if not c]

    first = Counter(_lw(p.words[0]) for p in parsed if p.words)
    first2 = Counter(join_tokens([_lw(t) for t in p.words[:2]]) for p in parsed if len(p.words) >= 2)
    by_first: dict[str, list] = defaultdict(list)
    for p in parsed:
        if p.words:
            by_first[_lw(p.words[0])].append(p.utt)

    questions = [p for p in parsed if p.utt.text.rstrip().endswith("?")]
    tags = [p for p in questions if re.search(lex["tags"], p.norm)]
    other_q = [p for p in questions if p not in tags]

    imps = [(p, imperatives(p, lex)) for p in parsed]
    imp_utts = [p.utt for p, v in imps if v]
    imp_verbs = Counter(v for _, vs in imps for v in vs)

    passive = [p.utt for p in parsed if any(_is_passive(t, lex) for t in p.doc)]

    return {
        "utterances": n_utt,
        "utterance_length": {
            "mean": round(statistics.mean(lens), 1),
            "median": statistics.median(lens),
            "p10": statistics.quantiles(lens, n=10)[0] if n_utt >= 10 else None,
            "p90": statistics.quantiles(lens, n=10)[-1] if n_utt >= 10 else None,
            "long_share": round(sum(x >= LONG_UTTERANCE for x in lens) / n_utt, 3),
            "short_share": round(sum(x <= 7 for x in lens) / n_utt, 3),
            # Coefficient of variation: a uniform staccato scores low, real speech mixes bursts and runs.
            "cv": round(statistics.pstdev(lens) / statistics.mean(lens), 2),
            "histogram": hist,
        },
        "clauses_per_utterance": round(sum(map(len, clauses)) / n_utt, 2),
        "subordinators_per_utterance": round(sum(subord) / n_utt, 2),
        "fragments": {"share": round(len(fragments) / n_utt, 3), "examples": examples(fragments)},
        "openings": [
            {"word": w, "count": k, "share": round(k / n_utt, 3), "examples": examples(by_first[w], 2)}
            for w, k in first.most_common(12)
        ],
        "openings_bigram": [{"words": w, "count": k} for w, k in first2.most_common(10)],
        "questions": {
            "share": round(len(questions) / n_utt, 3),
            "tag_questions": len(tags),
            "tag_examples": examples([p.utt for p in tags]),
            "other_examples": examples([p.utt for p in other_q], 6),
        },
        "imperatives": {
            "per_100_utterances": round(100 * len(imp_utts) / n_utt, 1),
            "top_verbs": [{"verb": v, "count": k} for v, k in imp_verbs.most_common(15)],
            "examples": examples(imp_utts),
        },
        "passive": {
            "per_1k_words": per_1k(len(passive), n_words),
            "examples": examples(passive),
            "note": "parser-based and approximate on speech; check the examples",
        },
    }


def _match_markers(parsed: list[Parsed], patterns: list[tuple[str, str]], n_words: int) -> tuple[list, list]:
    found, absent = [], []
    for label, pattern in patterns:
        rx = re.compile(pattern)
        per_rec: Counter = Counter()
        hits = []
        for p in parsed:
            k = len(rx.findall(p.norm))
            if k:
                per_rec[p.utt.recording_id] += k
                hits.append(p.utt)
        total = sum(per_rec.values())
        if not total:
            absent.append(label)
            continue
        found.append({"marker": label, "count": total, "per_1k": per_1k(total, n_words),
                      "recordings": len(per_rec), "by_recording": dict(per_rec), "examples": examples(hits, 3)})
    found.sort(key=lambda r: -r["count"])
    return found, absent


def _ngrams(parsed: list[Parsed], lo: int = 3, hi: int = 5) -> list[dict]:
    counts: Counter = Counter()
    recs: dict[tuple, set] = defaultdict(set)
    where: dict[tuple, list] = defaultdict(list)
    nouny: set[tuple] = set()
    for p in parsed:
        toks = [_lw(t) for t in p.words]
        for n in range(lo, hi + 1):
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i:i + n])
                if any(t.pos_ in ("NOUN", "PROPN") for t in p.words[i:i + n]):
                    nouny.add(g)
                counts[g] += 1
                recs[g].add(p.utt.recording_id)
                where[g].append(p.utt)
    n_recs = len({p.utt.recording_id for p in parsed})
    keep = [g for g, c in counts.items() if c >= 3 and len(recs[g]) >= min(2, n_recs)]
    # Drop an n-gram when a longer one containing it is (almost) as frequent.
    keep_set = set(keep)
    result = []
    for g in keep:
        longer = [h for h in keep_set if len(h) > len(g) and any(h[i:i + len(g)] == g for i in range(len(h) - len(g) + 1))]
        if any(counts[h] >= 0.8 * counts[g] for h in longer):
            continue
        result.append(g)
    result.sort(key=lambda g: (-counts[g] * len(g), g))
    # "phrasing" is portable to any topic; "formula" contains nouns (channel rituals or topic terms).
    return [{"phrase": join_tokens(list(g)), "count": counts[g], "recordings": len(recs[g]),
             "kind": "formula" if g in nouny else "phrasing", "examples": examples(where[g], 2)}
            for g in result[:40]]


def discourse(parsed: list[Parsed], lex: dict) -> dict:
    n_words = sum(len(p.words) for p in parsed)
    categories, absent = {}, {}
    for cat, patterns in lex["markers"].items():
        categories[cat], absent[cat] = _match_markers(parsed, patterns, n_words)

    emphatic = lex["emphatic"]
    doubled, stutter = [], []
    for p in parsed:
        toks = [_lw(t) for t in p.doc]  # punctuation kept: "ça, ça" is not a repeat
        for i, (a, b) in enumerate(zip(toks, toks[1:])):
            if a != b or not any(ch.isalpha() for ch in a) or a in lex["repeat_ok"]:
                continue
            run = 2 + sum(1 for t in toks[i + 2:i + 6] if t == a)
            (doubled if a in emphatic or run >= 3 else stutter).append((a, p.utt))
    return {
        "markers": categories,
        "absent_markers": absent,
        "self_correction": {
            "immediate_repeats_per_1k": per_1k(len(stutter), n_words),
            "examples": examples([u for _, u in stutter]),
            "note": "Whisper removes most disfluencies, so this undercounts real speech",
        },
        "emphatic_repetition": {
            "count": len(doubled),
            "words": dict(Counter(w for w, _ in doubled)),
            "examples": examples([u for _, u in doubled]),
        },
        "signature_phrases": _ngrams(parsed),
    }


def _negation(parsed: list[Parsed], lex: dict) -> tuple[list, list]:
    """Verbal negations split into (full, informal): fr keeps vs drops "ne"; en writes "not" vs "n't"."""
    full, informal = [], []
    for p in parsed:
        toks = list(p.doc)
        for i, tok in enumerate(toks):
            t = _lw(tok)
            if lex["negation"] == "fr_ne":
                if t not in lex["negation_words"]:
                    continue
                window = toks[max(0, i - 4):i]
                if not any(x.pos_ in ("VERB", "AUX") for x in window):
                    continue  # verbless negation: "pas chère", "pas plus"
                (full if any(_lw(x) in lex["ne"] for x in window) else informal).append(p.utt)
            elif lex["negation"] == "en_contraction":
                if t == "n't":
                    informal.append(p.utt)
                elif t == "not" and i > 0 and toks[i - 1].pos_ in ("AUX", "VERB"):
                    # "it's not", "I'm not": contracted on the verb, as informal as "isn't"
                    (informal if toks[i - 1].text.startswith(("'", "’")) else full).append(p.utt)
    return full, informal


def register(parsed: list[Parsed], lex: dict) -> dict:
    n_words = sum(len(p.words) for p in parsed)
    full, informal = _negation(parsed, lex)
    neg = len(full) + len(informal)

    def count(pattern: str) -> int:
        rx = re.compile(pattern)
        return sum(len(rx.findall(p.norm)) for p in parsed)

    prefs = [{"use": x["use"], "avoid": x["avoid"], "use_count": count(x["use_re"]), "avoid_count": count(x["avoid_re"])}
             for x in lex["preferences"]]

    fam_counts: Counter = Counter()
    fam_utts = []
    for p in parsed:
        hit = False
        for pattern in lex["familiar"]:
            for m in re.findall(pattern, p.norm):
                fam_counts[m] += 1
                hit = True
        if hit:
            fam_utts.append(p.utt)
    address = count(lex["address"])
    return {
        "negation": {
            "kind": lex["negation"],
            "negations": neg,
            "share_full": round(len(full) / neg, 3) if neg else None,  # fr: ne kept; en: "not" spelled out
            "informal_examples": examples(informal),
            "full_examples": examples(full),
        },
        "preferences": prefs,
        "address": {"count": address, "per_1k": per_1k(address, n_words)},
        "familiar_words": {"per_1k": per_1k(sum(fam_counts.values()), n_words), "words": dict(fam_counts),
                           "examples": examples(fam_utts)},
    }


def analyze(utterances: list[Utterance], language: str, speaker: str | None = None,
            duration_s: float | None = None) -> dict:
    if language not in LEXICONS:
        raise ValueError(f"no lexicons for language '{language}' (have: {', '.join(LEXICONS)})")
    lex = LEXICONS[language]
    nlp = load_nlp(lex["spacy_model"])
    parsed = []
    for u, doc in zip(utterances, nlp.pipe(u.text for u in utterances)):
        words = [t for t in doc if _is_word(t)]
        if words:
            parsed.append(Parsed(u, doc, _norm(u.text), words))
    if not parsed:
        raise ValueError("no text to analyze")

    lexical_ = lexical(parsed, language, lex)
    return {
        "speaker": speaker,
        "language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {
            "recordings": len({p.utt.recording_id for p in parsed}),
            "utterances": len(parsed),
            "words": lexical_["words"],
            "minutes": round(duration_s / 60, 1) if duration_s else None,
        },
        "lexical": lexical_,
        "syntax": syntax(parsed, lex),
        "discourse": discourse(parsed, lex),
        "register": register(parsed, lex),
        "not_measured": {
            "concreteness": "needs a concreteness lexicon; not bundled yet",
            "rhetoric": "rhetorical questions, contrasts, analogies, anecdotes, reframing: LLM annotation step",
        },
    }
