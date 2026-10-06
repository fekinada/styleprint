"""Style card: the measured profile rewritten as short instructions a model can follow.

Models follow qualitative rules ("short utterances, often without a verb") far
better than statistics ("median 8.0"), so numbers become rough frequencies and
explicit do/don't lists. Only portable style goes in; topic vocabulary and
channel rituals stay out. Every example quoted comes from the speaker's own
profile. The card is saved as Markdown so it can be read and edited by hand.
"""

from __future__ import annotations

from styleprint.lexicons import LEXICONS

# Markers below this many occurrences, or seen in fewer recordings, are too thin to prescribe.
MIN_COUNT = 5
MIN_RECORDINGS = 3
# Below this many words, "never used" is just "not seen yet".
MIN_WORDS_FOR_ABSENCE = 2000
# Measured, but never prescribed: subscribe/like rituals would leak into every topic.
NOT_PRESCRIBED = {"channel"}

RHETORIC_GUIDANCE = {
    "counterargument": "Voice the listener's objection or question before they do, then answer it.",
    "anecdote": "Drop in a short personal memory or the story behind the subject, then come straight back.",
    "reframing": "Correct the usual idea of something: the real version, what actually matters.",
    "list": "Quick three-part runs of items or adjectives.",
    "emphatic_repetition": "Repeat a word or phrase for emphasis.",
    "humor": "Light teasing of the audience or of themself.",
    "authority": "Back advice with their own experience and practice.",
    "rhetorical_question": "Ask a question and answer it straight away.",
    "contrast": "Set two options side by side.",
    "analogy": "Explain with an everyday comparison.",
    "provocative_opening": "Open with a hook that grabs attention.",
}


def _every(per_1k: float) -> str:
    """Turn a rate into a rough, human frequency."""
    if per_1k <= 0:
        return "never"
    words = 1000 / per_1k
    if words < 150:
        return f"about once every {round(words, -1):.0f} words"
    if words < 600:
        return f"about once every {round(words, -2):.0f} words"
    return "occasionally"


def _share(x: float) -> str:
    for frac, text in ((0.15, "rarely"), (0.35, "about a third of the time"), (0.6, "about half the time"),
                       (0.85, "most of the time")):
        if x < frac:
            return text
    return "almost always"


def _pct(x: float | None) -> str:
    return "a few" if x is None else f"{round(x * 100 / 5) * 5:.0f}%"


def _q(text: str) -> str:
    return f"« {text} »"


def build(ling: dict, rhet: dict | None, speaker: str) -> str:
    lx = LEXICONS[ling["language"]]
    text = lx["text"]
    lex, syn, dis, reg = ling["lexical"], ling["syntax"], ling["discourse"], ling["register"]
    ul = syn["utterance_length"]
    L = [f"# Style card: {speaker}", "",
         "Generated from the measured profile. Edit freely; generation uses this file as-is.", ""]

    # --- voice & address
    L += ["## Voice and address", ""]
    if reg["address"]["per_1k"] >= 10:
        L.append("- Talk directly to the listener: what they'll see, what they can do, what they're probably thinking.")
    strong = [p for p in reg["preferences"] if p["use_count"] >= 5 and p["use_count"] >= 5 * max(p["avoid_count"], 1)]
    for p in strong:
        L.append(f"- Say **{p['use']}**, not *{p['avoid']}* ({p['use']} : {p['avoid']} = {p['use_count']} : {p['avoid_count']}).")
    pron = {r["pronoun"]: r for r in lex["pronouns"]}
    fp = pron.get(lx["first_person"], {})
    if fp.get("baseline_per_1k") and fp["per_1k"] / fp["baseline_per_1k"] > 1.5:
        shown = "I" if lx["first_person"] == "i" else lx["first_person"]
        L.append(f"- Talk in the first person a lot (**{shown}**): what they do, what they prefer, what they remember.")
    L.append("")

    # --- rhythm
    frag = syn["fragments"]["share"]
    frag_ex = [e["text"] for e in syn["fragments"]["examples"] if len(e["text"].split()) <= 8][:3]
    if (ul.get("long_share") or 0) >= 0.15:
        rhythm = (f"- Uneven rhythm: short bursts **and** long runs. About {_pct(ul.get('short_share'))} of utterances "
                  f"are 7 words or fewer, but about {_pct(ul.get('long_share'))} run past 13 words: thoughts chained "
                  f"with commas and {text['chain_words']}, without a full stop. Never a steady series of tidy, equal sentences.")
    else:
        rhythm = (f"- Short, punchy utterances: about {_pct(ul.get('short_share'))} are 7 words or fewer, typically "
                  f"{ul['median']:.0f} words; long runs are rare. Still vary the length: one-word reactions next to "
                  "full sentences.")
    L += ["## Rhythm and sentences", "", rhythm,
          "- Build sentences the way people talk, not with written subordinate clauses.",
          f"- About {round(frag * 100, -1):.0f}% of utterances have no verb at all: exclamations, afterthoughts, "
          "quick reactions" + (f" ({', '.join(_q(x) for x in frag_ex)})" if frag_ex else "") + ".",
          "- Common ways to start an utterance: " + ", ".join(_q(o["words"]) for o in syn["openings_bigram"][:8]) + ".",
          ]
    neg = reg["negation"]
    if neg["share_full"] is not None and neg["negations"] >= 5 and neg["share_full"] < 0.9:
        L.append("- " + text["negation_card"].format(freq=_share(1 - neg["share_full"])))
    display = lx["imperative_display"]
    imps = [display.get(x["verb"], x["verb"].capitalize()) for x in syn["imperatives"]["top_verbs"]
            if x["count"] >= 2 and (not display or x["verb"] in display)]
    if imps:
        L.append("- Short commands to the listener: " + ", ".join(_q(x) for x in imps[:6]) + ".")
    L.append("")

    # --- markers
    L += ["## Words that carry their voice", "",
          "Rough rates, from their speech. Spread them out: never two in a row, never one in every sentence.", ""]
    regular, occasional = [], []
    for cat, markers in dis["markers"].items():
        if cat in NOT_PRESCRIBED or cat == "repair":
            continue
        for m in markers:
            if m["count"] >= MIN_COUNT and m["recordings"] >= MIN_RECORDINGS:
                (regular if m["per_1k"] >= 1000 / 600 else occasional).append(m)
    regular.sort(key=lambda m: -m["per_1k"])
    L += [f"- **{m['marker']}**: {_every(m['per_1k'])}" for m in regular]
    if occasional:
        L.append("- Now and then: " + ", ".join(f"*{m['marker']}*" for m in occasional) + ".")
    fam = sorted(reg["familiar_words"]["words"].items(), key=lambda kv: -kv[1])
    if fam:
        L.append("- Light familiar vocabulary, sparingly: " + ", ".join(w for w, _ in fam[:10]) + ".")
    grammar = lx["grammar_words"]
    phr = [s["phrase"] for s in dis["signature_phrases"]
           if s["kind"] == "phrasing" and s["recordings"] >= MIN_RECORDINGS
           and not set(s["phrase"].replace("'", "' ").split()) <= grammar]
    if phr:
        L.append("- Recurring turns of phrase: " + ", ".join(_q(p) for p in phr[:10]) + ".")
    L.append("")

    # --- absences
    never = [m for cat, ms in dis["absent_markers"].items() if cat not in NOT_PRESCRIBED for m in ms] \
        if ling["corpus"]["words"] >= MIN_WORDS_FOR_ABSENCE else []
    never += [p["avoid"] for p in strong if p["avoid_count"] == 0]
    L += ["## Never", ""]
    if never:
        L.append("- Never: " + ", ".join(f"*{m}*" for m in never) + ".")
    L += [f"- No written or formal connectors ({', '.join(lx['formal_connectors'])}).",
          "- No lists with bullet points, no headings, no stage directions.", ""]

    # --- rhetoric, illustrated with the speaker's own annotated quotes
    if rhet and rhet.get("devices"):
        L += ["## Rhetorical habits", ""]
        for d in rhet["devices"]:
            if d["recordings"] >= 2 and d["device"] in RHETORIC_GUIDANCE:
                ex = f" e.g. {_q(d['examples'][0]['quote'])}" if d.get("examples") else ""
                L.append(f"- **{d['device'].replace('_', ' ')}** ({d['count']}× in {d['recordings']} recordings): "
                         f"{RHETORIC_GUIDANCE[d['device']]}{ex}")
        L.append("")

    L += ["## Keep it portable", "",
          "- Copy *how* they talk, not *what* they talk about: leave out the subjects, objects and vocabulary "
          "of the source recordings unless the topic calls for them.",
          "- Leave out channel rituals (greeting formulas, asking to subscribe, plugging products) "
          "unless asked for an intro or outro.", ""]
    return "\n".join(L)
