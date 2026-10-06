"""Render a linguistic profile as a readable Markdown report."""

from __future__ import annotations


def _ratio(per_1k: float, base: float | None) -> str:
    if not base:
        return "–"
    r = per_1k / base
    return f"×{r:.1f}" if r >= 1 else f"÷{1 / r:.1f}" if r > 0 else "never"


def _ex(examples: list[dict], k: int = 2) -> str:
    return " · ".join(f"“{e['text']}”" for e in examples[:k])


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render(p: dict) -> str:
    c, lex, syn, dis, reg = p["corpus"], p["lexical"], p["syntax"], p["discourse"], p["register"]
    L: list[str] = [f"# Linguistic profile: {p['speaker'] or 'text'}", ""]
    mins = f", {c['minutes']} min" if c["minutes"] else ""
    L += [f"{c['recordings']} recordings{mins}, {c['words']:,} words, {c['utterances']} utterances. "
          f"Language: `{p['language']}`. Generated {p['generated_at']}.", "",
          "Rates are per 1,000 words. **×N** compares with general French usage (wordfreq). "
          "Counts under ~5 are anecdotal at this corpus size.", ""]

    # --- lexical
    L += ["## Lexical", "",
          f"- Vocabulary: {lex['types']:,} distinct forms; lexical diversity (MATTR, window "
          f"{lex['mattr_window']}): **{lex['mattr']}**", ""]
    L += ["### Pronouns", ""]
    L += _table(["pronoun", "count", "per 1k", "vs. general"],
                [[r["pronoun"], r["count"], r["per_1k"], _ratio(r["per_1k"], r["baseline_per_1k"])]
                 for r in lex["pronouns"] if r["count"] or (r["baseline_per_1k"] or 0) >= 0.5])
    pairs = {(x["use"], x["avoid"]): x for x in reg["preferences"]}
    shown = [x for (u, a), x in pairs.items() if (a, u) not in pairs or u < a]  # each pair once
    L += ["", "- " + " · ".join(f"{x['use']} : {x['avoid']} = {x['use_count']} : {x['avoid_count']}" for x in shown),
          f"- Addressing the listener: {reg['address']['per_1k']} per 1k words", ""]
    titles = {"function_discourse": "Distinctive function & discourse words (style)",
              "evaluative": "Distinctive adjectives (evaluation, partly topic)",
              "content_topic": "Distinctive content words (mostly topic, not style)"}
    for key, title in titles.items():
        rows = lex["distinctive"][key][:15]
        L += [f"### {title}", ""]
        L += [", ".join(f"**{r['word']}** {_ratio(r['per_1k'], r['baseline_per_1k'])} ({r['count']})" for r in rows)
              or "_none_", ""]
    if lex["underused"]:
        L += ["### Underused compared with general French", "",
              ", ".join(f"{r['word']} {_ratio(r['per_1k'], r['baseline_per_1k'])}" for r in lex["underused"]), ""]
    L += ["### Most frequent lemmas", ""]
    for pos, rows in lex["top_lemmas"].items():
        L.append(f"- **{pos}**: " + ", ".join(f"{r['lemma']} ({r['count']})" for r in rows[:12]))
    L.append("")

    # --- syntax
    ul = syn["utterance_length"]
    L += ["## Syntax", "",
          f"Utterances are split at final punctuation or pauses over 0.5 s.", "",
          f"- Utterance length: median **{ul['median']}** words, mean {ul['mean']}, "
          f"10th–90th percentile {ul['p10']}–{ul['p90']}",
          "- Distribution: " + " · ".join(f"{h['words']}: {h['share']:.0%}" for h in ul["histogram"]),
          f"- Finite clauses per utterance: **{syn['clauses_per_utterance']}**; "
          f"subordinators per utterance: {syn['subordinators_per_utterance']}",
          f"- Fragments (no finite verb): **{syn['fragments']['share']:.0%}** — {_ex(syn['fragments']['examples'], 3)}",
          f"- Questions: {syn['questions']['share']:.1%} of utterances; {syn['questions']['tag_questions']} check-in "
          f"tags — {_ex(syn['questions']['tag_examples'])}",
          f"  - Other questions: {_ex(syn['questions']['other_examples'], 4)}",
          f"- Imperatives: **{syn['imperatives']['per_100_utterances']} per 100 utterances** — "
          + ", ".join(f"{v['verb']} ({v['count']})" for v in syn["imperatives"]["top_verbs"][:10]),
          f"- Passive voice: {syn['passive']['per_1k_words']} per 1k words (approximate) — "
          f"{_ex(syn['passive']['examples'])}", ""]
    L += ["### How utterances open", ""]
    L += _table(["first word", "share", "example"],
                [[o["word"], f"{o['share']:.1%}", _ex(o["examples"], 1)] for o in syn["openings"]])
    L += ["", "Common two-word openings: " + ", ".join(f"*{o['words']}* ({o['count']})" for o in syn["openings_bigram"]), ""]

    # --- discourse
    L += ["## Discourse", ""]
    names = {"discourse_markers": "Discourse markers", "transitions": "Transitions", "hedging": "Hedging",
             "intensifiers": "Intensifiers", "contrast_qualification": "Contrast & qualification",
             "audience_framing": "Audience framing", "repair": "Repair", "evaluative": "Evaluative expressions"}
    for cat, rows in dis["markers"].items():
        L += [f"### {names.get(cat, cat)}", ""]
        if rows:
            L += _table(["marker", "count", "per 1k", "in recordings", "example"],
                        [[r["marker"], r["count"], r["per_1k"], f"{r['recordings']}/{c['recordings']}",
                          _ex(r["examples"], 1)] for r in rows])
        absent = dis["absent_markers"].get(cat)
        if absent:
            L += ["", f"Never used: {', '.join(absent)}"]
        L.append("")
    sc, em = dis["self_correction"], dis["emphatic_repetition"]
    L += ["### Repetition & self-correction", "",
          f"- Immediate repeats (hesitation/repair): {sc['immediate_repeats_per_1k']} per 1k — {_ex(sc['examples'])}",
          f"  - _{sc['note']}_",
          f"- Emphatic doubling: {em['count']} — {', '.join(f'{w} ({k})' for w, k in em['words'].items())}", ""]
    L += ["### Signature phrases", "",
          "Recurring 3–5 word sequences found in at least 2 recordings. *phrasing* carries over to any topic; "
          "*formula* contains nouns (channel rituals or topic).", ""]
    L += _table(["phrase", "count", "recordings", "kind"],
                [[s["phrase"], s["count"], s["recordings"], s["kind"]] for s in dis["signature_phrases"][:30]])
    L.append("")

    # --- register
    from styleprint.lexicons import LEXICONS

    text = LEXICONS[p["language"]]["text"]
    neg = reg["negation"]
    fam = reg["familiar_words"]
    share = "n/a" if neg["share_full"] is None else f"{neg['share_full']:.0%}"
    L += ["## Register", "",
          f"- {text['negation_label'].capitalize()}: **{share}** of {neg['negations']} verbal negations "
          f"({text['negation_written']}) — informal: {_ex(neg['informal_examples'])}",
          f"- Familiar words: {fam['per_1k']} per 1k — "
          + ", ".join(f"{w} ({k})" for w, k in sorted(fam["words"].items(), key=lambda kv: -kv[1])), ""]

    L += ["## Not measured yet", ""] + [f"- **{k}**: {v}" for k, v in p["not_measured"].items()]
    return "\n".join(L) + "\n"
