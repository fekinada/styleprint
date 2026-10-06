import pytest

from styleprint.generate import critique
from styleprint.linguistic import analyze, join_tokens
from styleprint.segment import Utterance, utterances_from_text
from styleprint.stylecard import build

pytest.importorskip("en_core_web_md")

SPEECH = """So, look, here's the thing.
You know, I've been doing this for years, and honestly, people overcomplicate it.
You don't need fancy gear. You really don't.
Grab a notebook, grab a pen, and just start. Okay?
I mean, it's not rocket science.
Now, you're gonna say, but wait, I don't have time.
Yeah, I get it. Trust me.
Pick one thing. Just one. Right?"""

FORMAL = """Furthermore, one must consider that the acquisition of new habits is not a trivial undertaking for most individuals.
It is therefore essential to establish a consistent routine, and I am convinced that this approach is effective.
Moreover, one should not underestimate the importance of rest, as it is fundamental to sustained performance over time."""


@pytest.fixture(scope="module")
def target():
    utts = [Utterance(f"r{k}", i, None, None, u.text, "punct")
            for k in range(4) for i, u in enumerate(utterances_from_text(SPEECH, language="en"))]
    return analyze(utts, "en", speaker="Coach")


def test_english_features(target):
    syn, reg = target["syntax"], target["register"]
    assert {"look", "grab", "pick"} <= {v["verb"] for v in syn["imperatives"]["top_verbs"]}
    assert syn["questions"]["tag_questions"] == 8  # Okay? and Right?, in 4 recordings
    assert reg["negation"]["share_full"] == 0  # don't, it's not: all contracted
    markers = {m["marker"]: m["count"] for m in target["discourse"]["markers"]["discourse_markers"]}
    assert markers["so (initial)"] == 4 and markers["you know"] == 4 and markers["I mean"] == 4
    prefs = {p["use"]: p for p in reg["preferences"]}
    assert prefs["it's"]["use_count"] == 4 and prefs["it's"]["avoid_count"] == 0


def test_english_critic_and_card(target):
    own = critique(target, analyze(utterances_from_text(SPEECH, language="en"), "en"))
    formal = critique(target, analyze(utterances_from_text(FORMAL, language="en"), "en"))
    assert own == []
    metrics = {i.metric for i in formal}
    assert {"utterance_length", "negation", "fragments", "prefer:you"} <= metrics
    card = build(target, None, "Coach")
    assert "Say **you**, not *one (generic)*" in card and "« Look »" in card
    assert "« " in card and "furthermore" in card  # formal connectors listed under Never


def test_join_tokens():
    assert join_tokens(["that", "'s", "it"]) == "that's it"
    assert join_tokens(["do", "n't"]) == "don't"
    assert join_tokens(["l'", "huile"]) == "l'huile"


def test_punchy_speaker_is_not_asked_for_long_runs(target):
    runon = ("You know, you take your time and you look at what you want and you figure out what matters "
             "and you just go for it without overthinking any of it at all, okay? ") * 4
    metrics = {i.metric: i for i in critique(target, analyze(utterances_from_text(runon, language="en"), "en"))}
    assert "Break the long ones up" in metrics["long_runs"].feedback
    short = critique(target, analyze(utterances_from_text(SPEECH, language="en"), "en"))
    assert "long_runs" not in {i.metric for i in short}
