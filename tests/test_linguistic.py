import pytest

from styleprint.linguistic import analyze, mattr
from styleprint.segment import utterances_from_text

pytest.importorskip("fr_core_news_md")

TEXT = """Bonjour à toutes et à tous.
Alors, aujourd'hui on va faire une soupe, d'accord ?
Regardez, je coupe les oignons, voilà.
Bon, c'est pas compliqué.
Ce n'est pas difficile non plus.
Coupez les carottes en dés.
La sauce est faite par le chef.
Je suis tombé devant l'hôtel.
Vous vous êtes lavé les mains ?
C'est très très bon, vraiment.
Voilà, on va faire la soupe, voilà."""


@pytest.fixture(scope="module")
def profile():
    return analyze(utterances_from_text(TEXT, language="fr"), "fr", speaker="test")


def test_corpus_counts(profile):
    assert profile["corpus"]["utterances"] == 11
    assert profile["lexical"]["words"] > 50


def test_register(profile):
    reg = profile["register"]
    assert reg["negation"]["negations"] == 2
    assert reg["negation"]["share_full"] == 0.5
    on = next(p for p in reg["preferences"] if p["use"] == "on")
    assert (on["use_count"], on["avoid_count"]) == (2, 0)


def test_imperatives_and_passive(profile):
    verbs = {v["verb"] for v in profile["syntax"]["imperatives"]["top_verbs"]}
    assert {"regardez", "coupez"} <= verbs
    passive = [e["text"] for e in profile["syntax"]["passive"]["examples"]]
    assert passive == ["La sauce est faite par le chef."]


def test_questions_and_tags(profile):
    q = profile["syntax"]["questions"]
    assert q["tag_questions"] == 1
    assert [e["text"] for e in q["other_examples"]] == ["Vous vous êtes lavé les mains ?"]


def test_markers(profile):
    markers = {r["marker"]: r["count"] for r in profile["discourse"]["markers"]["discourse_markers"]}
    assert markers["voilà"] == 3
    assert markers["bon"] == 1
    assert profile["discourse"]["emphatic_repetition"]["words"] == {"très": 1}
    assert profile["discourse"]["self_correction"]["immediate_repeats_per_1k"] == 0


def test_fragments(profile):
    frags = [e["text"] for e in profile["syntax"]["fragments"]["examples"]]
    assert "Bonjour à toutes et à tous." in frags
    assert "Coupez les carottes en dés." not in frags


def test_mattr():
    assert mattr(["a", "b", "a", "b"], window=2) == 1.0
    assert mattr(["a", "a", "a"], window=2) == 0.5
