import pytest

from styleprint.generate import clean, critique, generate, select_passages
from styleprint.linguistic import analyze
from styleprint.segment import Utterance, utterances_from_text
from styleprint.stylecard import build

pytest.importorskip("fr_core_news_md")

SPEECH = """Bonjour à toutes et à tous.
Alors, aujourd'hui on va faire un truc simple, vous allez voir.
Voilà, on prend le temps, c'est pas compliqué.
Là, je vous montre, vous mettez un peu de sel.
Et là, voilà, c'est très bon.
Je sais que vous allez me dire que c'est long.
Mais c'est pas grave, parce que pendant ce temps, vous faites autre chose.
Voilà.
Allez, on y va, donc on continue, un peu plus vite.
Mon père faisait ça, quand j'étais petit, c'était très bon.
Voilà, c'est fini, à bientôt !"""

FORMAL = """Nous allons examiner cette question, qui constitue un enjeu majeur pour chacun d'entre nous aujourd'hui.
Cependant, cela ne signifie pas que la solution soit simple, car de nombreux facteurs interviennent en même temps.
Du coup, il convient de procéder avec méthode, c'est-à-dire étape par étape, sans précipitation excessive."""


@pytest.fixture(scope="module")
def target():
    # Repeat across "recordings" so markers count as consistent habits.
    utts = [Utterance(f"r{k}", i, None, None, u.text, "punct")
            for k in range(4) for i, u in enumerate(utterances_from_text(SPEECH, language="fr"))]
    return analyze(utts, "fr")


def test_critic_flags_formal_text_not_own_speech(target, monkeypatch):
    monkeypatch.setattr("styleprint.stylecard.MIN_WORDS_FOR_ABSENCE", 0)  # tiny test corpus
    own = critique(target, analyze(utterances_from_text(SPEECH, language="fr"), "fr"))
    formal = critique(target, analyze(utterances_from_text(FORMAL, language="fr"), "fr"))
    assert len(own) == 0
    metrics = {i.metric for i in formal}
    assert {"utterance_length", "prefer:on", "marker:du coup", "marker:c'est-à-dire"} <= metrics


def test_generate_revises_and_keeps_best(target):
    replies = iter([FORMAL, SPEECH])
    calls = []

    def chat(messages, temperature):
        calls.append(messages)
        return next(replies)

    utts = utterances_from_text(SPEECH, language="fr")
    gen = generate(chat, "m", "Chef", "fr", "# card", select_passages(utts, None, n=2, words=20), target,
                   "parle du sommeil", words=120, revisions=2)
    assert len(gen.versions) == 2  # stops once a version has no issues
    assert gen.best == 1 and gen.text == SPEECH
    assert "# card" in calls[0][0]["content"] and "Passage 1" in calls[0][0]["content"]
    assert "parle du sommeil" in calls[0][1]["content"]
    assert "Say « on », not « nous »" in calls[1][-1]["content"]


def test_select_passages_spreads_roles_and_recordings():
    utts = [Utterance(f"r{k}", i, None, None, f"mot {i} " * 10, "punct") for k in range(3) for i in range(10)]
    ps = select_passages(utts, None, n=3, words=30)
    assert [p.role for p in ps] == ["opening", "middle", "closing"]
    assert len({p.recording_id for p in ps}) == 3


def test_clean_strips_markdown_and_directions():
    assert clean("# Titre\n**Script**\n[rires]\n- Alors, voilà.\n\n\n\nOn y va.") == "Alors, voilà.\n\nOn y va."


def test_stylecard_from_profile(target):
    card = build(target, None, "Chef")
    assert "Say **on**, not *nous*" in card
    assert "voilà" in card
    assert "## Never" in card


def test_critic_flags_uniform_staccato(target):
    staccato = "\n".join(["Vous êtes prêt, voilà.", "On regarde ça ensemble.", "C'est pas compliqué du tout.",
                          "Alors, on avance un peu.", "Là, vous respirez bien."] * 6)
    metrics = {i.metric for i in critique(target, analyze(utterances_from_text(staccato, language="fr"), "fr"))}
    assert "rhythm_variation" in metrics
    assert "long_runs" not in metrics  # the test speaker rarely uses long runs, so none are demanded
