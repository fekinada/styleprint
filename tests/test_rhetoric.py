import json

from styleprint.llm import LLMConfig
from styleprint.rhetoric import RhetoricAnnotator, aggregate, chunk_utterances, quote_in, render, verify
from styleprint.segment import Utterance


def utts(rid="r", n=6):
    texts = ["Bonjour à tous.", "Pourquoi un couscous ?", "Parce que vous me l'avez demandé.",
             "Quand j'étais enfant, mon père faisait un couscous extraordinaire.", "Allez, on y va.", "Voilà."]
    return [Utterance(rid, i, float(i), float(i) + 1, texts[i % len(texts)], "punct") for i in range(n)]


def fake_llm(calls):
    def llm(prompt, schema):
        calls.append(prompt)
        return {"items": [
            {"device": "rhetorical_question", "utterances": [1, 2], "quote": "Pourquoi un couscous ?",
             "why": "asks and answers"},
            {"device": "anecdote", "utterances": [3], "quote": "quand j'étais enfant, mon père faisait",
             "why": "childhood story"},
            {"device": "analogy", "utterances": [4], "quote": "comme un chef étoilé", "why": "invented"},
            {"device": "made_up", "utterances": [0], "quote": "Bonjour", "why": "bad device"},
        ]}
    return llm


def test_chunking_keeps_recordings_apart_and_adds_context():
    u = utts("a", 6) + utts("b", 6)
    chunks = chunk_utterances(u, max_words=10)
    assert {c.recording_id for c in chunks} == {"a", "b"}
    assert all(len({x.recording_id for x in c.utterances}) == 1 for c in chunks)
    assert sum(len(c.utterances) for c in chunks) == 12
    assert chunks[0].context == [] and chunks[1].context
    assert [c.number for c in chunks if c.recording_id == "b"][0] == 0


def test_verify_rejects_quotes_not_in_passage():
    chunk = chunk_utterances(utts())[0]
    kept, rejected = verify(fake_llm([])("", {})["items"], chunk)
    assert [k["device"] for k in kept] == ["rhetorical_question", "anecdote"]
    assert len(rejected) == 2
    assert kept[0]["utterances"] == [1, 2] and kept[0]["t"] == 1.0
    assert kept[1]["utterances"] == [3]


def test_annotator_caches_and_aggregates(tmp_path):
    calls = []
    a = RhetoricAnnotator(fake_llm(calls), tmp_path, "m", False)
    first = list(a.run(utts(), "fr"))
    second = list(a.run(utts(), "fr"))
    assert len(calls) == 1 and second[0].cached
    assert "PASSAGE:\n[0] Bonjour à tous." in calls[0]

    agg = aggregate(second, 100, "m")
    assert agg["annotations"] == 2 and agg["rejected_unverifiable"] == 2
    assert {d["device"]: d["per_1k"] for d in agg["devices"]} == {"rhetorical_question": 10.0, "anecdote": 10.0}
    assert "analogy" in agg["absent"]
    assert "Pourquoi un couscous ?" in render(agg, "Chef")


def test_config_precedence(tmp_path, monkeypatch):
    cfg = tmp_path / "styleprint.toml"
    cfg.write_text('[llm]\nurl = "http://a:1"\nmodel = "x"\nthinking = true\n')
    monkeypatch.setenv("STYLEPRINT_LLM_MODEL", "y")
    c = LLMConfig.load(cfg, url="http://b:2", thinking=None)
    assert (c.url, c.model, c.thinking) == ("http://b:2", "y", True)


def test_quote_with_ellipsis():
    passage = "j'ai eu la chance d'aller au bout du monde à tijuana"
    assert quote_in("J'ai eu la chance d'aller... à Tijuana", passage)
    assert quote_in("la chance … tijuana", passage)
    assert not quote_in("à Tijuana… j'ai eu la chance", passage)  # wrong order
    assert not quote_in("...", passage)
