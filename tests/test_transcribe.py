import json

import pytest

from styleprint.corpus import Corpus, CorpusError
from styleprint.transcribe import VERBATIM_PROMPTS, Transcriber
from test_corpus import make_audio


def fake_backend(calls):
    def backend(audio, model, language, prompt):
        calls.append((audio.name, model, language, prompt))
        return {
            "text": " Euh, bonjour. On y va.",
            "segments": [
                {"start": 0.0, "end": 1.2, "text": " Euh, bonjour.", "avg_logprob": -0.2,
                 "no_speech_prob": 0.01, "compression_ratio": 1.1,
                 "words": [{"word": " Euh,", "start": 0.0, "end": 0.4, "probability": 0.9},
                           {"word": " bonjour.", "start": 0.5, "end": 1.2, "probability": 0.99}]},
                {"start": 1.5, "end": 2.0, "text": " On y va.", "words": []},
            ],
        }
    return backend


@pytest.fixture
def corpus(tmp_path):
    c = Corpus(tmp_path / "corpus")
    c.init_speaker("Chef", languages=["fr"])
    c.add("chef", str(make_audio(tmp_path / "a.wav", freq=300)))
    c.add("chef", str(make_audio(tmp_path / "b.wav", freq=600)))
    return c


def test_transcribes_and_writes_outputs(corpus):
    calls = []
    t = Transcriber(corpus, backend=fake_backend(calls), model="m")
    results = list(t.run("chef"))

    assert [r.status for r in results] == ["done", "done"]
    assert calls[0] == ("a.wav", "m", "fr", VERBATIM_PROMPTS["fr"])
    data = json.loads(t.transcript_path("chef", "a").read_text())
    assert data["text"] == "Euh, bonjour. On y va."
    assert data["segments"][0]["words"][0] == {"word": "Euh,", "start": 0.0, "end": 0.4, "p": 0.9}
    assert data["language"] == "fr" and data["model"] == "m"
    assert t.transcript_path("chef", "a").with_suffix(".txt").read_text() == "Euh, bonjour.\nOn y va.\n"


def test_skips_current_and_redoes_on_model_change_or_force(corpus):
    calls = []
    t = Transcriber(corpus, backend=fake_backend(calls), model="m")
    list(t.run("chef"))
    assert [r.status for r in t.run("chef")] == ["skipped", "skipped"]
    assert [r.status for r in t.run("chef", ["b"], force=True)] == ["done"]
    t2 = Transcriber(corpus, backend=fake_backend(calls), model="other")
    assert [r.status for r in t2.run("chef")] == ["done", "done"]
    assert len(calls) == 5


def test_unknown_recording(corpus):
    with pytest.raises(CorpusError, match="unknown"):
        list(Transcriber(corpus, backend=fake_backend([])).run("chef", ["nope"]))
