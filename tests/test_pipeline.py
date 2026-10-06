import json

import pytest

from styleprint import pipeline as pl
from styleprint.cli import main
from styleprint.corpus import Corpus, CorpusError
from test_corpus import make_audio

pytest.importorskip("fr_core_news_md")


def test_expand_sources(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.mp3", "b.MP4", "sub/c.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    out = pl.expand_sources([str(tmp_path), "https://example.com/v"])
    assert [p.split("/")[-1] for p in out] == ["a.mp3", "b.MP4", "c.wav", "v"]


def test_resolve_and_ensure_speaker(tmp_path):
    c = Corpus(tmp_path)
    logs = []
    sp = pl.ensure_speaker(c, "Jean Dupont", None, logs.append)
    assert sp.id == "jean-dupont" and sp.languages == []  # detected later, from the audio
    for name in ("jean-dupont", "Jean Dupont", "jean dupont", "JEAN DUPONT"):
        assert c.resolve(name).id == "jean-dupont"
    assert pl.ensure_speaker(c, "jean dupont", "en", logs.append).languages[0] == "en"
    with pytest.raises(CorpusError, match="known: Jean Dupont"):
        c.resolve("Personne")


@pytest.fixture
def learned(tmp_path):
    """A speaker with a fake transcript, analyzed: everything `learn` needs except Whisper and the LLM."""
    c = Corpus(tmp_path / "corpus")
    sp = c.init_speaker("Chef", languages=["fr"])
    rec = c.add(sp.id, str(make_audio(tmp_path / "a.wav")))
    words, t = [], 0.0
    for sentence in ["Alors, voilà, on y va.", "C'est pas compliqué, vous allez voir.", "Allez, ensuite on mélange un peu."] * 4:
        for w in sentence.split():
            words.append({"word": w, "start": t, "end": t + 0.2, "p": 0.9})
            t += 0.25
    tr = {"recording_id": rec.id, "source_sha256": rec.sha256, "model": "m", "segments": [{"words": words}]}
    (c.speaker_dir(sp.id) / "transcripts").mkdir()
    (c.speaker_dir(sp.id) / "transcripts" / f"{rec.id}.json").write_text(json.dumps(tr))
    pl.analyze(c, sp, lambda m: None)
    return c, sp


def test_stylecard_keeps_hand_edits(learned):
    c, sp = learned
    log = []
    card = pl.stylecard(c, sp, log.append)
    path = c.speaker_dir(sp.id) / "profile" / "stylecard.md"
    assert path.read_text() == card
    path.write_text(card + "\n- my own rule\n")
    pl.stylecard(c, sp, log.append)
    assert "my own rule" in path.read_text() and "hand edits" in log[-1]
    pl.stylecard(c, sp, log.append, force=True)
    assert "my own rule" not in path.read_text()


def test_styles_command(learned, capsys):
    c, sp = learned
    pl.stylecard(c, sp, lambda m: None)
    assert main(["--root", str(c.root), "styles"]) == 0
    out = capsys.readouterr().out
    assert "Chef" in out and "ready (no rhetoric)" in out


def test_generate_unknown_style(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "generate", "Nobody", "topic"]) == 1
    assert "no style named 'Nobody'" in capsys.readouterr().err


class FakeTranscriber:
    """Stands in for Whisper: fixed language guesses per recording, no transcription."""

    def __init__(self, guesses):
        self.guesses = guesses
        self.ran = False

    def is_current(self, speaker_id, rec):
        return False

    def detect(self, speaker_id, rec):
        return self.guesses[rec.id]

    def run(self, speaker_id):
        self.ran = True
        return iter([])


def _two_recordings(tmp_path):
    c = Corpus(tmp_path / "corpus")
    sp = c.init_speaker("Coach")
    c.add(sp.id, str(make_audio(tmp_path / "a.wav", freq=300)))
    c.add(sp.id, str(make_audio(tmp_path / "b.wav", freq=600)))
    return c, sp


def test_language_detected_and_saved(tmp_path):
    c, sp = _two_recordings(tmp_path)
    log = []
    t = FakeTranscriber({"a": [("en", 0.97), ("fr", 0.02)], "b": [("fr", 0.95), ("en", 0.04)]})
    pl.transcribe(c, sp, log.append, transcriber=t)
    assert c.speaker(sp.id).languages == ["en"]
    assert any("detected language: en" in m for m in log)
    assert any("b sounds like 'fr'" in m for m in log)  # mismatch warning
    assert t.ran


def test_unsure_detection_asks_for_language(tmp_path):
    c, sp = _two_recordings(tmp_path)
    t = FakeTranscriber({"a": [("en", 0.45), ("fr", 0.40)], "b": [("en", 0.9)]})
    with pytest.raises(CorpusError, match="pass --language"):
        pl.transcribe(c, sp, lambda m: None, transcriber=t)
    assert c.speaker(sp.id).languages == [] and not t.ran


def test_unsupported_language_is_a_clear_error(tmp_path):
    c, sp = _two_recordings(tmp_path)
    sp.languages = ["de"]
    c.update_speaker(sp)
    with pytest.raises(CorpusError, match="supports only: fr, en"):
        pl.analyze(c, sp, lambda m: None)
