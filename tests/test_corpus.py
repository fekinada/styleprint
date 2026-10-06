import subprocess
import wave
from pathlib import Path

import pytest

from styleprint.cli import main
from styleprint.corpus import SAMPLE_RATE, Corpus, CorpusError, ffmpeg_binary, slugify


def make_audio(path: Path, seconds: float = 1.0, freq: int = 440, rate: int = 44_100, channels: int = 2) -> Path:
    subprocess.run(
        [ffmpeg_binary(), "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-ar", str(rate), "-ac", str(channels), str(path)],
        check=True,
    )
    return path


@pytest.fixture
def corpus(tmp_path):
    c = Corpus(tmp_path / "corpus")
    c.init_speaker("Ada Lovelace", description="test speaker")
    return c


def test_init_creates_layout(corpus):
    d = corpus.speaker_dir("ada-lovelace")
    assert (d / "speaker.json").exists()
    assert (d / "manifest.jsonl").exists()
    assert (d / "raw").is_dir() and (d / "audio").is_dir()
    with pytest.raises(CorpusError):
        corpus.init_speaker("Ada Lovelace")


def test_add_normalizes_and_records(corpus, tmp_path):
    src = make_audio(tmp_path / "Talk One.mp3", seconds=2)
    rec = corpus.add("ada-lovelace", str(src), genre="lecture", register="prepared", rights="own recording")

    assert rec.id == "talk-one"
    assert rec.duration_s == pytest.approx(2, abs=0.1)
    d = corpus.speaker_dir("ada-lovelace")
    assert (d / rec.raw_path).read_bytes() == src.read_bytes()
    with wave.open(str(d / rec.audio_path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)

    assert [r.id for r in corpus.recordings("ada-lovelace")] == ["talk-one"]
    assert corpus.validate("ada-lovelace", check_hashes=True) == []


def test_duplicates_and_id_collisions(corpus, tmp_path):
    a = make_audio(tmp_path / "a.wav", freq=440)
    corpus.add("ada-lovelace", str(a), title="Same", rights="x")
    with pytest.raises(CorpusError, match="already in corpus"):
        corpus.add("ada-lovelace", str(a), rights="x")

    b = make_audio(tmp_path / "b.wav", freq=880)
    rec = corpus.add("ada-lovelace", str(b), title="Same", rights="x")
    assert rec.id == "same-2"


def test_rejects_bad_input(corpus, tmp_path):
    with pytest.raises(CorpusError, match="no such file"):
        corpus.add("ada-lovelace", str(tmp_path / "missing.wav"))
    with pytest.raises(CorpusError, match="genre"):
        corpus.add("ada-lovelace", "x", genre="rap")
    junk = tmp_path / "junk.mp3"
    junk.write_text("not audio")
    with pytest.raises(CorpusError, match="ffmpeg failed"):
        corpus.add("ada-lovelace", str(junk))
    assert corpus.recordings("ada-lovelace") == []
    assert list((corpus.speaker_dir("ada-lovelace") / "audio").iterdir()) == []


def test_remove_and_validate(corpus, tmp_path):
    rec = corpus.add("ada-lovelace", str(make_audio(tmp_path / "a.wav")))
    assert corpus.validate("ada-lovelace") == []

    (corpus.speaker_dir("ada-lovelace") / rec.audio_path).unlink()
    assert any("missing normalized audio" in p for p in corpus.validate("ada-lovelace"))

    corpus.remove("ada-lovelace", "a")
    assert corpus.recordings("ada-lovelace") == []
    assert corpus.validate("ada-lovelace") == []


def test_stats(corpus, tmp_path):
    corpus.add("ada-lovelace", str(make_audio(tmp_path / "a.wav", 1, 440)), genre="speech", register="scripted")
    corpus.add("ada-lovelace", str(make_audio(tmp_path / "b.wav", 2, 880)), genre="interview", multi_speaker=True)
    st = corpus.stats("ada-lovelace")
    assert st["recordings"] == 2 and st["multi_speaker"] == 1
    assert st["total_s"] == pytest.approx(3, abs=0.2)
    assert set(st["by_register"]) == {"scripted", "spontaneous"}


def test_cli_roundtrip(tmp_path, capsys):
    root = str(tmp_path / "corpus")
    assert main(["--root", root, "corpus", "init", "Grace Hopper"]) == 0
    a, b = make_audio(tmp_path / "a.wav", freq=300), make_audio(tmp_path / "b.wav", freq=600)
    assert main(["--root", root, "corpus", "add", "grace-hopper", str(a), str(b),
                 "--genre", "interview", "--rights", "own recording"]) == 0
    assert main(["--root", root, "corpus", "validate", "grace-hopper"]) == 0
    assert main(["--root", root, "corpus", "stats", "grace-hopper"]) == 0
    out = capsys.readouterr().out
    assert "+ a" in out and "+ b" in out and "2 recordings" in out
    assert main(["--root", root, "corpus", "list", "nobody"]) == 1


def test_slugify_keeps_accented_letters():
    assert slugify("🥬 La vraie salade César  le secret") == "la-vraie-salade-cesar-le-secret"
    assert slugify("Poulet rôti, carotte & Patate douce") == "poulet-roti-carotte-patate-douce"
    assert slugify("🍲") == "recording"
