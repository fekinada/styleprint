from styleprint.segment import utterances_from_text, utterances_from_transcript


def W(word, start, end):
    return {"word": word, "start": start, "end": end, "p": 0.9}


def transcript(words, **seg):
    return {"recording_id": "r", "language": "fr", "segments": [{"words": words, **seg}]}


def test_splits_on_punctuation_and_pauses():
    t = transcript([W("Bonjour.", 0, 0.5), W("On", 0.6, 0.7), W("y", 0.7, 0.8), W("va", 0.8, 1.0),
                    W("donc", 2.0, 2.2), W("c", 2.3, 2.4), W("'est", 2.4, 2.6), W("parti", 2.6, 2.9), W("!", 2.9, 3.0)])
    u = utterances_from_transcript(t)
    assert [(x.text, x.split) for x in u] == [
        ("Bonjour.", "punct"), ("On y va", "pause"), ("donc c'est parti !", "punct")]
    assert (u[1].start, u[1].end) == (0.6, 1.0)


def test_long_runs_split_at_shorter_pauses():
    words = [W(f"mot{i}", i * 0.3, i * 0.3 + 0.05) for i in range(30)]  # 0.25 s gaps, no punctuation
    lengths = [len(x.text.split()) for x in utterances_from_transcript(transcript(words))]
    assert lengths == [20, 10]


def test_drops_hallucinated_segments():
    t = transcript([W("Sous-titres", 0, 1)], no_speech_prob=0.95, avg_logprob=-1.5)
    assert utterances_from_transcript(t) == []


def test_from_text():
    u = utterances_from_text("Alors, on y va. Vous voyez ?\nVoilà", language="fr")
    assert [x.text for x in u] == ["Alors, on y va.", "Vous voyez ?", "Voilà"]


def test_english_spacing():
    t = {"recording_id": "r", "language": "en",
         "segments": [{"words": [W("So", 0, 0.2), W("you", 0.2, 0.4), W("know", 0.4, 0.6), W("?", 0.6, 0.7),
                                 W("It", 0.8, 0.9), W("doesn", 0.9, 1.0), W("'t", 1.0, 1.1), W("matter", 1.1, 1.3), W("!", 1.3, 1.4)]}]}
    assert [x.text for x in utterances_from_transcript(t)] == ["So you know?", "It doesn't matter!"]
