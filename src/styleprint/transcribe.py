"""Speech-to-text: turn each recording's normalized audio into a timestamped transcript.

Output per recording, next to the corpus audio:

    corpus/<speaker_id>/transcripts/<recording_id>.json   segments + word timestamps
    corpus/<speaker_id>/transcripts/<recording_id>.txt    plain text, one segment per line

The JSON is the raw speech-to-text output. Cleaning (removing hallucinations,
normalizing artifacts) is a separate, later step that reads it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from styleprint.corpus import Corpus, CorpusError, Recording

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Whisper tends to tidy speech up and drop disfluencies, which are part of a
# speaker's style. A prompt written in verbatim register nudges it to keep them.
VERBATIM_PROMPTS = {
    "fr": "Euh, alors, bon... Voilà, hein. Donc euh, on y va, quoi.",
    "en": "Um, so, uh... you know, like, I mean, yeah.",
}

# backend(audio_path, model, language, prompt) -> Whisper-style result dict
Backend = Callable[[Path, str, str, str | None], dict]
# detector(audio_path, model) -> {language code: probability}
Detector = Callable[[Path, str], dict[str, float]]


def mlx_detector(audio: Path, model: str) -> dict[str, float]:
    """Whisper's language identification on two 30 s windows (start and middle), averaged.

    The middle window guards against intros with music or a jingle.
    """
    try:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder
    except ImportError:
        raise CorpusError("mlx-whisper is not installed: run `uv sync --extra transcribe`") from None
    m = ModelHolder.get_model(model, mx.float16)
    mel = log_mel_spectrogram(str(audio), n_mels=m.dims.n_mels, padding=N_SAMPLES)
    content = mel.shape[-2] - N_FRAMES
    starts = sorted({0, max(0, content // 2 - N_FRAMES // 2)})
    total: dict[str, float] = {}
    for start in starts:
        window = pad_or_trim(mel[start:start + N_FRAMES], N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = m.detect_language(window)
        for lang, prob in probs.items():
            total[lang] = total.get(lang, 0.0) + float(prob) / len(starts)
    return total


def mlx_backend(audio: Path, model: str, language: str, prompt: str | None) -> dict:
    try:
        import mlx_whisper
    except ImportError:
        raise CorpusError("mlx-whisper is not installed: run `uv sync --extra transcribe`") from None
    return mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=model,
        language=language,
        initial_prompt=prompt,
        word_timestamps=True,
        # Prevents one bad segment from propagating; also limits repetition loops.
        condition_on_previous_text=False,
        # Skip text Whisper invents over long silences or music.
        hallucination_silence_threshold=2.0,
    )


@dataclass
class Result:
    recording: Recording
    status: str  # "done" or "skipped"
    path: Path


def _round(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


def to_transcript(raw: dict, rec: Recording, model: str, language: str, prompt: str | None) -> dict:
    segments = []
    for s in raw.get("segments", []):
        segments.append({
            "start": _round(s["start"]),
            "end": _round(s["end"]),
            "text": s["text"].strip(),
            "avg_logprob": _round(s.get("avg_logprob")),
            "no_speech_prob": _round(s.get("no_speech_prob")),
            "compression_ratio": _round(s.get("compression_ratio")),
            "words": [
                {"word": w["word"].strip(), "start": _round(w["start"]), "end": _round(w["end"]),
                 "p": _round(w.get("probability"))}
                for w in s.get("words", [])
            ],
        })
    return {
        "recording_id": rec.id,
        "source_sha256": rec.sha256,
        "model": model,
        "language": language,
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": rec.duration_s,
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }


class Transcriber:
    def __init__(self, corpus: Corpus, backend: Backend = mlx_backend, model: str = DEFAULT_MODEL,
                 detector: Detector = mlx_detector):
        self.corpus = corpus
        self.backend = backend
        self.model = model
        self.detector = detector

    def detect(self, speaker_id: str, rec: Recording) -> list[tuple[str, float]]:
        """Most likely spoken languages of a recording, best first."""
        probs = self.detector(self.corpus.speaker_dir(speaker_id) / rec.audio_path, self.model)
        return sorted(probs.items(), key=lambda kv: -kv[1])

    def transcript_path(self, speaker_id: str, recording_id: str) -> Path:
        return self.corpus.speaker_dir(speaker_id) / "transcripts" / f"{recording_id}.json"

    def is_current(self, speaker_id: str, rec: Recording) -> bool:
        path = self.transcript_path(speaker_id, rec.id)
        if not path.exists():
            return False
        meta = json.loads(path.read_text())
        return meta.get("source_sha256") == rec.sha256 and meta.get("model") == self.model

    def run(self, speaker_id: str, recording_ids: list[str] | None = None, *,
            force: bool = False, prompt: str | None = None) -> Iterator[Result]:
        """Transcribe recordings, skipping ones already done with the same audio and model."""
        speaker = self.corpus.speaker(speaker_id)
        if not speaker.languages:
            raise CorpusError(f"no language set for '{speaker.name}': detect it first or pass --language")
        language = speaker.languages[0]
        if prompt is None:
            prompt = VERBATIM_PROMPTS.get(language)

        recs = self.corpus.recordings(speaker_id)
        if recording_ids:
            unknown = set(recording_ids) - {r.id for r in recs}
            if unknown:
                raise CorpusError(f"unknown recording(s): {', '.join(sorted(unknown))}")
            recs = [r for r in recs if r.id in recording_ids]

        d = self.corpus.speaker_dir(speaker_id)
        for rec in recs:
            out = self.transcript_path(speaker_id, rec.id)
            if not force and self.is_current(speaker_id, rec):
                yield Result(rec, "skipped", out)
                continue
            raw = self.backend(d / rec.audio_path, self.model, language, prompt)
            transcript = to_transcript(raw, rec, self.model, language, prompt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(transcript, ensure_ascii=False, indent=1) + "\n")
            out.with_suffix(".txt").write_text("\n".join(s["text"] for s in transcript["segments"]) + "\n")
            yield Result(rec, "done", out)
