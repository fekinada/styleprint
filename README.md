# Styleprint

**Styleprint** is an experimental system for learning a person's communication style from speech and using that style to generate new text.

The system takes recordings of a target speaker, transcribes and analyzes them, and builds a structured representation of their communication habits. This representation captures characteristics such as vocabulary, sentence structure, rhythm, rhetorical patterns, level of formality, use of repetition, argument structure, and conversational tendencies.

Given a new topic or instruction, Styleprint uses the learned style representation—together with representative examples from the source corpus—to generate original text that reflects the target speaker's communication style while expressing new content.

The project deliberately focuses on **text generation**. Audio is used only as the source from which communication style is learned; voice synthesis and speech generation are outside the scope of the initial system.

## Pipeline

```
REFERENCE AUDIO                       styleprint corpus
       │
Speech-to-text → cleaned transcript   styleprint transcribe
       │
       ├── linguistic analysis       styleprint analyze
       │   (vocabulary, sentence length, syntax,
       │   discourse markers, rhetorical devices, structure)
       └── audio analysis (speech rate, pauses, emphasis, pitch)
       │
STYLE PROFILE → style examples / memory
       │
NEW USER INSTRUCTION → style-conditioned generation → GENERATED TEXT
```

## Setup

Requires Python ≥ 3.11 and `ffmpeg` (on `PATH`, or set `FFMPEG_BINARY`).

```sh
uv sync                     # core
uv sync --extra download    # adds yt-dlp, for ingesting URLs
uv sync --all-extras        # + mlx-whisper (transcribe) and spaCy/wordfreq (analyze)
cp styleprint.example.toml styleprint.toml   # then choose OpenAI or a local server
```

### LLM

The rhetoric step and generation use an LLM through the OpenAI chat API. That can be OpenAI itself, or any compatible server (llama.cpp, vLLM, Ollama, LM Studio). Configure it in `styleprint.toml`; the file is git-ignored, and `styleprint.example.toml` is the template.

**OpenAI**

```toml
[llm]
url = "https://api.openai.com"
model = "gpt-4.1"
api_key_env = "OPENAI_API_KEY"   # the env var's name, not the key
```
```sh
export OPENAI_API_KEY=sk-...
```

**Local or self-hosted server**

```toml
[llm]
url = "http://localhost:8080"
model = ""          # empty: whatever the server serves
thinking = false    # Qwen3-style reasoning: better judgments, much slower
```

The server type is detected from the URL; set `kind = "openai"` or `kind = "local"` to override, for example for a proxy. Requests are adapted to each: OpenAI gets `max_completion_tokens` and no local-only options. If a model rejects a parameter (reasoning models refuse a custom `temperature`), it is dropped and the request is retried. The API key is read from the named environment variable (or `STYLEPRINT_LLM_API_KEY`), and only sent to the server it's configured for: an exported `OPENAI_API_KEY` never goes to a local server.

Overrides: `--url`, `--model` and `--thinking` on the commands, or the `STYLEPRINT_LLM_URL` and `STYLEPRINT_LLM_MODEL` environment variables.

## Quick start

```sh
# 1. Learn a style from recordings (files, folders or URLs). Re-run it after adding audio.
uv run styleprint learn "Harry Potter" inbox/

# 2. Write a spoken script in that style.
uv run styleprint generate "Harry Potter" "Explique comment réussir ses examens"

# 3. See which styles exist.
uv run styleprint styles
```

`learn` runs every step: it creates the style if it's new, imports the recordings, transcribes them, analyzes the transcripts, annotates rhetoric with the LLM, and builds the style card. Each step skips work that's already done. Re-running with nothing new takes seconds, and adding one recording only processes that recording. If the LLM is unreachable, the rhetoric step is skipped with a warning and the style still works; run `learn` again later to add rhetoric. If you edit the style card by hand, `learn` keeps your edits (`--rebuild-card` overwrites them).

### What `learn` accepts

| Input | Example | Notes |
|---|---|---|
| Audio files | `.mp3` `.wav` `.m4a` `.aac` `.flac` `.ogg` `.opus` `.wma` `.aiff` `.aif` `.caf` `.amr` | Phone voice memos work as they are |
| Video files | `.mp4` `.mov` `.mkv` `.webm` `.avi` `.m4v` `.3gp` | Only the audio track is used |
| Folders | `inbox/` | Scanned recursively for the extensions above; other files are ignored |
| URLs | `"https://www.youtube.com/watch?v=..."` | Any site yt-dlp supports (YouTube, Vimeo, podcasts…); needs `uv sync --extra download`. One video per URL: playlists aren't expanded. The recording is named after the video's title |

A file you name explicitly can be in any other format ffmpeg reads; only folder scanning is limited to the extensions above. Every input is converted to 16 kHz mono WAV for transcription, and the original is kept as-is. Exact duplicates are skipped.

What to feed it:

- **One speaker per style.** If other people talk in a recording, add `--multi-speaker`. Their words would otherwise count as the speaker's style; separating speakers isn't automated yet.
- **Spontaneous speech** (interviews, vlogs, talks without notes) carries the most style. Read or scripted text mostly shows how someone writes.
- **Length:** 10–15 minutes is enough to start; 30–60 minutes makes rarer habits reliable. Many short clips beat one long one.
- **Clean audio:** little music or crosstalk.

**Languages.** French and English are supported. The language is detected the first time a style is learned: Whisper identifies the language of the first recording (sampling its start and middle) and saves it with the style. If the detection is unsure, `learn` stops and asks for `--language`. Each recording added later is checked too, with a warning if it sounds like a different language. Generated text is always in the style's language, whatever language the instruction is written in. A style in another language can be transcribed but not analyzed yet.

Styles can be referred to by name in any spelling (`"Harry Potter"`, `harry-potter`) or by id.

| `learn` option | |
|---|---|
| `--language en` | language of the recordings; by default it's detected from the audio |
| `--genre`, `--register`, `--multi-speaker` | metadata for the recordings being imported (see below) |
| `--no-rhetoric` | skip the LLM step |
| `--url`, `--model`, `--thinking` | override `styleprint.toml` |

## How it works

The sections below describe each step `learn` runs. Each step is also available as its own command, for debugging and finer control.

## Reference corpus

Each target speaker gets a directory under `corpus/`:

```
corpus/<speaker_id>/
  speaker.json     who the speaker is
  manifest.jsonl   one line of metadata per recording (source of truth)
  raw/             originals, byte-for-byte
  audio/           normalized 16 kHz mono WAV, ready for speech-to-text
```

```sh
styleprint corpus init "Jane Doe" --languages fr --description "Host of the Example podcast"

styleprint corpus add jane-doe talks/*.mp3 \
    --genre lecture --register prepared

styleprint corpus add jane-doe "https://www.youtube.com/watch?v=..." \
    --title "Panel on X" --genre debate --register spontaneous \
    --multi-speaker --date 2024-05-02

styleprint corpus list jane-doe
styleprint corpus stats jane-doe       # coverage by genre and register
styleprint corpus validate jane-doe    # manifest vs. files on disk
```

Each recording is tagged with:

- **genre**: `speech`, `lecture`, `interview`, `podcast`, `conversation`, `debate`, `monologue`, `other`
- **register**: `scripted`, `prepared`, or `spontaneous`. Spontaneous speech carries the most personal style. Scripted speech may reflect a speechwriter more than the speaker.
- **multi_speaker**: whether other voices are present. Those recordings need diarization before analysis, so that only the target speaker's turns are used.

Exact duplicates are rejected using a content hash. The whole `corpus/` folder is git-ignored: it holds recordings, transcripts and profiles of real people, often under copyright.

### What makes a good corpus

- Aim for at least an hour of the target speaker's own speech, and more if possible. Breadth matters more than total length.
- Mix genres and registers. Weight it toward spontaneous speech.
- Prefer clean audio with little crosstalk or music.
- Note the dates. Style drifts over the years, so a narrow time window gives a more consistent profile.

## Transcription

```sh
styleprint transcribe jane-doe            # all recordings; skips ones already done
```

Runs Whisper (`whisper-large-v3-turbo`) locally on Apple Silicon via `mlx-whisper`, in the speaker's first language. It keeps word timestamps and uses a verbatim prompt so fillers are less likely to be removed. Output goes to `corpus/<speaker>/transcripts/<id>.json` (segments + words) and `.txt`.

## Linguistic profile

```sh
styleprint analyze jane-doe
```

Splits transcripts into utterances (at final punctuation or pauses over 0.5 s; see `utterances.jsonl`). It then writes `profile/linguistic.json` and a readable `profile/linguistic.md`:

- **Lexical:** lexical diversity (MATTR), pronouns, frequent lemmas, and words used unusually often or rarely compared with general usage (`wordfreq`). These are split into style words and topic words.
- **Syntax:** utterance length, clauses, fragments, openings, questions and check-in tags, imperatives, passive voice.
- **Discourse:** markers, transitions, hedging, intensifiers, contrast, audience framing, repairs, evaluative expressions, repetition, and signature phrases that recur across recordings.
- **Register:** informal negation (*ne* dropping, *n't* contractions), preferred forms, how much the speaker addresses the listener, familiar vocabulary.

Every feature includes verbatim examples. Language-specific resources live in `lexicons.py`, for French and English. They cover markers, familiar words, how informal negation looks (French drops *ne*, English contracts *n't*), preferred forms (*on/nous*, *ça/cela*, *vous/tu*; *yeah/yes*, *gonna/going to*, *it's/it is*), command detection and passive detection. Adding a language means adding one entry there and its spaCy model. The same `analyze()` runs on any text via `segment.utterances_from_text`, so generated text can be compared with the profile.

## Rhetorical profile

```sh
styleprint rhetoric jane-doe      # needs `styleprint analyze` first
```

An LLM annotates the utterances in passages of about 350 words. It looks for rhetorical questions, contrast, analogy, anecdote, emphatic repetition, lists, reframing, provocative openings, counterarguments, humor and appeals to authority. Each annotation must quote the transcript; quotes not found in the passage are discarded. Results go to `profile/rhetoric.json` and `profile/rhetoric.md`. Each passage is cached in `profile/rhetoric_cache/`, so an interrupted run resumes where it stopped.

The model is configured in `styleprint.toml` (see [LLM](#llm)).

## Generation

```sh
styleprint stylecard jane-doe                     # optional: build/inspect the style card first
styleprint generate jane-doe "explique comment bien dormir" --words 250
styleprint compare jane-doe some-text.txt         # score any text against the profile
```

The output is a **spoken script**: what the speaker would say out loud. Generation has three parts:

1. **Style card** (`profile/stylecard.md`): the measured profile rewritten as short rules. It covers how they address the listener, rhythm, the words that carry their voice (with rough rates), words they never use, and their rhetorical habits. Topic vocabulary and channel rituals are left out. The card is built once and then used as-is, so you can edit it by hand. Rebuild it with `stylecard --force`.
2. **Example passages:** four real passages chosen by what they do (an opening, a closing, and passages rich in rhetorical habits), spread across recordings. The model is told to copy *how* they talk, not *what* they talk about.
3. **Critic:** the draft goes through the same `analyze` code and is compared with the profile. It checks utterance length, fragments, *ne* dropping, *on/nous*, *ça/cela*, *vous/tu*, markers the speaker never uses, and core markers that are missing or overused. Issues become feedback for a revision (`--revisions`, default 1), and the version with the fewest issues is kept.

The critic is calibrated so that the speaker's own transcripts score 0. Each run is saved to `generations/` as `.txt` and as `.json` (with the full prompt, every version and its issues).

