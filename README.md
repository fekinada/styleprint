# Styleprint

**Styleprint** is an experimental system for learning a person's communication style from speech and using that style to generate new text.

The system takes recordings of a target speaker, transcribes and analyzes them, and builds a structured representation of their communication habits. This representation captures characteristics such as vocabulary, sentence structure, rhythm, rhetorical patterns, level of formality, use of repetition, argument structure, and conversational tendencies.

Given a new topic or instruction, Styleprint uses the learned style representation—together with representative examples from the source corpus—to generate original text that reflects the target speaker's communication style while expressing new content.

The project deliberately focuses on **text generation**. Audio is used only as the source from which communication style is learned; voice synthesis and speech generation are outside the scope of the initial system.

## Pipeline

```
REFERENCE AUDIO (files, folders, URLs)
       │
Speech-to-text (Whisper, language detected) → utterances
       │
       ├── linguistic analysis: vocabulary, rhythm, syntax,
       │   discourse markers, register
       ├── rhetorical analysis (LLM): anecdotes, objections,
       │   reframing, questions…
       └── audio analysis: speech rate, pauses, emphasis,
           pitch                                   (not implemented yet)
       │
STYLE PROFILE → style card + real example passages
       │
NEW INSTRUCTION → generation → critic → revision → GENERATED TEXT
```

`styleprint learn` runs everything up to the style profile; `styleprint generate` runs the rest.

Audio is used only for transcription today. Word timestamps are kept, and pauses split speech into utterances, but prosody (speech rate, pause patterns, emphasis, pitch) isn't measured or used for generation yet.

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

Measures *how* the speaker talks, with deterministic code: no LLM, so the same transcripts always give the same profile. It writes `profile/linguistic.json` (for the generator and the critic) and a readable `profile/linguistic.md`.

### Utterances, not sentences

Speech has no reliable sentences, and Whisper's punctuation is inconsistent. Transcripts are therefore split into **utterances**: a stretch of speech that ends at final punctuation (`.` `?` `!`) or at a pause longer than 0.5 s, measured with the word timestamps. In a long stretch with no punctuation, a shorter pause (0.2 s) also ends the utterance once it reaches 20 words. Utterances are saved in `utterances.jsonl`, and all syntax and discourse measures use them.

### How to read the numbers

- **Rates are per 1,000 words**, so styles learned from different amounts of audio can be compared.
- **Compared with general usage.** Word rates are compared with the language's everyday frequency (from `wordfreq`). The profile shows what is *distinctive*, e.g. "*voilà* ×12", not just what is frequent: everyone says *le* and *the* a lot.
- **Consistency across recordings.** Each marker records how many recordings it appears in. A habit found in every recording is style; one found in a single recording may come from that day's topic.
- **Style vs. topic.** Distinctive words are split into function and discourse words (style, portable to any topic), adjectives (partly evaluation, partly topic) and content words (mostly topic, never used as style).
- **Examples everywhere.** Every feature keeps a few verbatim utterances, spread across recordings, so each number can be checked against real speech.
- **Absence.** Words the speaker never uses ("never says *du coup*") are only reported above about 2,000 words of speech. Below that, "never" just means "not seen yet".

### What is measured

| Group | Feature | What it captures |
|---|---|---|
| **Lexical** | Lexical diversity (MATTR) | How varied the vocabulary is, independent of corpus size |
| | Pronouns | *I / you / we / one…* vs. general usage: talks about themself? addresses the listener? |
| | Distinctive words | Words used far more (or less) than usual, split into style / adjectives / topic |
| | Frequent lemmas | Favourite verbs, nouns, adjectives, adverbs |
| **Syntax** | Utterance length | Median, spread, share of short (≤ 7 words) and long (≥ 13 words) utterances, and variation: punchy, flowing, or both |
| | Clauses and subordination | How much is packed into one utterance |
| | Fragments | Utterances with no verb: reactions, afterthoughts ("Wow.", "Ten minutes.", "Just one.") |
| | Openings | Most common first words and word pairs ("so I", "alors on", "you know") |
| | Questions | Share of questions, and check-in tags ("right?", "d'accord ?") |
| | Commands | Imperatives addressed to the listener ("look", "regardez") |
| | Passive voice | Approximate, from the parser |
| **Discourse** | Markers | Words that glue speech together: *so, well, you know, I mean / voilà, bon, alors, du coup…* |
| | Transitions, hedging, intensifiers, contrast | *then / ensuite*, *kind of / un peu*, *really / vraiment*, *but / mais*… |
| | Audience framing | Talking to the listener: *you'll see, let me show you / vous allez voir, on va…* |
| | Evaluative words | *amazing, perfect / magnifique, génial…* |
| | Repetition | Emphatic doubling ("very very", "très très") vs. hesitation repeats |
| | Signature phrases | 3–5 word sequences that recur across several recordings |
| **Register** | Informal negation | French: dropping *ne* ("c'est pas"); English: contractions ("don't", "it's not") |
| | Preferred forms | Strong preferences: *on/nous*, *ça/cela*, *vous/tu*; *yeah/yes*, *gonna/going to*, *it's/it is*, *you/one* |
| | Addressing the listener | How often they speak *to* someone |
| | Familiar vocabulary | *truc, super, ouais / stuff, gonna, yeah…* |

### Languages

Language-specific resources live in `lexicons.py`, for French and English: marker lists, familiar words, how informal negation looks, preferred-form pairs, and the rules for spotting commands and passive voice. Commands and passive voice need custom rules because spaCy's built-in tags miss most of them in transcribed speech. Adding a language means adding one entry there and installing its spaCy model.

The same `analyze()` runs on any text (via `segment.utterances_from_text`). That's how the critic compares generated text with the speaker, and how `styleprint compare` scores any text file.

**Limits.** Whisper removes most hesitations ("euh", "um"), so self-corrections are undercounted. Passive voice and commands are rule-based approximations; check the examples. Small corpora (under ~5,000 words) make rare features anecdotal; the report says how many examples each finding rests on.

## Rhetorical profile

```sh
styleprint rhetoric jane-doe      # needs `styleprint analyze` first
```

Captures *how the speaker persuades and engages*, which takes judgment rather than counting. An LLM reads the transcript and marks rhetorical devices, each backed by a verbatim quote.

### Devices

| Device | What the model looks for | Illustration |
|---|---|---|
| **Rhetorical question** | A question the speaker doesn't expect answered, or answers straight away | "Why does this matter? Because…" |
| **Counterargument** | Voicing the listener's objection before they do, then answering it | "You're going to say you don't have time. I get it, but…" |
| **Anecdote** | A personal memory or the story behind something | "When I was twenty, I thought…" |
| **Reframing** | Correcting the usual idea: "not X, but Y", "the real X" | "Motivation isn't the point. Habits are." |
| **Contrast** | Two options or situations set side by side | "Two schools: butter or olive oil." |
| **Analogy** | An everyday comparison or image from another domain | "It's like learning to ride a bike." |
| **List** | Three or more items, or a three-part rhythm | "Cheap, quick, and good." |
| **Emphatic repetition** | Repeating a word or phrase for emphasis | "Ten minutes. Just ten minutes." |
| **Humor** | Jokes, teasing the audience or themself | |
| **Authority** | Backing a point with their own experience or status | "I've done this for twenty years…" |
| **Provocative opening** | A hook that grabs attention at the start | |

### How it works

1. **Passages.** Each recording's utterances are grouped into passages of about 350 words. The model also sees the few utterances just before each passage, for context, but only annotates the passage itself.
2. **Annotation.** For each passage, the model returns a structured JSON list: device, utterance numbers, an exact quote, and a one-line explanation. The answer is requested as structured output with bounded lengths. Local servers such as llama.cpp enforce that shape while generating, which keeps a model from looping; with OpenAI it's guidance, and malformed answers are retried.
3. **Verification.** Every quote is checked against the transcript; quotes the model invented or paraphrased are discarded. A quote shortened with "…" is accepted if each part is found, in order. The report says how many annotations were kept and discarded.
4. **Aggregation.** Counts per device, rate per 1,000 words, how many recordings each appears in, and up to five examples spread across recordings → `profile/rhetoric.md` and `profile/rhetoric.json`.
5. **Caching.** Each passage's result is saved in `profile/rhetoric_cache/`. Re-running only annotates new passages, and an interrupted run resumes where it stopped.

### How it's used

The style card lists the devices found in at least two recordings, each with a generic instruction ("Voice the listener's objection before they do, then answer it") and one of the speaker's own quotes as an example. The passages chosen as examples for generation favour those that contain several different devices.

**Limits.** These are model judgments, not measurements. In our tests about one annotation in six was mislabelled (a self-correction labelled as reframing, a two-item enumeration labelled as a list). Treat the counts as rough and the quotes as the useful part. Rare devices (one or two occurrences) say little. Speed depends on the model: about a minute per passage on a local 27B model with thinking off, much faster on hosted APIs.

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

