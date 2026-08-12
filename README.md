# xtrack-er

A CLI application for vocal removal and stem separation, built on top of [demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`). Split any song into vocals, drums, bass, guitar, piano and other, then mix them back together live — mute or solo any track, and change playback speed without touching the pitch.

## Features

- **Six-stem separation** — vocals, drums, bass, guitar, piano, other, powered by `htdemucs_6s`.
- **Live mixer** — mute and solo any combination of tracks, with click-free gain ramps.
- **Pitch-preserving speed control** — ±40% in 1% steps (5% steps with `,`/`.`), tone stays intact.
- **Content-addressed cache** — a song is separated once; reopening it (even under a different filename) loads instantly.
- **System audio capture** — record what's playing on your machine (or from a specific app, with the right loopback setup) to a file, then separate it like any other track.
- **No `ffmpeg` required** — reading and writing audio both go through libraries that don't need it (`sphn`/libsndfile).

## Requirements

- Python 3.10–3.13
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- No `ffmpeg` install needed for MP3/WAV. Optional: installing `ffmpeg` unlocks a few extra compressed formats (M4A, AAC, ...) as a bonus, not a requirement.

## Installation

```bash
uv sync
```

This installs everything into `.venv`, including `demucs` and `torch`. The `htdemucs_6s` model itself (~80 MB) downloads automatically the first time you separate a file.

## Usage

`record`, `separate` and `play` are three independent tools — not modes of one app. They connect only through the filesystem: a recording is a file, a separation is a cache entry keyed off that file's content.

```bash
uv run python -m src.main record        # capture audio to a file
uv run python -m src.main separate      # split a file into cached tracks
uv run python -m src.main play          # mix and play separated tracks
```

A typical session:

1. **`record`** (optional) — capture a song from a microphone, a line input, or system audio (see below) to a WAV file. If you already have an MP3 or WAV, skip this step.
2. **`separate`** — pick a file (or pass one directly: `separate path/to/song.mp3`) and wait a couple of minutes while it's split into six stems. Re-running it on the same file reuses the cache instead of re-separating, unless you choose to overwrite.
3. **`play`** — pick a separated song from the list. In the mixer:
   - `1`–`6` mute a track, `q w e r t y` solo it (positionally paired: `q` sits above `1`)
   - `a` / `n` unmute / mute everything
   - `-`/`+` nudge speed by 1%, `,`/`.` by 5%, `0` resets to 100%
   - `←`/`→` seek, `space` play/pause, `Esc` back to the menu

Every menu uses `Esc` to go back or quit, depending on how deep you are.

### Capturing system audio (macOS)

macOS has no built-in loopback device, so recording "what's playing" needs a virtual audio driver:

1. `brew install --cask blackhole-2ch`
2. Restart your Mac.
3. Open **Audio MIDI Setup** and create a **Multi-Output Device** containing both your speakers and BlackHole 2ch.
4. Set that Multi-Output Device as the system output in Sound settings — otherwise you'd record audio but hear nothing.
5. In `record`, pick "BlackHole 2ch" as the source.

The app detects when this isn't set up yet and prints these steps itself. Note that Spotify and Apple Music have no per-app output selector, so this route captures all system audio, not a single application — true per-app capture on macOS would need a ScreenCaptureKit helper, which doesn't exist yet (see Limitations).

`record` and `play` always resolve their audio devices independently, and `play` never defaults to a loopback/Multi-Output device — otherwise the app's own playback would feed straight back into whatever you're recording.

## Separation quality

`SeparationEngine` picks the best available device automatically (CUDA, then Apple Silicon's `mps`, then CPU) and separates with `shifts=5, overlap=0.5` — a well-documented demucs quality trade-off (test-time shift averaging plus finer chunk overlap) that the demucs API doesn't apply by default. On a GPU or Apple Silicon this costs little; on CPU-only machines it takes longer, but a song is only separated once and then cached.

## Testing

```bash
uv run pytest
```

The suite is hermetic and fast — no real audio files, no model downloads — except for one integration test that opens a real audio device and skips itself automatically when none is available.

## Limitations

- **No backing-vocal split** — `vocals` is a single stem; separating it further into lead vs. backing vocals isn't implemented yet.
- **No per-instance separation** — if a song has two guitars, they come out mixed together in one `guitar` stem. Splitting same-category instruments (two guitars, two voices with similar timbre) is an open research problem, not a solved one; no production-ready model does this today.
- **Per-application audio capture** (e.g. "record only Spotify") needs a native macOS helper (ScreenCaptureKit) that doesn't exist yet. Today, capture is either a physical input or system-wide loopback audio.
- **Windows loopback capture** is not implemented; the capture backend abstraction supports it, but nothing registers a WASAPI backend yet.

## License

[MIT](LICENSE)
