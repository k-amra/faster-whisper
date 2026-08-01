# Benchmarking faster_whisper against transcribe_vod_fasterwhisper.py

`pipeline_benchmark.py` reproduces the exact transcription pipeline used by
`transcribe_vod_fasterwhisper.py` (WhisperModel -> BatchedInferencePipeline with
the same decode options) so you can measure whether a change inside
`faster_whisper` actually speeds up your script.

## Setup (one-time)

```sh
uv venv --python 3.12
uv pip install --python .venv -r requirements.txt yt-dlp nvidia-cudnn-cu12 nvidia-cublas-cu12 scipy
```

> faster-whisper is installed **editable** into the venv via `uv sync` (see
> `pyproject.toml`), so `import faster_whisper` resolves to the local source
> tree. This means switching code revisions is as simple as `git checkout`.

### VAD model asset

`faster_whisper/vad.py` loads the Silero VAD ONNX model from
`faster_whisper/assets/silero_vad_v6.onnx`. That file is **not tracked in git**
(yet) — place it in the repo once:

```sh
curl -L -o faster_whisper/assets/silero_vad_v6.onnx \
  https://huggingface.co/bitsydarel/silero-vad-onnx/resolve/main/silero_vad_v6.2.1.onnx
```

`faster_whisper/vad.py` was also patched to use the v6 model's `state`/`sr`
ONNX interface (the v6.2.1 export has `input`/`state`/`sr` inputs and a single
probability output), so the VAD filter works at all.

All commands below should be run from the repo root with the venv python, e.g.
`.venv\Scripts\python.exe benchmark\pipeline_benchmark.py ...`.

## Quick check (fast, uses first 5 minutes of audio)

```sh
.venv\Scripts\python.exe benchmark\pipeline_benchmark.py --name smoke --limit-minutes 5 --repeat 1 --warmup 1
```

## A/B workflow

1. **Baseline** (current code):

```sh
uv sync --extra dev --extra benchmark
.venv\Scripts\python.exe benchmark\pipeline_benchmark.py ...
```

2. **Change the code** — either edit `faster_whisper/` in place, or switch
   revisions with git:

   ```sh
   git stash        # or: git checkout <other-ref>
   ```

3. **Modified run**:

   ```sh
   .venv\Scripts\python.exe benchmark\pipeline_benchmark.py --name modified
   ```

4. **Compare**:

   ```sh
   .venv\Scripts\python.exe benchmark\compare.py benchmark\results\baseline.json benchmark\results\modified.json
   ```

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--audio` | `you.mp4` (repo root) | Input media. Non-Opus files are converted to 16 kHz mono 32k Opus once and cached as `<name>.opus`. |
| `--limit-minutes N` | — | Transcribe only the first N minutes (cached as `<name>.<N>min.opus`). |
| `--repeat N` | 3 | Number of timed runs. |
| `--warmup N` | 1 | Untimed warmup runs (lets the GPU reach boost clocks). |
| `--gpu` | off | Sample GPU utilization / power / temperature during runs. |
| `--name NAME` | `run` | Report filename (`benchmark/results/<NAME>.json`). |
| `--model-size`, `--device`, `--compute-type`, `--language`, `--beam-size`, `--chunk-length`, `--batch-size`, `--no-vad`, `--vad-parameters` | mirror the script | Transcription options. |

## What is measured

- `model_load` — model loading; reported but excluded from the verdict (dominated by disk I/O).
- `decode` — PyAV audio decoding + resampling (`faster_whisper/audio.py`).
- `transcribe` — feature extraction, VAD, and batched decoding (the rest).
- `total` — decode + transcribe (headline metric).
- Real-time factor (RTF) = audio seconds / wall seconds is reported for each phase.

Timing is wall-clock `perf_counter`. Transcript text is never written to disk (that
overhead belongs to the script, not faster_whisper).

## Notes

- Results are written to `benchmark/results/` which is gitignored.
- For a stable signal: close other GPU processes, let the machine idle-warm,
  and prefer the median over the min (the min is the least affected by noise).
- A full run on the 3.8 h `you.mp4` takes a long time. Iterate with
  `--limit-minutes` and only use the full file for final confirmation.
