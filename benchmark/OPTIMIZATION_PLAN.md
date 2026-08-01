# Optimization Plan — faster_whisper on an RTX 3060 (12 GB)

Goal: make the `transcribe_vod_fasterwhisper.py` pipeline (large-v3, cuda,
float16, batched) as fast as possible on a single RTX 3060 12 GB.

Constraints decided with the user:
- Model is **large-v3 only** (no large-v3-turbo / distil swaps).
- **No Flash Attention** (requires a custom ctranslate2 source build; too
  heavy on Windows).

Every change must be validated with the benchmark:
`compare.py baseline modified` + logged in `benchmark/RESULTS.md`.
One variable at a time.

## Current state (baseline before any optimization)

`benchmark/results/baseline.json` — commit bc755ed, `you.opus` (13710 s):

| Phase      | median   | share |
| ---------- | -------- | ----- |
| decode     | 15.7 s   | 7%    |
| transcribe | 211.6 s  | 93%   |
| total      | 227.3 s  | 100%  |

Environment facts that matter (discovered during setup):
- `torch 2.7.0+cu128` (pytorch `cu128` index) → `GpuMelExtractor` is active,
  feature extraction runs on the GPU. (Previously the default PyPI wheel
  resolves to the CPU build `torch 2.13.0+cpu`, which disables GPU mel.)
- `onnxruntime-gpu 1.28` → **VAD runs on CUDA** (~17x faster per batch than
  the CPU `onnxruntime` build; ~3 ms vs ~54 ms per 2000-window block).
- `ctranslate2 4.8.1` → CUDA works (1 device), GPU decode is active.

> IMPORTANT: The initial baseline only splits decode vs transcribe. After
> Phase 0 the benchmark splits transcribe into `vad` / `extract` / `forward`
> so each item below can be measured precisely. The baseline must be re-taken
> on the instrumented benchmark before trusting any comparison.

---

## Phase 0 — Measurement (do first)

- **0.1 Instrument the benchmark.** Split the `transcribe` phase into
  `vad` (Silero VAD), `extract` (mel/STFT), and `forward` (ctranslate2
  encode+generate + segment post-processing). Without this, none of the
  items below can be judged.
- **0.2 Re-take the baseline** on the instrumented benchmark and record the
  per-phase numbers in `RESULTS.md`.

---

## Phase 1 — Environment + parameter sweep (cheap, likely biggest wins)

### 1.1 Install CUDA torch (unlocks GPU mel extraction)
- **What:** Replace the CPU torch wheel with a CUDA build
  (e.g. `uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128`).
  This makes `torch.cuda.is_available()` True, which activates the existing
  `GpuMelExtractor` (transcribe.py:144) — batched STFT+mel on the GPU instead
  of per-chunk CPU scipy.
- **Impact:** medium (moves CPU work to GPU; how much wall-clock it saves
  depends on whether extraction is currently hidden behind GPU compute by the
  pipelined thread pool).
- **Effort:** low. **Risk:** torch+CUDA wheel is ~2.5 GB; needs cu12 runtime
  (already present via nvidia-cudnn/cublas-cu12).
- **Validate:** `--name <tag> --gpu`, compare `extract` phase.

### 1.2 Install onnxruntime-gpu (VAD on CUDA) — DONE
- `onnxruntime-gpu>=1.14,<2` (was CPU `onnxruntime`) → `CUDAExecutionProvider`
  auto-selected by `vad.py`; VAD batch ~17x faster (3 ms vs 54 ms). Outputs
  byte-identical. Negligible end-to-end gain (VAD already ~0.00 s after 2.2),
  but a clean win on GPU and removes the CPU-only constraint.

### 1.3 Sweep batch_size
- **What:** Try 8 / 16 / 24 / 32 (`--batch-size`). The 12 GB 3060 can hold
  large-v3 fp16 comfortably; bigger batches fill the GPU better, but encoder
  attention memory grows O(batch x len^2).
- **Impact:** potentially the biggest knob for the `forward` phase.
- **Effort:** trivial. **Risk:** VRAM OOM at larger sizes — lower if it fails.
- **Validate:** sweep one value per run, compare `forward`.

### 1.4 Sweep compute_type
- **What:** `float16` (current) vs `int8_float16` vs `int8`
  (`--compute-type`). On Ampere fp16 tensor cores are usually best, but the
  autoregressive decoder can win with int8. large-v3 stays the same model.
- **Impact:** small–medium on `forward`; watch quality (transcribe a short
  clip and sanity-check output).
- **Effort:** trivial. **Risk:** quality loss with int8.
- **Validate:** compare `forward`; eyeball transcript quality on 5 min.

---

## Phase 2 — faster_whisper code changes (measured one at a time)

### 2.1 Overlap VAD with audio decode
- **What:** `decode_audio` (PyAV) and `get_speech_timestamps` (onnxruntime)
  both release the GIL. Today VAD runs serially after the whole file is
  decoded (transcribe.py:454). Run VAD on the decoded signal in a background
  thread while decoding continues, or start VAD on the first decoded super-
  chunk. With 1.2 (GPU VAD) the overlap still hides launch/completion latency.
- **Impact:** small–medium (hides `vad` wall time if it isn't already
  overlapped).
- **Effort:** medium. **Risk:** low; VAD must see the complete signal
  (or per-super-chunk VAD must be equivalent — see 2.2).

### 2.2 Streaming super-chunk pipeline
- **What:** Instead of decode-all → VAD-all → transcribe-all, process the file
  in overlapping super-chunks (e.g. 3–5 min): decode chunk i+1 while VAD + 
  transcribe chunk i. Also reduces peak RAM (~880 MB float32 buffer for the
  full 3.8 h file).
- **Impact:** medium — hides `decode` (15.7 s) and `vad` behind `forward`,
  lowers memory pressure.
- **Effort:** high (touches the generator + clip timestamping). **Risk:**
  medium — VAD boundaries at chunk edges must stay identical (pad overlap).

### 2.3 Trim Python post-processing in forward()
- **What:** In `forward()` (transcribe.py:186) every subsegment calls
  `get_compression_ratio(decoded)` (a zlib compress per segment) and
  `tokenizer.decode`. With thousands of segments this is pure CPU serialized
  with GPU work. Options: skip `compression_ratio` when
  `compression_ratio_threshold is None`, batch the tokenizer decodes, or
  compute lazily in the generator.
- **Impact:** small (likely a few seconds). **Effort:** low. **Risk:** low.
- **Validate:** compare `forward` phase.

### 2.4 Move GPU-mel padding to one batched op
- **What:** `GpuMelExtractor.extract_batch` does a per-chunk `np.pad` loop on
  CPU (feature_extractor.py:304-313). Pad all chunks in one numpy/torch op.
- **Impact:** small, only matters once 1.1 is done. **Effort:** low–medium.
  **Risk:** low.
- **Validate:** compare `extract` phase.

### 2.5 Tune extraction parallelism
- **What:** The extraction thread pool uses 2 workers (transcribe.py:772) and
  scipy FFT caps at 4 workers (feature_extractor.py:12). Try more workers /
  higher FFT threads once 1.1 is in (they matter less with GPU mel).
- **Impact:** small. **Effort:** trivial. **Risk:** low.
- **Validate:** compare `extract` phase.

---

## Phase 3 — Tuning / stability (only after 1–2)

### 3.1 VAD block size
- **What:** `block_windows = 2000` (vad.py:389) controls the ONNX batch. Try
  1000 / 4000 / 8000 once VAD is on GPU (1.2).
- **Impact:** small. **Effort:** trivial. **Risk:** low.

### 3.2 Lock GPU clocks for stable measurements
- **What:** `nvidia-smi -lgc <min>,<max>` pins clocks so thermal/boost variance
  doesn't fake a regression. Not a speedup — measurement hygiene.
- **Effort:** trivial. **Risk:** revert with `nvidia-smi -rgc`.

---

## Prioritized roadmap

1. Phase 0: instrument + re-baseline.
2. 1.1 CUDA torch (re-baseline). 1.2 onnxruntime-gpu done.
3. 1.3 batch sweep → pick best. 1.4 compute sweep → pick best.
4. 2.x code changes in order 2.3 (cheap) → 2.1 → 2.4 → 2.5 → 2.2 (biggest, last).
5. Phase 3 tuning.

Each step: run `pipeline_benchmark.py --name <X> --gpu`, compare with the
current best report, and append the verdict to `benchmark/RESULTS.md`.

## Status (2026-08-01)

DONE:
- Phase 0 (instrumented benchmark + re-baseline).
- 1.1 CUDA torch — ~1% (noise), kept. 1.2 onnxruntime-gpu — installed
  (VAD on CUDA, 17x faster per batch, byte-identical; near-zero end-to-end).
- 1.3 batch sweep — batch 16 best at 30 s chunks; 32 is 3-4x SLOWER.
- 1.4 compute sweep — float16 best; int8/int8_float16 ~+6% slower.
- 2.x `chunk_length` optimization — padded feature batches to the actual
  longest chunk (was fixed 3000 frames); enables short chunks. **BEST:
  `chunk_length=10`, `batch_size=64`** → total −33%, forward −39%.
- **2.1 streaming decode ∥ VAD** — decode_audio_chunks + StreamingVad; decode
  and VAD now overlap (~30 s of CPU work in ~15 s wall). → another −11%.
  **Cumulative: 227 s → 135 s (−40%), ~102x realtime.**
- **2.2 super-chunk streaming** — `_super_chunk_transcribe` runs decode ∥ VAD ∥
  GPU inference: a VAD worker processes blocks while the main thread re-runs
  the segmentizer per block, feeds only newly-final segments into a
  `collect_chunks`-equivalent accumulator and forwards `batch_size` super-chunks
  at a time. Requires a known language (2.1 is the fallback). Byte-identical
  transcripts vs sequential (70/70 on the 13-min clip). decode(20 s)+vad(14 s)
  now hidden under forward(115 s). → another −6.6%.
  **Cumulative: 227 s → 123.4 s (−45.6%), ~111x realtime.**
- **2.2b incremental segmentizer** — added `IncrementalSpeechSegmenter`
  (Silero state machine carried across blocks; `feed` returns only newly-final
  segments, `finish` closes the trailing one) and switched `_super_chunk_transcribe`
  to it. Eliminates the per-block whole-prefix segmentizer re-run (4.11 s /
  37 calls → 0) and the full-prefix probs re-fetch. Verified byte-identical
  (random probs, 13-min 214 segs, 3.8 h 4598 segs; transcripts 40/40).
  → another −3.1% (best 119.79 s).
  **Cumulative: 227 s → 119.8 s (−47.2%), ~114.5x realtime.**
- **A1 prefetched extraction (NEGATIVE on GPU-mel)** — moved `_compute_features`
  to a worker thread so batch N+1's mel is extracted during batch N's generate.
  On GPU-mel it ~1 s regressed (worker contends with ctranslate2 for SMs; no
  spare GPU capacity). **Gated on `_get_gpu_mel_extractor() is None`** so the
  worker is only used for the CPU-numpy mel path (genuine CPU/GPU overlap,
  hides ~2.5 s of extraction). GPU-mel stays synchronous. Byte-identical both
  modes (40/40).
- **A2 no full-audio materialization (6.3)** — `_super_chunk_transcribe` now
  returns the VAD speech segments it already collected instead of the caller
  recomputing them via `streamer.finish()`; the VAD cache key uses an
  incremental `blake2b` fingerprint (`StreamingVad.audio_fingerprint`) instead
  of hashing the materialized waveform. Eliminates the post-loop 876 MB concat +
  full-audio hash (~0.92 s) and the full-prefix segmentizer recompute (~1.0 s)
  that ran with the GPU idle. Byte-identical (40/40).
  → another −5.7 s best (−3.1 s of it run noise).
  **Cumulative: 227 s → 114.1 s (−49.7%), ~120x realtime.**
  Note: forward post-processing (compression_ratio, tokenizer.decode) is only
  ~0.20 s / 16 calls — 2.3 is a dead end.

REMAINING (untested):
- A3 trim `_read` copies (~0.70 s / 7973 calls; contiguous buffer + slice views).
- B ctranslate2 `workspace` / decoder `max_batch_size`; fine-tune chunk 8-10 x
  batch 64-80.
- C7 GPU clock lock (`nvidia-smi -lgc 2100`), C8 CUDA graphs (4.8.1 API check).
- D VAD block size (block_windows=2000), PyAV decode threads.
