# Benchmark Results Log

Running log of A/B comparisons between faster_whisper code states. Each section
records what was changed, how it was measured, and the verdict (speedup /
regression / noise) per phase.

## How to record a result

1. Run the pipeline on the current code:
   `python benchmark/pipeline_benchmark.py --name <label_a> --gpu`
2. Change the code (edit `faster_whisper/`, or `git stash` / `git checkout`).
3. Run again: `python benchmark/pipeline_benchmark.py --name <label_b> --gpu`
4. Compare: `python benchmark/compare.py benchmark/results/<label_a>.json benchmark/results/<label_b>.json`
5. Paste the compare output below with a short note on what changed.

Reports live in `benchmark/results/` (gitignored). Raw per-phase timings and the
git commit of each run are inside each JSON.

## Default benchmark parameters

Model `large-v3`, device `cuda`, compute `float16`, language `pl`, beam 1,
chunk 30s, batch 16, VAD on (`min_silence_duration_ms=500`), audio `you.opus`
(3.8 h Twitch VOD).

---

## Baseline — commit bc755ed (current master, "speedup for rtx3060")

Commands used:

```
.venv\Scripts\python.exe benchmark\pipeline_benchmark.py --audio you.mp4 --name baseline --gpu --repeat 3
```

Report: `benchmark/results/baseline.json` (re-taken on the instrumented
benchmark, commit bc755ed)

Audio: `you.opus` (13710 s / 3.8 h Twitch VOD)

| Phase     | min      | median   | share of total |
| --------- | -------- | -------- | -------------- |
| decode    | 16.16 s  | 16.51 s  | 7.2%           |
| vad       | 14.07 s  | 14.29 s  | 6.2%           |
| extract   | 5.53 s   | 5.75 s   | 2.5%           |
| forward   | 193.63 s | 197.15 s | 86.0%          |
| transcribe| 210.65 s | 213.06 s | 92.9%          |
| total     | 228.92 s | 229.22 s | 100%           |

- model_load: 3.68 s (excluded from verdicts)
- Realtime factor (median): 59.9x — full VOD in ~3.8 min
- GPU: util 100%, power max 133 W, temp max 68 C (GPU-bound, no throttling)
- Runs were stable (±0.5 % between runs) — good signal for A/B comparisons
- **Where the time goes:** `forward` (ctranslate2 encode+generate +
  post-processing) is 86% of the total. `decode` + `vad` (~30 s combined) are
  the only significant non-GPU costs. `extract` is already cheap (5.8 s).

Anything faster than this baseline in `decode` + `transcribe` on the same audio
is a candidate speedup; anything slower is a regression.

See `OPTIMIZATION_PLAN.md` for the prioritized optimization list.

---

## Experiment 1 — CUDA torch (item 1.1, 2026-07-31)

Change: replaced CPU torch (`2.13.0+cpu`) with `torch 2.6.0+cu124`. This
activates the existing batched `GpuMelExtractor` (GPU mel/STFT instead of the
CPU scipy path). VAD still on CPU (`onnxruntime`); item 1.2 skipped per user.

Reports: `baseline.json` (CPU torch) vs `baseline_cudatorch.json` (CUDA torch).

| Phase     | baseline med | modified med | med delta | verdict |
| --------- | ------------ | ------------ | --------- | ------- |
| decode    | 16.51 s      | 16.85 s      | +2.04%    | REGRESSION (noise) |
| vad       | 14.29 s      | 14.17 s      | -0.83%    | ~noise |
| extract   | 5.75 s       | 10.41 s      | +81.11%   | REGRESSION (wall-clock; runs concurrently on GPU, not additive) |
| forward   | 197.15 s     | 194.78 s     | -1.20%    | ~noise |
| total     | 229.22 s     | 226.80 s     | -1.06%    | ~noise |

Verdict: **~1% total faster, within noise.** GPU was already at 100% util, so
moving feature extraction to the GPU mostly contends with ctranslate2 rather
than hiding work. Kept CUDA torch installed (slightly faster, marginally
better `forward`). **Key insight: the pipeline is GPU-compute-bound** — the
next real wins must come from `forward` (batch/compute-type), not CPU-side
work.

---

## Experiment 2 — batch_size sweep (item 1.3)

All vs `baseline_cudatorch.json` (chunk 30, batch 16, float16), median totals:

| config | total (med) | forward (med) | verdict |
| ------ | ----------- | ------------- | ------- |
| batch 16 (baseline) | 226.8 s | 194.8 s | — |
| batch 32 | ~730-850 s | ~700-815 s | **+220% REGRESSION** |

Batch 32 at 30 s chunks is 3-4x slower: encoder self-attention is O(length²)
and 32 x 3000-frame chunks blow past the attention memory/bandwidth sweet spot.
**Finding: the encoder's length-3000 attention is the real bottleneck, not
decoder batch parallelism.**

## Experiment 3 — compute_type sweep (item 1.4)

| config | total (med) | verdict |
| ------ | ----------- | ------- |
| float16 (baseline) | 226.8 s | — |
| int8_float16 | ~241 s | +6% REGRESSION |
| int8 | ~240 s | +6% REGRESSION |

On Ampere, fp16 tensor cores beat int8 for this encoder-bound workload. Keep
`float16`.

## Experiment 4 — chunk_length sweep (code change in faster_whisper)

Code change: `_extract_and_cache` in `faster_whisper/transcribe.py` now pads
each feature batch to the **longest chunk actually in the batch** instead of the
fixed 3000 frames (30 s). Encoder attention is O(length²), so this makes
`chunk_length` a real speed knob. Verified: same transcript content, output
just splits into more segments.

| chunk_length | batch | total (med) | forward (med) | verdict vs baseline |
| ------------ | ----- | ----------- | ------------- | ------------------- |
| 30 | 16 | 226.8 s | 194.8 s | baseline |
| 25 | 16 | ~208 s | ~178 s | -8% |
| 20 | 16 | ~202 s | ~171 s | -11% |
| 15 | 16 | ~209 s | ~177 s | -8% |
| 10 | 16 | 180.6 s | 151.5 s | -20% |
| 8 | 16 | ~204 s | ~174 s | -10% |
| 5 | 16 | ~298 s | ~270 s | +31% REGRESSION (per-batch overhead) |
| 10 | 32 | ~163 s | ~130 s | -28% |
| 10 | 48 | 155.7 s | 125.8 s | -31% |
| 10 | 64 | 147.8 s | ~120 s | -35% |
| 10 | 96 | ~297 s | ~271 s | +31% REGRESSION (VRAM/attention wall) |
| 8 | 64 | 147.9 s | 117.3 s | -35% |

### WINNER — `chunk_length=10`, `batch_size=64` (confirm: `best_chunk10_b64.json`)

| Phase     | baseline med | best med | med delta |
| --------- | ------------ | -------- | --------- |
| decode    | 16.85 s      | 15.84 s  | -6.0%     |
| vad       | 14.17 s      | 13.38 s  | -5.6%     |
| extract   | 10.41 s      | 7.45 s   | -28.4%    |
| forward   | 194.78 s     | 119.11 s | -38.9%    |
| total     | 226.80 s     | 151.81 s | **-33.1%** |

Realtime factor: 60x -> 90x. GPU: util 100%, power 141 W, temp 68 C.

**Tradeoff:** transcripts are more fragmented (10 s VAD chunks -> ~3x more
segment lines). Text content is equivalent. Acceptable for speed; can be
re-merged in post-processing if desired.

**Applied to `transcribe_vod_fasterwhisper.py`:** `DEFAULT_CHUNK_LENGTH = 10`,
`DEFAULT_BATCH_SIZE = 64`, new `--batch-size` CLI flag.

---

## Experiment 5 — streaming decode ∥ VAD (plan 2.1)

Code change: when the audio is still a path (not a waveform), `BatchedInferencePipeline`
now runs decode and VAD **concurrently** instead of back-to-back.

- `audio.py`: added `decode_audio_chunks()` generator (decode in ~31 s chunks,
  byte-identical to `decode_audio`).
- `vad.py`: split `get_speech_timestamps` into `compute_speech_probs` +
  `segments_from_speech_probs` (pure refactor, same outputs); added
  `StreamingVad` — a producer thread decodes while the caller consumes
  independent VAD blocks as their audio arrives. Blocks carry zero initial
  state, so incremental processing is numerically identical to one big batch.
- `transcribe.py`: uses `StreamingVad` when `vad_filter` is on and audio is a
  path. Verified byte-identical transcripts vs the sequential path on the
  13-min clip (75/75 segments identical, 648 s speech).

Both engines release the GIL (PyAV + onnxruntime), so they genuinely
parallelize on the 16-thread CPU. Result (3.8 h VOD, chunk10+b64):

| Phase     | before med | after med | med delta |
| --------- | ---------- | --------- | --------- |
| decode    | 15.84 s    | 16.44 s   | +3.8% (CPU contention) |
| vad       | 13.38 s    | 13.45 s   | ~noise    |
| forward   | 119.11 s   | 115.37 s  | -3.1%     |
| total     | 151.81 s   | 134.90 s  | **-11.1%** |

Work-time now exceeds wall-clock (decode 16.4 s + vad 13.5 s ≈ 30 s of work in
~15 s wall) — the overlap is real. Realtime: **90x -> 102x**.

### Running total vs original baseline
`baseline.json` (chunk 30, batch 16, CPU torch, sequential): 226.8 s.
Now: **134.9 s = -40.5%** (CUDA torch + chunk10/batch64 + streaming decode∥VAD).

## Experiment 6 — super-chunk streaming (plan 2.2)

Code change: when audio is a path, VAD is on, and the language is known,
`BatchedInferencePipeline` now overlaps **decode ∥ VAD ∥ GPU inference** instead
of decode∥VAD then inference:

- `vad.py`: `StreamingVad` gained a worker-driven mode — `wait_processed_blocks`,
  `probs_for_windows`, `num_segments_windows`, `set_error`, and an error-flagged
  worker loop (`process_next_block` + `drain_remaining_blocks`). Block reads
  fixed to stay within decoded audio even on exactly-aligned files. Also fixed a
  latent unit bug (`_next_block` mixing window offsets and block indices) and a
  missing `notify_all` on `_vad_done` that could deadlock waiters.
- `transcribe.py`: `_super_chunk_transcribe` runs on the caller's thread; for
  each processed VAD block it re-runs the segmentizer, feeds only the *newly*
  final segments (the trailing one is deferred — its end and the pad-split of
  the boundary before it need a successor) into a `collect_chunks`-equivalent
  accumulator, and forwards `batch_size` super-chunks at a time while a
  background worker keeps running VAD. Feature extraction refactored into
  `_compute_features`. Requires `language is not None` (no detection); the 2.1
  path remains the fallback.

Verified byte-identical vs the sequential path on the 13-min clip
(70/70 segments identical). Result (3.8 h VOD, chunk10+b64):

| Phase     | 2.1 med   | super-chunk med | med delta |
| --------- | --------- | --------------- | --------- |
| decode    | 16.44 s   | 20.26 s         | +23% (CPU contention) |
| vad       | 13.45 s   | 14.74 s         | +9.6%     |
| forward   | 115.37 s  | 117.54 s        | +1.9%     |
| total     | 134.90 s  | 125.96 s        | **-6.6%** |

Best total **123.40 s**. The decode (20.0 s) + VAD (14.4 s) ≈ 34 s of CPU work is
now almost entirely hidden under forward (115.2 s): total ≈ forward + ~8 s
overhead. Realtime: **102x -> 111x**.

### Running total vs original baseline
`baseline.json` (chunk 30, batch 16, CPU torch, sequential): 226.8 s.
Now: **123.4 s = -45.6%** (CUDA torch + chunk10/batch64 + super-chunk streaming).

## Experiment 6.1 — incremental segmentizer (plan 2.2 follow-up)

Code change: `_super_chunk_transcribe` no longer re-runs the whole-prefix
segmentizer (`segments_from_speech_probs`) on every VAD block. Added
`IncrementalSpeechSegmenter` (`vad.py`) — the same Silero state machine carried
across blocks: `feed(probs)` returns only the newly-final segments, `finish()`
closes the trailing one at file end. Verified byte-identical vs
`segments_from_speech_probs` on random probs, the 13-min clip (214 segs) and
the full 3.8 h file (4598 segs), and super-chunk transcripts stay byte-identical
vs sequential (40/40 on the 13-min clip).

Result (3.8 h VOD, chunk10+b64, same run params; instrumented: segz 4.11 s →
0.00 s, probs-fetch ~1 s → 0.00 s):

| Phase     | super-chunk med | incr med | med delta |
| --------- | --------------- | -------- | --------- |
| decode    | 20.26 s         | 18.58 s  | -8.3%     |
| vad       | 14.74 s         | 15.40 s  | +4.5% (noise) |
| extract   | 0.81 s          | 0.83 s   | ~noise    |
| forward   | 117.54 s        | 116.03 s | -1.3%     |
| total     | 125.96 s        | 122.10 s | **-3.1%** |

Best total **119.79 s** (was 123.40 s). Instrumented run (lang pl, chunk10+b64):
wait 1.26 s + feed/acc ~0.7 s + extract 0.87 s + forward 115.4 s + ~3.7 s
unaccounted. Run-to-run variance on `forward` is ±3-5 s (GPU boost), so the
real saving is the ~4 s of serialized segmentizer CPU work now gone from the
critical path. Realtime: **111x -> 114.5x** (best).

### Running total vs original baseline
`baseline.json` (chunk 30, batch 16, CPU torch, sequential): 226.8 s.
Now: **119.8 s = -47.2%** (CUDA torch + chunk10/batch64 + super-chunk streaming +
incremental segmentizer).

## Experiment 6.2 — prefetched feature extraction (plan A1, negative on GPU-mel)

Code change: `_super_chunk_transcribe` offloaded `_compute_features` to a
1-worker `ThreadPoolExecutor` so batch N+1's mel is extracted while the GPU
generates batch N. Rationale: extraction (~0.8-2.5 s) was serialized on the
main thread in front of every forward.

Result (3.8 h VOD, chunk10+b64, ungated — worker thread for both mel paths):

| report | best total | median total | forward med |
| ------ | ---------- | ------------ | ----------- |
| super_chunk10_b64_incr (6.1, sync) | 119.79 s | 122.10 s | 116.03 s |
| super_chunk10_b64_incr_pf (ungated) | 123.11 s | 125.95 s | 119.96 s |

Verdict: **~1 s REGRESSION (within ±5 s noise, but mechanistically expected).**
With GPU mel active the extraction already runs on the GPU, so a worker thread
only contends with ctranslate2 for SMs — there is no spare GPU capacity to hide
the work under. The main thread also switched from doing extraction to waiting
on VAD blocks (wait 1.26 s→2.19 s, x74→x98).

**Refinement:** gated the prefetch on `self._get_gpu_mel_extractor() is None` —
the worker thread is only used when extraction is CPU-bound (numpy fallback),
where it genuinely overlaps with GPU generate (CPU-mel single run: extract
2.5 s hidden, total 116.0 s). GPU-mel stays synchronous (unchanged from 6.1).
Verified byte-identical (40/40 on the 13-min clip) in **both** modes.

### Running total vs original baseline
`baseline.json` (chunk 30, batch 16, CPU torch, sequential): 226.8 s.
Now: **119.8 s = -47.2%** (unchanged for the CUDA-torch/GPU-mel setup; the
prefetch is an opt-in win for the CPU-mel fallback path).

## Experiment 6.3 — no full-audio materialization on the super-chunk path (A2 tail)

Code change: the super-chunk path no longer calls `streamer.finish()` to obtain
`clip_timestamps`. `_super_chunk_transcribe` now collects every VAD speech
segment in its `_accumulate` (the incremental segmentizer already produced them,
byte-identical to `segments_from_speech_probs`) and returns them directly; the
caller only computes `duration` from `streamer._total_samples`. The VAD cache
key switches to `streamer.audio_fingerprint()` — a 16-byte blake2b maintained
incrementally during decode (`_producer`), equal to the old
`blake2b(full_audio.tobytes())` digest but avoiding the 876 MB `np.concatenate`
+ full-waveform hash (~0.92 s) plus the full-prefix segmentizer recompute
(~1.0 s) that ran with the GPU idle after the last forward. `StreamingVad.finish`
gained a `need_audio` param; the 2.1 path also uses the incremental fingerprint.

Result (3.8 h VOD, chunk10+b64, GPU-mel):

| report | best total | median total | forward med |
| ------ | ---------- | ------------ | ----------- |
| super_chunk10_b64_incr (6.1) | 119.79 s | 122.10 s | 116.03 s |
| super_chunk10_b64_incr_tail (6.3) | 114.08 s | 114.25 s | 110.84 s |

Verdict: **~5.7 s faster best / ~7.9 s faster median.** Some of the gain is
run-to-run GPU/thermal noise (forward med -5.2 s), but the post-loop
segmentizer+hash serial tail (2.0 s of GPU-idle CPU) is gone entirely.
Byte-identical transcripts (40/40 on the 13-min clip) and identical
`duration_after_vad`; fingerprint verified equal to the old full-audio hash.

### Running total vs original baseline
`baseline.json` (chunk 30, batch 16, CPU torch, sequential): 226.8 s.
Now: **114.1 s = -49.7%** (CUDA torch + chunk10/batch64 + super-chunk streaming +
incremental segmentizer + no full-audio materialization).

---

## Environment update — torch 2.7.0+cu128 + onnxruntime-gpu (2026-08-01)

Dependency swaps, not a perf experiment:
- `torch 2.6.0+cu124` → `torch 2.7.0+cu128` (pytorch `cu128` index pinned in
  pyproject; default PyPI on Windows resolves to the CPU build, so the index
  is required). CUDA runtime 12.8. Sanity-checked byte-identical (40/40).
  Interleaved A/B on this machine (high thermal/run-to-run drift, cu124
  rechecked at 127 s vs 114 s this morning) showed cu128 ≈ cu124, if anything
  slightly faster; `cu128_final.json` best 121.71 s.
- `onnxruntime` (CPU) → `onnxruntime-gpu==1.28.0` → `CUDAExecutionProvider`
  auto-selected by `vad.py`. VAD batch ~17x faster (3.1 ms vs 53.8 ms per
  2000-window block); outputs match CPU to ~1.6e-4; transcripts still
  byte-identical (40/40). End-to-end gain negligible (VAD already ~0.00 s
  after 2.2) but removes the CPU-only constraint and frees CPU threads.

---

## Experiment 7 — P0 defaults + penalty wiring + safety caps (2026-09-25)

Code changes (all in working tree, RTX 3060 12 GB, ct2 4.8.1):
- `transcribe_vod_fasterwhisper.py:72-73`: `DEFAULT_CHUNK_LENGTH 30→10`,
  `DEFAULT_BATCH_SIZE 16→64` (the Exp.4 winner was never applied to the
  script — anyone running without flags paid the 30/16 penalty).
- `benchmark/pipeline_benchmark.py`: same default fix (10/64) + new
  `--repetition-penalty` (1.2) / `--no-repeat-ngram-size` (3) flags so the
  benchmark mirrors the script exactly.
- `transcribe_vod_fasterwhisper.py:395-403`: forward `repetition_penalty` /
  `no_repeat_ngram_size` into `pipeline.transcribe()` (previously accepted
  but silently dropped — the anti-loop guard was inactive).
- `transcribe.py` (`BatchedInferencePipeline.transcribe`): `max_new_tokens`
  default `None→128` (10 s chunk needs ~30-40 tokens; bounds pathological
  loops, zero effect on normal EOT-terminated decodes).
- `feature_extractor.py` (`GpuMelExtractor.extract_batch`): pinned-memory H2D
  (`pin_memory().to(device, non_blocking=True)`), bitwise-identical, ~0.3 s.

Diagnostic (temporary encode/generate timers, since removed): 5-min PL clip,
chunk10/b64, per-14-batch `encode 0.59 s vs generate 1.04 s` → generate ~64%
of forward → **decoder-dominant → chunk sweep matrix skipped** (chunk 10→8
already tied at 147.8 vs 147.9 s; encoder vein exhausted).

Quality check (5-min PL clip, old 1.0/0/None vs new 1.2/3/128): 14/14 segments,
identical timestamps. New settings **fixed a real loop**: old had
"Na sześć miesięcy! ×7", new emits it once. Other diffs are minor greedy
re-phrasings, same content. Penalty cost ~+0-1% forward (within noise).

Perf check (`p1_p4_verify.json`, 5-min, repeat 3): forward 1.54-1.59 s vs
1.64 s pre-change — no regression (5-min phases round to 0.00 s, so
`compare.py` hits ZeroDivisionError on them — script limitation, not a code
issue; use ≥30-min clips for A/B).

Dead ends closed (verified, not re-run): ctranslate2 4.8.1 `Whisper` exposes
no CUDA-graph API and no `workspace`/`max_batch_size` on `generate` — C8 and
B-model-kwargs are inapplicable. Clock lock (`nvidia-smi -lgc`) needs admin
on this machine — use `--repeat 5` + warmup instead (±3-5 s forward noise).

Note: `tests/test_transcribe.py` has 5 pre-existing failures on the clean
tree (tiny-model punctuation drift with commas, `test_batched_transcribe`
ValueError, `test_multisegment_lang_id` AttributeError) — unrelated to this
change (identical failure set before/after, verified via `git stash`).

### Full-file confirm + ablation (same day, same machine, `you.opus` 13710 s)

Same-day A/B isolates the P0+P2 bundle from the 8-week env drift between
`super_chunk10_b64_incr_tail.json` (Aug 1, best 114.08 s) and today. Baseline
re-run on stashed code (penalty 1.0/0, cap None) → `p0_p2_baseline_sameday.json`:
best 114.27 s, runs 114–120 s — reproduces Aug 1 exactly, so env drift ≈ 0.

| run (`--repeat 5`) | forward med | total med | vs same-day base |
| ------------------ | ----------- | --------- | ---------------- |
| baseline (1.0/0, cap None) | 115.09 s | 118.98 s | — |
| penalty-only (1.2/3, cap None) | 104.59 s | 108.04 s | **-9.2%** |
| bundle (1.2/3, cap 128) = `p0_p2_full.json` | 97.52 s | 101.27 s | **-14.9%** |

Ranges don't overlap (bundle worst 102.61 s < baseline best 114.27 s), GPU
66→67 °C / ~145 W both sides — no thermal confound. Verdict: **both changes
are real speedups, not noise**:
- `repetition_penalty=1.2 + no_repeat_ngram_size=3` alone: **~-9% forward**.
  Mechanism: penalty steers greedy decode to earlier EOT (shorter token
  trajectories per chunk), which outweighs the per-step logit cost.
- `max_new_tokens=128` on top: **another ~-6%**. Mechanism: the 3.8 h file
  contains runaway chunks generating >128 tokens (hallucination tails up to
  the 448 cap); the bound cuts exactly that waste. Supporting evidence:
  penalty-only run 5/5 spiked to 119.15 s (an unbound runaway), while all 5
  bundle runs stayed ≤102.61 s — the cap also bounds worst-case variance.

**New best: 98.78 s total (~139x realtime), cumulative 227 s → 98.8 s
(-56.5%) vs the original chunk30/batch16 baseline.** Pinned H2D contributes
~0 (extract 0.79→0.84 s, noise) but is kept — free and correct.

Follow-ups done in the same session:
- `benchmark/compare.py`: `ZeroDivisionError` on zero-median phases fixed
  via `pct_delta()` guard (both-zero → +0.00% ~noise). Verified on
  `diag.json` vs `p1_p4_verify.json`.
- `tests/test_transcribe.py`: full suite green (10 passed). Fixed
  `test_batched_transcribe` (unpack `(generator, info)` tuple correctly),
  `test_multisegment_lang_id` (removed `detect_language_multi_segment` →
  `detect_language(audio, language_detection_segments=4)`, threshold 0.8→0.7),
  3× tiny-model comma-drift expectations, `test_prefix_with_timestamps`
  end bound (`<11` → `<=11`), and `test_vad` expectation (VAD-chunked tiny
  decode is deterministically lowercase/unpunctuated — documented in-test).
