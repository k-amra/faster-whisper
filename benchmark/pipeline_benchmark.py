"""
Benchmark for the transcription pipeline used by transcribe_vod_fasterwhisper.py.

Replicates the exact pipeline of that script (WhisperModel -> BatchedInferencePipeline
with the same decode/transcription options) and times it phase by phase so that
changes inside faster_whisper (audio decode, feature extraction, VAD, decoding)
can be A/B compared across git revisions.

Workflow:
    1.  python benchmark/pipeline_benchmark.py --name baseline
    2.  modify faster_whisper (or `git stash` / `git checkout <ref>`)
    3.  python benchmark/pipeline_benchmark.py --name modified
    4.  python benchmark/compare.py benchmark/results/baseline.json benchmark/results/modified.json

Timed phases:
    model_load   - loading the CTranslate2 model (reported, excluded from verdicts)
    decode       - PyAV audio decoding + resampling (faster_whisper/audio.py)
    transcribe   - feature extraction + VAD + batched decoding (everything else)
    total        - decode + transcribe wall time (the headline metric)

Reports a real-time factor (RTF) = audio_seconds / wall_seconds for each phase.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from statistics import mean, median
from typing import Dict, List, Optional

# Ensure the local faster_whisper source (repo root) is importable regardless of
# the cwd or how this script is invoked. faster-whisper is intentionally NOT
# installed into the venv so that `git checkout` between revisions is enough.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Environment configuration (must run before any third-party imports) ---
# Same workaround as transcribe_vod_fasterwhisper.py for a potential
# libiomp5md.dll conflict on Windows with Intel libs.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# --- Safe DLL Loading for Windows (same as transcribe_vod_fasterwhisper.py) ---
_DLL_DIR_HANDLES: List[object] = []
if os.name == "nt":
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        _DLL_DIR_HANDLES.append(
            os.add_dll_directory(os.path.dirname(nvidia.cudnn.lib.__file__))
        )
        _DLL_DIR_HANDLES.append(
            os.add_dll_directory(os.path.dirname(nvidia.cublas.lib.__file__))
        )
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load NVIDIA libraries via Python: {e}")
        print("If using GPU, ensure you ran: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12")

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402
import faster_whisper.transcribe as _transcribe  # noqa: E402


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".ts", ".mov", ".flv")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_media_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    raise RuntimeError(f"Could not determine duration of {path} (is ffprobe installed?)")


def convert_to_opus(input_path: str, output_path: str) -> None:
    """Convert to 16 kHz mono 32 kbps Opus, matching transcribe_vod_fasterwhisper.py."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vn", "-c:a", "libopus", "-b:a", "32k",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]
    print(f"Converting {input_path} -> {output_path} (cached for future runs)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed with exit code {result.returncode}:\n{result.stderr}"
        )


def truncate_audio(input_path: str, output_path: str, minutes: int) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path, "-t", str(minutes * 60),
        "-c", "copy", output_path,
    ]
    print(f"Truncating {input_path} to first {minutes} min -> {output_path} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg truncation failed with exit code {result.returncode}:\n{result.stderr}"
        )


def prepare_audio(args) -> str:
    """Return a path to a 16 kHz mono Opus file, converting/truncating as needed.

    The converted (and optionally truncated) files are cached so repeated
    benchmark runs do not redo the ffmpeg work.
    """
    source = os.path.abspath(args.audio)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Input file not found: {source}")

    base, ext = os.path.splitext(source)
    opus_path = base + ".opus"

    if ext.lower() != ".opus":
        if not os.path.exists(opus_path):
            convert_to_opus(source, opus_path)

    if args.limit_minutes:
        limited_path = base + f".{args.limit_minutes}min.opus"
        if not os.path.exists(limited_path):
            truncate_audio(opus_path, limited_path, args.limit_minutes)
        return limited_path
    return opus_path


class GpuMonitor(threading.Thread):
    """Samples nvidia-smi (util/power/temp) in the background during runs."""

    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: Dict[str, float] = {}
        self.count = 0
        self._stop_event = threading.Event()

    def run(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    fields = [f.strip() for f in result.stdout.strip().split(",")]
                    if len(fields) == 3:
                        util, power, temp = (float(fields[0]), float(fields[1]), float(fields[2]))
                        self.samples["util_max"] = max(self.samples.get("util_max", 0.0), util)
                        self.samples["power_max_w"] = max(self.samples.get("power_max_w", 0.0), power)
                        self.samples["temp_max_c"] = max(self.samples.get("temp_max_c", 0.0), temp)
                        self.count += 1
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()


class PhaseTimers:
    """Monkeypatches faster_whisper.transcribe functions to time each phase.

    Phases:
      decode  - PyAV audio decoding + resampling (decode_audio)
      vad     - Silero VAD speech detection (get_speech_timestamps)
      extract - mel / STFT feature extraction (FeatureExtractor + GpuMelExtractor)
      forward - ctranslate2 encode + generate + segment post-processing
                (BatchedInferencePipeline.forward)
    """

    def __init__(self):
        self.times = {"decode": 0.0, "vad": 0.0, "extract": 0.0, "forward": 0.0}
        self._saved = []

    def _wrap(self, module, name, key):
        original = getattr(module, name)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.times[key] += time.perf_counter() - start

        self._saved.append((module, name, original))
        setattr(module, name, timed)

    def _wrap_streaming_vad(self):
        """Time the streaming decode+VAD path (used when audio is still a path).

        decode and vad run concurrently (either inside StreamingVad.run, the 2.1
        overlap, or across a super-chunk run, the 2.2 overlap), so we add each
        engine's accumulated work time to its phase slot after finish() returns
        (same totals as the sequential path; the wall-clock saving shows up in
        'total').  finish() is the single exit point for both paths.
        """
        original = _transcribe.StreamingVad.finish

        def timed(self_obj):
            try:
                return original(self_obj)
            finally:
                self.times["decode"] += self_obj.decode_time
                self.times["vad"] += self_obj.vad_time

        self._saved.append((_transcribe.StreamingVad, "finish", original))
        setattr(_transcribe.StreamingVad, "finish", timed)

    def __enter__(self):
        self._wrap(_transcribe, "decode_audio", "decode")
        self._wrap(_transcribe, "get_speech_timestamps", "vad")
        self._wrap_streaming_vad()
        self._wrap(_transcribe.FeatureExtractor, "__call__", "extract")
        self._wrap(_transcribe.GpuMelExtractor, "extract_batch", "extract")
        self._wrap(_transcribe.BatchedInferencePipeline, "forward", "forward")
        return self

    def __exit__(self, *exc):
        for module, name, original in self._saved:
            setattr(module, name, original)


def run_pipeline_once(model, args, audio_path) -> Dict[str, float]:
    pipeline = BatchedInferencePipeline(model)
    vad_params = json.loads(args.vad_parameters) if args.vad_parameters else None

    with PhaseTimers() as timers:
        start = time.perf_counter()
        segments, info = pipeline.transcribe(
            audio_path,
            language=args.language,
            beam_size=args.beam_size,
            chunk_length=args.chunk_length,
            vad_filter=not args.no_vad,
            vad_parameters=vad_params if not args.no_vad else None,
            batch_size=args.batch_size,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        for _ in segments:
            pass
        total = time.perf_counter() - start

    return {
        "decode": timers.times["decode"],
        "vad": timers.times["vad"],
        "extract": timers.times["extract"],
        "forward": timers.times["forward"],
        "transcribe": total - timers.times["decode"],
        "total": total,
    }


def summarize(times: List[float]) -> Dict[str, float]:
    return {"min": min(times), "median": median(times), "mean": mean(times)}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the transcribe_vod_fasterwhisper.py transcription pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        default=os.path.join(REPO_ROOT, "you.mp4"),
        help="Media file to transcribe. Non-Opus inputs are converted once and cached.",
    )
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="pl", help="Language code, or 'auto'.")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter.")
    parser.add_argument(
        "--vad-parameters",
        default='{"min_silence_duration_ms": 500}',
        help="VAD parameters as a JSON string (ignored when --no-vad).",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Timed runs per benchmark.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs.")
    parser.add_argument(
        "--limit-minutes",
        type=int,
        default=None,
        help="Only transcribe the first N minutes (cached). Use for quick iteration.",
    )
    parser.add_argument("--gpu", action="store_true", help="Sample GPU util/power/temp.")
    parser.add_argument("--gpu-interval", type=float, default=1.0)
    parser.add_argument("--name", default="run", help="Name of this run (result filename).")
    parser.add_argument(
        "--out-dir", default=os.path.join(REPO_ROOT, "benchmark", "results")
    )
    args = parser.parse_args()

    if args.language and args.language.lower() in ("auto", "none", "null"):
        args.language = None

    audio_path = prepare_audio(args)
    audio_duration = get_media_duration(audio_path)
    commit = get_git_commit()

    print(f"Repo commit : {commit}")
    print(f"Audio       : {audio_path} ({audio_duration:.0f}s)")
    print(f"Model       : {args.model_size} on {args.device} ({args.compute_type})")
    print(f"Params      : beam={args.beam_size} chunk={args.chunk_length}s "
          f"batch={args.batch_size} vad={not args.no_vad} lang={args.language or 'auto'}")

    print("Loading model...")
    load_start = time.perf_counter()
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    model_load = time.perf_counter() - load_start
    print(f"Model loaded in {model_load:.2f}s")

    gpu = None
    if args.gpu:
        gpu = GpuMonitor(args.gpu_interval)
        gpu.start()

    try:
        print(f"Warmup runs: {args.warmup}")
        for i in range(args.warmup):
            run_pipeline_once(model, args, audio_path)
            print(f"  warmup {i + 1}/{args.warmup} done")

        results = {k: [] for k in ("decode", "vad", "extract", "forward", "transcribe", "total")}
        print(f"Timed runs: {args.repeat}")
        for i in range(args.repeat):
            timing = run_pipeline_once(model, args, audio_path)
            for phase, seconds in timing.items():
                results[phase].append(seconds)
            print(
                f"  run {i + 1}/{args.repeat}: total={timing['total']:.2f}s "
                f"decode={timing['decode']:.2f}s vad={timing['vad']:.2f}s "
                f"extract={timing['extract']:.2f}s forward={timing['forward']:.2f}s"
            )
    finally:
        if gpu is not None:
            gpu.stop()
            gpu.join(timeout=args.gpu_interval + 2)

    report = {
        "tool": "pipeline_benchmark",
        "commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": {
            "audio": os.path.basename(audio_path),
            "model_size": args.model_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "language": args.language,
            "beam_size": args.beam_size,
            "chunk_length": args.chunk_length,
            "batch_size": args.batch_size,
            "vad_filter": not args.no_vad,
            "vad_parameters": json.loads(args.vad_parameters) if args.vad_parameters else None,
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
        "audio": {"duration_seconds": audio_duration},
        "phases": {
            "model_load": {"seconds": model_load},
            "decode": summarize(results["decode"]),
            "vad": summarize(results["vad"]),
            "extract": summarize(results["extract"]),
            "forward": summarize(results["forward"]),
            "transcribe": summarize(results["transcribe"]),
            "total": summarize(results["total"]),
        },
        "gpu": gpu.samples if gpu else None,
    }
    report["rtf"] = {
        "decode": {k: v / audio_duration for k, v in report["phases"]["decode"].items()},
        "vad": {k: v / audio_duration for k, v in report["phases"]["vad"].items()},
        "extract": {k: v / audio_duration for k, v in report["phases"]["extract"].items()},
        "forward": {k: v / audio_duration for k, v in report["phases"]["forward"].items()},
        "transcribe": {k: v / audio_duration for k, v in report["phases"]["transcribe"].items()},
        "total": {k: v / audio_duration for k, v in report["phases"]["total"].items()},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote report to {out_path}")

    total_min = report["phases"]["total"]["min"]
    print(
        f"Summary: best total {total_min:.2f}s  "
        f"({audio_duration / total_min:.2f}x realtime)  "
        f"decode {report['phases']['decode']['min']:.2f}s  "
        f"transcribe {report['phases']['transcribe']['min']:.2f}s"
    )


if __name__ == "__main__":
    main()
