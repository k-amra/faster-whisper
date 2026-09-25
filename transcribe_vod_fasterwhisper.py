"""
This script provides a pipeline to download a video from a URL (specifically designed for Twitch VODs on CloudFront),
convert its audio to Opus format, and then transcribe the audio using the Faster Whisper model.

It handles downloading with yt-dlp, audio conversion with ffmpeg, and transcription with faster-whisper.

The input can be either a video URL or a path to an already-downloaded local
media file (e.g. an .mp4) — local files skip the download step automatically.

Example (URL):
    python transcribe_vod_fasterwhisper.py "https://d....cloudfront.net/<hash>_<streamer>_<vodid>_<ts>/chunked/index-dvr.m3u8"
Example (local file):
    python transcribe_vod_fasterwhisper.py /path/to/video.mp4
"""

import os

# --- Environment configuration (must run before any third-party imports) ---
# Workaround for a potential libiomp5md.dll conflict on Windows with Intel libs.
# This only takes effect if set BEFORE the OpenMP runtime is loaded, which can happen
# as early as `import faster_whisper` (via ctranslate2) — so it must stay at the top.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import logging
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# --- Safe DLL Loading for Windows ---
# The handles returned by os.add_dll_directory() MUST be kept alive: once a handle is
# closed or garbage-collected, the directory is removed from the DLL search path again.
_DLL_DIR_HANDLES: List[Any] = []
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
        print("If you are running on CPU, you can ignore this.")
        print("If using GPU, ensure you ran: pip install nvidia-cudnn-cu12 nvidia-cublas-cu12")

import yt_dlp
from faster_whisper import WhisperModel, BatchedInferencePipeline

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MODEL_SIZE = "large-v3"  # Or try "large-v3-turbo"
DEFAULT_DEVICE = "cuda"  # Or "cpu"
DEFAULT_COMPUTE_TYPE = (
    "float16"  # Or "int8_float16", "int8", "float32" based on GPU/CPU
)
CPU_FALLBACK_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 1 # was 5
DEFAULT_CHUNK_LENGTH = 10  # Seconds of audio per transcription chunk (proven best on RTX 3060 12GB: RESULTS.md Exp.4)
DEFAULT_BATCH_SIZE = 64  # Batched chunks per inference step (proven best with chunk 10: total -33%, forward -39%)
DEFAULT_VAD_PARAMS: Dict[str, Any] = {"min_silence_duration_ms": 500}
DEFAULT_LANGUAGE = "pl"
# Anti-hallucination decoder constraints: penalize repetition of already
# generated tokens (repetition_penalty > 1) and forbid repeating 3-grams, which
# stops loops like "Dziękuję bardzo, dziękuję bardzo, ...".
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
FFMPEG_LOG_LEVEL = "error"  # Use "info" or "debug" for more detailed ffmpeg logs
FFMPEG_TIMEOUT_SECONDS = 3600  # Safety net so a hung ffmpeg can't block the pipeline
YTDLP_RETRIES = 3
YTDLP_CONCURRENT_FRAGMENTS = 16

# Matches a Twitch CloudFront path segment shaped like:
#   <hash>_<streamer>_<vodid>_<timestamp>
# The vod ID and timestamp are numeric; the streamer name itself may contain
# underscores, so the numeric fields are anchored at the END of the segment
# instead of splitting left-to-right.
_CLOUDFRONT_SEGMENT_RE = re.compile(
    r"^[^_]+_(?P<streamer>.+)_(?P<vod_id>\d+)_(?P<timestamp>\d+)$"
)

# Extensions yt-dlp may produce when downloading the video.
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".ts", ".mov", ".flv")


def _format_duration(seconds: float) -> str:
    """Formats a wall-clock duration as '<Hh <Mm <Ss' (omitting larger units)."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _redact_url(url: str) -> str:
    """Returns the URL with any query string stripped, safe for logging.

    CloudFront signed URLs embed credentials (Policy/Signature/Key-Pair-Id) in the
    query string; those should not end up in log files.
    """
    parsed = urlparse(url)
    if not parsed.query and not parsed.fragment:
        return url
    return parsed._replace(query="", fragment="").geturl() + "?<redacted>"


def parse_cloudfront_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a CloudFront URL to extract the streamer name and VOD ID.

    Looks for a path segment shaped like <hash>_<streamer>_<vodid>_<timestamp>.
    The query string is ignored so signed-URL signatures can't produce false matches,
    and streamer names containing underscores are handled correctly.

    Args:
        url (str): The full CloudFront URL.

    Returns:
        tuple: A tuple containing (streamer_name, vod_id). Returns (None, None) if parsing fails.
    """
    try:
        path = urlparse(url).path
        for part in path.split("/"):
            match = _CLOUDFRONT_SEGMENT_RE.match(part)
            if match:
                streamer_name = match.group("streamer")
                vod_id = match.group("vod_id")
                logger.info(
                    f"Parsed from '{part}': Streamer='{streamer_name}', VOD_ID='{vod_id}'"
                )
                return streamer_name, vod_id
        logger.error(
            "Could not find a '<hash>_<streamer>_<vodid>_<timestamp>' segment "
            f"in URL path: {path}"
        )
        return None, None
    except Exception as e:
        logger.error(
            f"Error parsing CloudFront URL '{_redact_url(url)}': {e}", exc_info=True
        )
        return None, None


def download_video(url: str, output_base: str) -> Optional[str]:
    """
    Downloads a video from a URL using yt-dlp.

    Args:
        url (str): URL of the video to download.
        output_base (str): Output path WITHOUT extension. yt-dlp appends the real
                           extension of whatever format it downloads/merges.

    Returns:
        str or None: The actual path of the downloaded file, or None if download fails.
    """
    output_dir = os.path.dirname(output_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": output_base + ".%(ext)s",
        "quiet": False,
        "progress": True,
        "concurrent_fragment_downloads": YTDLP_CONCURRENT_FRAGMENTS,
        "retries": YTDLP_RETRIES,
        "merge_output_format": "mp4",
    }
    try:
        logger.info(f"Attempting to download video from {_redact_url(url)}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # extract_info(download=True) also handles the "already downloaded" case:
            # it returns the info dict pointing at the existing file instead of
            # silently skipping, so re-runs work.
            info = ydl.extract_info(url, download=True)
            if info is None:
                logger.error("yt-dlp returned no video info.")
                return None
            # The authoritative final path (after merging/remuxing) lives in
            # requested_downloads; fall back to prepare_filename on older yt-dlp.
            requested = info.get("requested_downloads")
            if requested:
                downloaded_file_path = requested[0].get("filepath")
            else:
                downloaded_file_path = ydl.prepare_filename(info)
        if downloaded_file_path and os.path.exists(downloaded_file_path):
            logger.info(f"Video available at {downloaded_file_path}")
            return downloaded_file_path
        logger.error("yt-dlp finished but the downloaded file was not found on disk.")
        return None
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download failed: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during download: {e}", exc_info=True
        )
        return None


def convert_to_opus(input_file: str, output_file: str) -> bool:
    """
    Converts a media file to Opus audio format using ffmpeg.

    This function extracts the audio, converts it to mono Opus at 32 kbps bitrate,
    and resamples it to 16 kHz, which is what Whisper models expect.

    Args:
        input_file (str): Path to the input media file.
        output_file (str): Path to save the converted Opus file.

    Returns:
        bool: True if conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If input_file does not exist.
        RuntimeError: If ffmpeg command is not found.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found for conversion")
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOG_LEVEL,
        "-y",  # Overwrite existing files without asking
        "-i",
        input_file,
        "-vn",  # Drop the video stream
        "-c:a",
        "libopus",  # Codec
        "-b:a",
        "32k",  # Bitrate
        "-ar",
        "16000",  # Resample to 16 kHz (what Whisper expects); uses swresample, no libsoxr needed
        "-ac",
        "1",  # Force mono audio
        output_file,
    ]
    try:
        logger.info(f"Converting {input_file} to Opus format at {output_file}...")
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        logger.debug(f"FFmpeg stdout:\n{process.stdout}")
        logger.debug(f"FFmpeg stderr:\n{process.stderr}")
        logger.info(f"Successfully converted to Opus: {output_file}")
        return True
    except FileNotFoundError:
        logger.error(
            "ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH."
        )
        raise RuntimeError("ffmpeg not found")
    except subprocess.TimeoutExpired:
        logger.error(
            f"FFmpeg conversion timed out after {FFMPEG_TIMEOUT_SECONDS} seconds."
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed with exit code {e.returncode}")
        logger.error(f"FFmpeg stderr:\n{e.stderr}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during conversion: {e}", exc_info=True
        )
    # Clean up potentially incomplete output file
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.info(f"Removed incomplete Opus file: {output_file}")
        except OSError as rm_err:
            logger.error(f"Failed to remove incomplete file {output_file}: {rm_err}")
    return False


def transcribe_audio(
    input_file: str,
    output_file: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: Optional[str] = DEFAULT_LANGUAGE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    vad_params: Optional[Dict[str, Any]] = None,
    use_vad: bool = True,
    force: bool = False,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
) -> bool:
    """
    Transcribes an audio file using the FasterWhisper model.

    Args:
        input_file (str): Path to the audio file to transcribe.
        output_file (str): Path to save the transcription output (.txt).
        model_size (str): Size of the Whisper model to use (e.g., 'large-v3', 'medium').
        device (str): Device to run the model on ('cuda' or 'cpu'). Falls back to CPU
                      automatically if the requested device is unavailable.
        compute_type (str): Computation type ('float16', 'int8', etc.).
        language (str, optional): Language code (e.g., 'pl', 'en') or None for auto-detect.
        beam_size (int): Beam size for decoding.
        chunk_length (int): Length of audio chunks in seconds to process at a time.
        batch_size (int): Maximum number of chunks to process in parallel per step.
        vad_params (dict, optional): Parameters for voice activity detection.
        use_vad (bool): Whether to use Voice Activity Detection.
        force (bool): Re-transcribe even if output_file already exists.
        repetition_penalty (float): Penalty applied to the score of previously
            generated tokens (set > 1 to penalize). Reduces hallucinated
            repetition loops.
        no_repeat_ngram_size (int): Prevent repetitions of ngrams of this size
            (set 0 to disable). Reduces hallucinated repetition loops.

    Returns:
        bool: True if transcription was successful, False otherwise.

    Raises:
        FileNotFoundError: If input_file does not exist.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input audio file {input_file} not found")
    output_dir = os.path.dirname(output_file)
    if (
        output_dir
    ):  # Ensure output_dir is not empty (happens if output is in current dir)
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file) and not force:
        logger.info(
            f"Transcription output already exists: {output_file}. "
            "Skipping (use --force to redo)."
        )
        return True  # Treat existing file as success

    # Configure VAD parameters if VAD is enabled
    effective_vad_params = DEFAULT_VAD_PARAMS.copy()
    if vad_params:
        effective_vad_params.update(vad_params)

    # Write to a temp file first; it is renamed to output_file only on success.
    tmp_output_file = output_file + ".tmp"
    try:
        # Load the model, falling back to CPU if the requested device is unavailable
        # (e.g. CUDA requested but no GPU present or cuDNN/cuBLAS DLLs missing).
        logger.info(
            f"Initializing Whisper model: {model_size} on {device} ({compute_type})"
        )
        _model_t0 = time.perf_counter()
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            if device == "cpu":
                raise
            logger.warning(
                f"Could not initialize the model on '{device}'. "
                f"Falling back to CPU ({CPU_FALLBACK_COMPUTE_TYPE}).",
                exc_info=True,
            )
            model = WhisperModel(
                model_size, device="cpu", compute_type=CPU_FALLBACK_COMPUTE_TYPE
            )

        logger.info(f"Model loaded in {_format_duration(time.perf_counter() - _model_t0)}.")
        logger.info(f"Starting transcription for {input_file}...")
        logger.info(f"  Language: {'auto-detect' if language is None else language}")
        logger.info(f"  VAD enabled: {use_vad}")
        if use_vad:
            logger.info(f"  VAD parameters: {effective_vad_params}")
        logger.info(f"  Beam size: {beam_size}")
        logger.info(f"  Chunk length: {chunk_length}s")
        pipeline = BatchedInferencePipeline(model)
        segments, info = pipeline.transcribe(
            input_file,
            language=language,
            beam_size=beam_size,
            chunk_length=chunk_length,
            vad_filter=use_vad,
            vad_parameters=effective_vad_params if use_vad else None,
            batch_size=batch_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        detected_lang = info.language
        detected_lang_prob = info.language_probability
        logger.info(
            f"Detected language: {detected_lang} (Probability: {detected_lang_prob:.4f})"
        )
        if language and detected_lang.lower() != language.lower():
            logger.warning(
                f"Specified language '{language}' but detected '{detected_lang}'!"
            )

        # NOTE: `segments` is a lazy generator — the actual transcription happens while
        # iterating it below, not inside model.transcribe(). The atomic tmp->final rename
        # ensures a crash mid-write can't leave a partial file that the skip-if-exists
        # check above would later mistake for a complete transcript.
        logger.info(f"Decoding and writing transcript to {output_file}...")
        _transcribe_t0 = time.perf_counter()
        with open(tmp_output_file, "w", encoding="utf-8") as f:
            for segment in segments:
                segment_output = (
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s] "
                    f"{segment.text.strip()}\n"
                )
                f.write(segment_output)
        os.replace(tmp_output_file, output_file)
        logger.info(
            f"Transcription took {_format_duration(time.perf_counter() - _transcribe_t0)} "
            f"(saved to {output_file})."
        )
        return True
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        # Only remove the temp file — a pre-existing output_file (with --force) must survive.
        if os.path.exists(tmp_output_file):
            try:
                os.remove(tmp_output_file)
                logger.info(f"Removed incomplete transcription file: {tmp_output_file}")
            except OSError as rm_err:
                logger.error(
                    f"Failed to remove incomplete file {tmp_output_file}: {rm_err}"
                )
        return False


def main(args: argparse.Namespace) -> bool:
    """
    Main processing pipeline.

    Orchestrates the download, conversion, and transcription steps based on the
    provided arguments. The source can be either a CloudFront URL or a path to an
    already-downloaded local media file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        bool: True if every requested step succeeded, False otherwise
              (used for the process exit code).
    """
    source = args.url

    # --- Determine source type: local file or URL ---
    if "://" in source:
        local_input_path = None
    elif os.path.isfile(source):
        local_input_path = os.path.abspath(source)
    else:
        logger.error(f"Source is neither a URL nor an existing file: {source}")
        return False

    if local_input_path:
        # --- Local file input: no download, derive output names from the file ---
        logger.info(f"Using local input file: {local_input_path}")
        downloaded_video_path: Optional[str] = local_input_path
        base_name = os.path.splitext(os.path.basename(local_input_path))[0]
        base_output_dir = args.output_dir or os.path.dirname(local_input_path)
        os.makedirs(base_output_dir, exist_ok=True)
        # Outputs land next to the input file (or in --output-dir if given).
        video_output_base = os.path.join(base_output_dir, base_name)
        logger.info(f"Output will be saved in: {base_output_dir}")
    else:
        # --- URL input: parse streamer/VOD ID to build output paths ---
        logger.info(f"Original URL: {_redact_url(source)}")
        streamer_name, vod_id = parse_cloudfront_url(source)
        if not streamer_name or not vod_id:
            logger.error("Failed to parse streamer/VOD ID from URL. Exiting.")
            return False
        base_output_dir = args.output_dir or "."  # Default to current directory
        streamer_dir = os.path.join(base_output_dir, streamer_name)
        os.makedirs(streamer_dir, exist_ok=True)
        logger.info(f"Parsed Streamer: {streamer_name}, VOD ID: {vod_id}")
        logger.info(f"Output will be saved in: {streamer_dir}")
        # yt-dlp decides the real video extension.
        video_output_base = os.path.join(streamer_dir, vod_id)
        downloaded_video_path = None

    # Audio/transcript names are fixed relative to the base name.
    opus_path = video_output_base + ".opus"
    transcript_path = video_output_base + ".txt"

    converted_audio_path: Optional[str] = None
    pipeline_ok = False
    try:
        # --- Step 1: Download Video ---
        if local_input_path:
            logger.info("Local file provided. Skipping download.")
        elif not args.skip_download:
            logger.info("-" * 20 + " Starting Download " + "-" * 20)
            _step_t0 = time.perf_counter()
            downloaded_video_path = download_video(source, video_output_base)
            logger.info(
                f"Download step took {_format_duration(time.perf_counter() - _step_t0)}."
            )
            if not downloaded_video_path:
                logger.error("Video download failed. Cannot proceed.")
                return False
        else:
            # If skipping download, assume the video exists next to the output template
            for ext in VIDEO_EXTENSIONS:
                candidate = video_output_base + ext
                if os.path.exists(candidate):
                    downloaded_video_path = candidate
                    break
            if downloaded_video_path:
                logger.info(
                    f"Skipping download. Using existing video file: {downloaded_video_path}"
                )
            else:
                logger.error(
                    "Skip download specified, but no video file found at "
                    f"'{video_output_base}' with any of the extensions {VIDEO_EXTENSIONS}. "
                    "Cannot proceed."
                )
                return False

        # --- Step 2: Convert to Opus Audio ---
        input_for_transcription = downloaded_video_path  # Default to video
        if downloaded_video_path.lower().endswith(".opus"):
            # Guard against ffmpeg reading and writing the same file, and save the work.
            logger.info("Input is already an Opus file. Skipping conversion.")
        elif not args.skip_conversion:
            logger.info("-" * 20 + " Starting Conversion " + "-" * 20)
            _step_t0 = time.perf_counter()
            converted_ok = convert_to_opus(downloaded_video_path, opus_path)
            logger.info(
                f"Conversion step took {_format_duration(time.perf_counter() - _step_t0)}."
            )
            if converted_ok:
                converted_audio_path = opus_path
                input_for_transcription = opus_path  # Use Opus for transcription
                logger.info(
                    f"Using converted Opus file for transcription: {input_for_transcription}"
                )
            else:
                logger.warning(
                    "Audio conversion failed. Attempting transcription from video file."
                )
        elif os.path.exists(opus_path):
            logger.info(f"Skipping conversion. Using existing Opus file: {opus_path}")
            input_for_transcription = opus_path
        else:
            logger.info(
                "Skipping conversion (and Opus file not found). "
                f"Using video file for transcription: {input_for_transcription}"
            )

        # --- Step 3: Transcribe Audio ---
        if not args.skip_transcription:
            logger.info("-" * 20 + " Starting Transcription " + "-" * 20)
            _step_t0 = time.perf_counter()
            pipeline_ok = transcribe_audio(
                input_file=input_for_transcription,
                output_file=transcript_path,
                model_size=args.model_size,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                beam_size=args.beam_size,
                chunk_length=args.chunk_length,
                batch_size=args.batch_size,
                use_vad=(not args.no_vad),
                force=args.force,
            )
            logger.info(
                f"Transcription step took {_format_duration(time.perf_counter() - _step_t0)}."
            )
            if pipeline_ok:
                logger.info("Transcription completed successfully.")
            else:
                logger.error("Transcription failed.")
        else:
            logger.info("Skipping transcription.")
            pipeline_ok = True  # Everything that was requested has been done
        logger.info("-" * 20 + " Pipeline Finished " + "-" * 20)
        return pipeline_ok
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return False
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main pipeline: {e}", exc_info=True
        )
        return False
    finally:
        # Only clean up after a successful run — deleting the video after a failed
        # transcription would destroy the only local copy needed for a retry.
        # Only files created by THIS run are removed: a user-supplied local input
        # file, a --skip-download video, or a pre-existing Opus file is never touched.
        if args.cleanup and pipeline_ok:
            intermediates = []
            if not args.skip_download and local_input_path is None:
                intermediates.append(downloaded_video_path)
            intermediates.append(converted_audio_path)  # set only if we converted this run
            for path in intermediates:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info(f"Cleaned up intermediate file: {path}")
                    except OSError as e:
                        logger.warning(f"Could not clean up file {path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, and transcribe video from a URL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help
    )
    parser.add_argument(
        "url",
        metavar="source",
        help="The CloudFront (or other) video URL to process, or a path to an already-downloaded local media file (e.g. video.mp4).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Base directory to store output files (streamer name will be subdirectory). Defaults to current directory.",
    )
    # Processing Steps Control
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step, URL mode only (assumes video file exists; local file inputs always skip download).",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip the conversion to Opus step (transcribe directly from video or existing Opus).",
    )
    parser.add_argument(
        "--skip-transcription", action="store_true", help="Skip the transcription step."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="After a successful run, delete the intermediate video/audio files created by this run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if the transcript file already exists.",
    )
    # Transcription Options
    parser.add_argument(
        "--model-size", default=DEFAULT_MODEL_SIZE, help="FasterWhisper model size."
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language code for transcription (e.g., 'pl', 'en'). Use 'auto', 'none' or an empty value for auto-detect.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device for transcription ('cuda', 'cpu'). Falls back to CPU if unavailable.",
    )
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="Compute type for transcription (e.g., 'float16', 'int8', 'float32').",
    )
    parser.add_argument(
        "--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="Beam size for decoding."
    )
    parser.add_argument(
        "--no-vad", action="store_true", help="Disable Voice Activity Detection (VAD)."
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH,
        help="Length of audio chunks in seconds to process for transcription.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum number of chunks to process in parallel per inference step.",
    )
    # Parse arguments
    parsed_args = parser.parse_args()
    # Normalize language: empty string, 'none' and 'auto' all mean auto-detect
    if not parsed_args.language or parsed_args.language.lower() in ("none", "null", "auto"):
        parsed_args.language = None
    # Run main function; exit code reflects success/failure
    sys.exit(0 if main(parsed_args) else 1)