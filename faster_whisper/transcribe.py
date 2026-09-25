import itertools
import json
import logging
import os
import threading
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from inspect import signature
from math import ceil
from typing import BinaryIO
from warnings import warn

import ctranslate2
import numpy as np
import tokenizers
from tqdm import tqdm

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor, GpuMelExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_end, get_logger
from faster_whisper.vad import (
    IncrementalSpeechSegmenter,
    SpeechTimestampsMap,
    StreamingVad,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float | None
    # True when generation stopped because it hit the token budget
    # (max_new_tokens / model max_length) instead of emitting EOT.
    # CTranslate2 strips EOT, so hitting the budget exactly means the text
    # is likely cut mid-word. Defaults to False for backwards compatibility.
    truncated: bool = False

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: float | None
    no_speech_threshold: float | None
    compression_ratio_threshold: float | None
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: list[float]
    initial_prompt: str | Iterable[int] | None
    prefix: str | None
    suppress_blank: bool
    suppress_tokens: list[int] | None
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    max_new_tokens: int | None
    clip_timestamps: str | list[float]
    hallucination_silence_threshold: float | None
    hotwords: str | None


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


class BatchedInferencePipeline:
    def __init__(
        self,
        model,
        use_cache: bool = True,
    ):
        self.model: WhisperModel = model
        self.last_speech_timestamp = 0.0
        self.use_cache = use_cache

        # Per-instance caches to skip redundant CPU work when the same audio
        # is transcribed more than once (retries, parameter sweeps, benchmarks).
        # Multi-entry dicts keyed by audio fingerprint so caches survive across
        # different audio clips in the same session (e.g. artemis bench running
        # sequential then concurrent phases on multiple scenarios).
        self._vad_cache: dict = {}  # {(audio_fp, vad_params): clip_timestamps}
        self._feat_cache: dict = {}  # {(audio_fp, batch_start, batch_size): np.ndarray}
        self._cache_lock = threading.Lock()  # guards all cache fields above
        self._gpu_mel = None  # lazily-built GpuMelExtractor (optional)
        self._gpu_mel_init = False

    def _get_gpu_mel_extractor(self):
        """Lazily build the optional torch+CUDA batched mel extractor.

        Returns None when torch or CUDA is unavailable, or when disabled via
        the FASTER_WHISPER_DISABLE_GPU_MEL environment variable; in that case
        feature extraction uses the CPU numpy path."""
        if not self._gpu_mel_init:
            self._gpu_mel_init = True
            if os.environ.get("FASTER_WHISPER_DISABLE_GPU_MEL"):
                return None
            try:
                ext = GpuMelExtractor(self.model.feature_extractor)
                if ext.is_cuda:
                    self._gpu_mel = ext
                    self.model.logger.info("GPU mel feature extraction enabled (torch + CUDA)")
            except Exception as e:
                self.model.logger.debug("GPU mel extraction unavailable: %s", e)
        return self._gpu_mel

    def _compute_features(self, chunks, max_frames):
        """Compute mel features for a list of audio chunks (GPU batched when
        available, CPU numpy otherwise).  ``max_frames`` is the (even-rounded)
        frame count each chunk is padded to, so the batch is rectangular."""
        gpu_mel = self._get_gpu_mel_extractor()
        if gpu_mel is not None:
            return gpu_mel.extract_batch(chunks, max_frames=max_frames)
        _n_mels = self.model.feature_extractor.mel_filters.shape[0]
        result = np.zeros((len(chunks), _n_mels, max_frames), dtype=np.float32)
        for j, chunk in enumerate(chunks):
            f = self.model.feature_extractor(chunk)
            # f.shape = (n_mels, n_frames+1); exclude the last frame to match
            # the original [..,-1] slice, then copy up to max_frames.
            n = min(f.shape[-1] - 1, max_frames)
            result[j, :, :n] = f[:, :n]
        return result

    def forward(self, features, tokenizer, chunks_metadata, options):
        encoder_output, outputs = self.generate_segment_batched(features, tokenizer, options)

        segmented_outputs = []
        segment_sizes = []
        for chunk_metadata, output in zip(chunks_metadata, outputs, strict=False):
            duration = chunk_metadata["duration"]
            segment_size = int(ceil(duration) * self.model.frames_per_second)
            segment_sizes.append(segment_size)
            (
                subsegments,
                seek,
                single_timestamp_ending,
            ) = self.model._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=output["tokens"],
                time_offset=chunk_metadata["offset"],
                segment_size=segment_size,
                segment_duration=duration,
                seek=0,
            )
            segmented_outputs.append(
                [
                    dict(
                        text=(decoded := tokenizer.decode(subsegment["tokens"])),
                        avg_logprob=output["avg_logprob"],
                        no_speech_prob=output["no_speech_prob"],
                        tokens=subsegment["tokens"],
                        start=subsegment["start"],
                        end=subsegment["end"],
                        compression_ratio=get_compression_ratio(decoded),
                        seek=int(chunk_metadata["offset"] * self.model.frames_per_second),
                        truncated=output.get("truncated", False),
                    )
                    for subsegment in subsegments
                ]
            )
        if options.word_timestamps:
            self.last_speech_timestamp = self.model.add_word_timestamps(
                segmented_outputs,
                tokenizer,
                encoder_output,
                segment_sizes,
                options.prepend_punctuations,
                options.append_punctuations,
                self.last_speech_timestamp,
            )

        return segmented_outputs

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ):
        batch_size = features.shape[0]

        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(
                tokenizer.encode(options.initial_prompt)
                if options.initial_prompt is not None
                else []
            ),
            without_timestamps=options.without_timestamps,
            hotwords=options.hotwords,
        )

        if options.max_new_tokens is not None:
            max_length = min(len(prompt) + options.max_new_tokens, self.model.max_length)
        else:
            max_length = self.model.max_length

        if max_length <= len(prompt):
            raise ValueError(
                f"Prompt ({len(prompt)} tokens) leaves no room for generation "
                f"(max_length={max_length}). Shorten prefix/hotwords/initial_prompt "
                "or raise max_new_tokens."
            )

        encoder_output = self.model.encode(features)
        prompts = [prompt.copy() for _ in range(batch_size)]

        if options.multilingual:
            language_tokens = [
                tokenizer.tokenizer.token_to_id(segment_langs[0][0])
                for segment_langs in self.model.model.detect_language(encoder_output)
            ]
            language_token_index = prompt.index(tokenizer.language)

            for i, language_token in enumerate(language_tokens):
                prompts[i][language_token_index] = language_token

        results = self.model.model.generate(
            encoder_output,
            prompts,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
        )

        output = []
        budget = max_length - len(prompt)
        for result in results:
            tokens = result.sequences_ids[0]
            # CTranslate2 strips EOT; hitting the budget exactly means we were
            # cut off by max_length rather than stopping naturally.
            truncated = len(tokens) >= budget
            if truncated:
                self.model.logger.warning(
                    "Chunk hit generation budget (%d tokens) - text is likely "
                    "truncated. Raise/remove max_new_tokens.",
                    budget,
                )
            # return scores
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)

            output.append(
                dict(
                    avg_logprob=cum_logprob / (seq_len + 1),
                    no_speech_prob=result.no_speech_prob,
                    tokens=tokens,
                    truncated=truncated,
                )
            )

        return encoder_output, output

    def transcribe(
        self,
        audio: str | BinaryIO | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 1,  # greedy: ~4-5x faster decode than beam 5
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: float | list[float] | tuple[float, ...] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: str | Iterable[int] | None = None,
        prefix: str | None = None,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = [-1],
        without_timestamps: bool = True,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        vad_filter: bool = True,
        vad_parameters: dict | VadOptions | None = None,
        max_new_tokens: int | None = None,
        chunk_length: int | None = None,
        clip_timestamps: list[dict] | None = None,
        hallucination_silence_threshold: float | None = None,
        batch_size: int = 16,  # large-v3 fp16 on a 12 GB card still has headroom at 16
        hotwords: str | None = None,
        language_detection_threshold: float | None = 0.5,
        language_detection_segments: int = 1,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        """transcribe audio in chunks in batched fashion and return with language info.

        Arguments:
            audio: Path to the input file (or a file-like object), or the audio waveform.
            language: The language spoken in the audio. It should be a language code such
                as "en" or "fr". If not set, the language will be detected in the first 30 seconds
                of audio.
            task: Task to execute (transcribe or translate).
            log_progress: whether to show progress bar or not.
            beam_size: Beam size to use for decoding.
            best_of: Number of candidates when sampling with non-zero temperature.
            patience: Beam search patience factor.
            length_penalty: Exponential length penalty constant.
            repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
            no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
            temperature: Temperature for sampling. If a list or tuple is passed,
                only the first value is used.
            initial_prompt: Optional text string or iterable of token ids to provide as a
                prompt for the each window.
            suppress_blank: Suppress blank outputs at the beginning of the sampling.
            suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                of symbols as defined in `tokenizer.non_speech_tokens()`.
            without_timestamps: Only sample text tokens.
            word_timestamps: Extract word-level timestamps using the cross-attention pattern
                and dynamic time warping, and include the timestamps for each word in each segment.
                Set as False.
            prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the next word
            append_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the previous word
            multilingual: Perform language detection on every segment.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model
                https://github.com/snakers4/silero-vad.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            max_new_tokens: Maximum number of new tokens to generate per-chunk.
                If None (default), the model's full context budget is used
                (max_length 448 for all Whisper sizes). Setting this too low
                (e.g. 128) silently truncates dense speech in token-inefficient
                languages (Polish, Czech, Hungarian, Turkish, ...) which can
                routinely need 150-250 tokens per 30 s window. A low cap only
                bounds worst-case decode length / repetition loops at the cost
                of cut words; pass it explicitly if you want that trade-off.
            chunk_length: The length of audio segments. If it is not None, it will overwrite the
                default chunk_length of the FeatureExtractor.
            clip_timestamps: Optionally provide list of dictionaries each containing "start" and
                "end" keys that specify the start and end of the voiced region within
                `chunk_length` boundary. vad_filter will be ignored if clip_timestamps is used.
            batch_size: the maximum number of parallel requests to model for decoding.
            hotwords:
                Hotwords/hint phrases to the model. Has no effect if prefix is not None.
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Unused Arguments
            compression_ratio_threshold: If the gzip compression ratio is above this value,
                treat as failed.
            log_prob_threshold: If the average log probability over sampled tokens is
                below this value, treat as failed.
            no_speech_threshold: If the no_speech probability is higher than this value AND
                the average log probability over sampled tokens is below `log_prob_threshold`,
                consider the segment as silent.
            condition_on_previous_text: If True, the previous output of the model is provided
                as a prompt for the next window; disabling may make the text inconsistent across
                windows, but the model becomes less prone to getting stuck in a failure loop,
                such as repetition looping or timestamps going out of sync. Set as False
            prompt_reset_on_temperature: Resets prompt if temperature is above this value.
                Arg has effect only if condition_on_previous_text is True. Set at 0.5
            prefix: Optional text to provide as a prefix at the beginning of each window.
            max_initial_timestamp: The initial timestamp cannot be later than this, set at 0.0.
            hallucination_silence_threshold: Optional[float]
                When word_timestamps is True, skip silent periods longer than this threshold
                (in seconds) when a possible hallucination is detected. set as None.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """

        sampling_rate = self.model.feature_extractor.sampling_rate

        if multilingual and not self.model.model.is_multilingual:
            self.model.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False

        chunk_length = chunk_length or self.model.feature_extractor.chunk_length

        _need_segments = clip_timestamps is None
        if _need_segments and vad_filter:
            # Normalize VAD options up front; both the streaming and the classic
            # path need the dataclass form (with max_speech_duration_s injected).
            if vad_parameters is None:
                vad_parameters = VadOptions(
                    max_speech_duration_s=chunk_length,
                    min_silence_duration_ms=160,
                )
            elif isinstance(vad_parameters, dict):
                vad_parameters.pop("max_speech_duration_s", None)
                vad_parameters = VadOptions(**vad_parameters, max_speech_duration_s=chunk_length)

        # When the audio is still a path (not a waveform), decode it and run VAD
        # concurrently via StreamingVad so the ~15 s decode and ~13 s VAD overlap
        # instead of running back-to-back.  Both engines release the GIL, so they
        # genuinely parallelize on multi-core CPUs.
        _stream_vad = not isinstance(audio, np.ndarray) and vad_filter and _need_segments
        # Super-chunk mode: on top of decode//VAD overlap, start GPU inference on
        # each chunk of speech as soon as VAD has decided it, so the ~115 s of
        # forward time also hides the decode+VAD work entirely.  Requires a known
        # language (detection would need the full VAD result up front) and a path
        # input (the streaming decoder reads the file in chunks).
        _super_chunk = _stream_vad and language is not None
        if _super_chunk:
            streamer = StreamingVad(vad_parameters, sampling_rate, progress=log_progress)
            streamer.start(audio)  # decode thread; VAD + transcription follow
            duration = None
            clip_timestamps = None
            _audio_fp = None
        elif _stream_vad:
            streamer = StreamingVad(vad_parameters, sampling_rate, progress=log_progress)
            audio, clip_timestamps = streamer.run(audio)
            duration = audio.shape[0] / sampling_rate

            # The fingerprint is computed incrementally during decode, so this
            # avoids hashing the full waveform again for the cache key.
            _audio_fp = streamer.audio_fingerprint() if self.use_cache else None

            # The VAD already ran (concurrently with decode); the cache only
            # helps repeated transcribes of the same audio, and a hit costs
            # nothing extra since the work was hidden behind decode.
            if self.use_cache:
                from dataclasses import astuple as _astuple

                _vad_key = (_audio_fp, _astuple(vad_parameters))
                with self._cache_lock:
                    _cached = self._vad_cache.get(_vad_key)
                if _cached is not None:
                    clip_timestamps = _cached
                else:
                    with self._cache_lock:
                        self._vad_cache[_vad_key] = clip_timestamps
        else:
            if not isinstance(audio, np.ndarray):
                audio = decode_audio(audio, sampling_rate=sampling_rate)
            duration = audio.shape[0] / sampling_rate

            import hashlib as _hashlib

            _audio_fp = (
                _hashlib.blake2b(audio.tobytes(), digest_size=16).digest()
                if self.use_cache
                else None
            )

        if duration is not None:
            self.model.logger.info("Processing audio with duration %s", format_timestamp(duration))

        # if no segment split is provided, use vad_model and generate segments
        if _need_segments and not _super_chunk:
            if vad_filter:
                if not _stream_vad:
                    from dataclasses import astuple as _astuple

                    _vad_key = (_audio_fp, _astuple(vad_parameters))
                    clip_timestamps = None
                    if self.use_cache:
                        with self._cache_lock:
                            clip_timestamps = self._vad_cache.get(_vad_key)
                    if clip_timestamps is None:
                        clip_timestamps = get_speech_timestamps(
                            audio, vad_parameters, progress=log_progress
                        )
                        if self.use_cache:
                            with self._cache_lock:
                                self._vad_cache[_vad_key] = clip_timestamps
            # run the audio if it is less than 30 sec even without clip_timestamps
            elif duration < chunk_length:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
            else:
                raise RuntimeError(
                    "No clip timestamps found. "
                    "Set 'vad_filter' to True or provide 'clip_timestamps'."
                )

            clip_timestamps_provided = False
            audio_chunks, chunks_metadata = collect_chunks(
                audio, clip_timestamps, max_duration=chunk_length
            )

        elif not _super_chunk:
            clip_timestamps_provided = True
            clip_timestamps = [
                {k: int(v * sampling_rate) for k, v in segment.items()}
                for segment in clip_timestamps
            ]

            audio_chunks, chunks_metadata = [], []
            for i, clip in enumerate(clip_timestamps):
                audio_chunks.append(audio[clip["start"] : clip["end"]])

                clip_duration = (clip["end"] - clip["start"]) / sampling_rate
                if clip_duration > 30:
                    self.model.logger.warning(
                        "Segment %d is longer than 30 seconds, "
                        "only the first 30 seconds will be transcribed",
                        i,
                    )

                chunks_metadata.append(
                    {
                        "offset": clip["start"] / sampling_rate,
                        "duration": clip_duration,
                        "segments": [clip],
                    }
                )

        duration_after_vad = None
        if not _super_chunk:
            duration_after_vad = (
                sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
                / sampling_rate
            )

            self.model.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

        all_language_probs = None

        if language is None:
            if not self.model.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                # detect_language only consumes the first
                # language_detection_segments * 30s of features, so only extract
                # those chunks here. Pre-extracting features for the whole file
                # (the old behaviour) serialized minutes of CPU work in front of
                # GPU inference on long audio; the rest of the chunks are now
                # extracted lazily in the generator, overlapped with inference.
                _detect_features = []
                if duration_after_vad:
                    _needed_frames = (
                        language_detection_segments * self.model.feature_extractor.nb_max_frames
                    )
                    _frames = 0
                    for chunk in audio_chunks:
                        f = self.model.feature_extractor(chunk)[..., :-1]
                        _detect_features.append(f)
                        _frames += f.shape[-1]
                        if _frames >= _needed_frames:
                            break
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(
                    features=np.concatenate(
                        _detect_features
                        + [np.full((self.model.model.n_mels, 1), -1.5, dtype="float32")],
                        axis=1,
                    ),  # add a dummy feature to account for empty audio
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.model.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.model.is_multilingual and language != "en":
                self.model.logger.warning(
                    "The current model is English-only but the language "
                    f"parameter is set to '{language}'; using 'en' instead."
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            temperatures=(
                temperature[:1] if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            hallucination_silence_threshold=None,
            condition_on_previous_text=False,
            clip_timestamps=clip_timestamps,
            prompt_reset_on_temperature=0.5,
            multilingual=multilingual,
            without_timestamps=without_timestamps,
            max_initial_timestamp=0.0,
        )

        if _super_chunk:
            segments, clip_timestamps = self._super_chunk_transcribe(
                streamer,
                tokenizer,
                batch_size,
                options,
                log_progress,
                chunk_length,
                sampling_rate,
                vad_parameters,
            )
            # The VAD ran concurrently with decode + inference and finalized
            # every speech segment, so both the waveform materialization and the
            # full-prefix segmentizer pass can be skipped here.  The cache key
            # uses the running decode fingerprint.
            streamer.join()
            duration = streamer._total_samples / sampling_rate
            duration_after_vad = (
                sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
                / sampling_rate
            )

            self.model.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

            # The VAD ran concurrently with decode + inference; a cache hit for a
            # repeated transcribe costs nothing extra since the work was hidden.
            if self.use_cache:
                from dataclasses import astuple as _astuple

                _vad_key = (
                    streamer.audio_fingerprint(),
                    _astuple(vad_parameters),
                )
                with self._cache_lock:
                    self._vad_cache[_vad_key] = clip_timestamps

            info = TranscriptionInfo(
                language=language,
                language_probability=language_probability,
                duration=duration,
                duration_after_vad=duration_after_vad,
                transcription_options=options,
                vad_options=vad_parameters,
                all_language_probs=all_language_probs,
            )
            segments = restore_speech_timestamps(segments, clip_timestamps, sampling_rate)
            return segments, info

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        # Always pass raw audio chunks so the generator pipelines feature
        # extraction with GPU inference. Use an empty list when there is no
        # voiced audio so the generator exits immediately.
        _gen_input = [] if not duration_after_vad else audio_chunks
        segments = self._batched_segments_generator(
            _gen_input,
            tokenizer,
            chunks_metadata,
            batch_size,
            options,
            log_progress,
            audio_fp=_audio_fp,
        )
        if not clip_timestamps_provided:
            segments = restore_speech_timestamps(segments, clip_timestamps, sampling_rate)

        return segments, info

    def _batched_segments_generator(
        self,
        features_or_chunks,
        tokenizer,
        chunks_metadata,
        batch_size,
        options,
        log_progress,
        audio_fp=None,
    ):
        from concurrent.futures import ThreadPoolExecutor

        # Distinguish pre-extracted features (np.ndarray, shape (n,80,3000)) from
        # raw audio chunks (list of 1-D arrays).  Pre-extracted features arrive when
        # multilingual language detection was required; audio chunks arrive in the
        # common case where the language is already known.
        precomputed = isinstance(features_or_chunks, np.ndarray)
        n_items = len(features_or_chunks)

        pbar = tqdm(total=n_items, disable=not log_progress, position=0)
        seg_idx = 0
        batch_starts = list(range(0, n_items, batch_size))

        if precomputed:
            # Features already available — iterate directly without extra threads.
            for i in batch_starts:
                results = self.forward(
                    features_or_chunks[i : i + batch_size],
                    tokenizer,
                    chunks_metadata[i : i + batch_size],
                    options,
                )
                for result in results:
                    for segment in result:
                        seg_idx += 1
                        yield Segment(
                            seek=segment["seek"],
                            id=seg_idx,
                            text=segment["text"],
                            start=round(segment["start"], 3),
                            end=round(segment["end"], 3),
                            words=(
                                None
                                if not options.word_timestamps
                                else [Word(**word) for word in segment["words"]]
                            ),
                            tokens=segment["tokens"],
                            avg_logprob=segment["avg_logprob"],
                            no_speech_prob=segment["no_speech_prob"],
                            compression_ratio=segment["compression_ratio"],
                            temperature=options.temperatures[0],
                            truncated=segment.get("truncated", False),
                        )
                    pbar.update(1)
        else:
            # Pipelined path: a single background thread extracts features for batch N+1
            # while the GPU is busy with batch N.  Because ctranslate2 releases the GIL
            # during CUDA operations, the background thread runs freely and hides most
            # of the feature-extraction latency under GPU compute.
            audio_chunks = features_or_chunks

            def _extract_and_cache(start):
                # Audio fingerprint is part of the key so entries for different audio
                # files coexist in the cache (multi-audio support).
                key = (audio_fp, start, batch_size)
                if self.use_cache:
                    with self._cache_lock:
                        cached = self._feat_cache.get(key)
                    if cached is not None:
                        return cached
                batch = audio_chunks[start : start + batch_size]
                # Frames each chunk actually contains (same formula the extractor
                # produces: (len+160)//160 - 1 frames, capped at the 3000-frame cap).
                frames = [max(1, min((len(c) + 160) // 160 - 1, 3000)) for c in batch]
                # Encode each batch at the longest chunk actually present instead
                # of the fixed 3000-frame (30 s) cap. Encoder attention cost is
                # O(length^2), so with shorter chunks (smaller chunk_length) this
                # directly reduces GPU work. Round up to an even frame count for
                # the encoder's stride-2 convolutions.
                max_frames = max(frames)
                max_frames += max_frames % 2
                result = self._compute_features(batch, max_frames)
                if self.use_cache:
                    with self._cache_lock:
                        self._feat_cache[key] = result
                return result

            # Fast path: all batches already cached from a prior call on the same audio.
            # Skip the executor entirely and go straight to GPU inference.
            with self._cache_lock:
                _all_cached = self.use_cache and all(
                    (audio_fp, i, batch_size) in self._feat_cache for i in batch_starts
                )
            if _all_cached:
                for i in batch_starts:
                    with self._cache_lock:
                        features = self._feat_cache[(audio_fp, i, batch_size)]
                    results = self.forward(
                        features,
                        tokenizer,
                        chunks_metadata[i : i + batch_size],
                        options,
                    )
                    for result in results:
                        for segment in result:
                            seg_idx += 1
                            yield Segment(
                                seek=segment["seek"],
                                id=seg_idx,
                                text=segment["text"],
                                start=round(segment["start"], 3),
                                end=round(segment["end"], 3),
                                words=(
                                    None
                                    if not options.word_timestamps
                                    else [Word(**word) for word in segment["words"]]
                                ),
                                tokens=segment["tokens"],
                                avg_logprob=segment["avg_logprob"],
                                no_speech_prob=segment["no_speech_prob"],
                                compression_ratio=segment["compression_ratio"],
                                temperature=options.temperatures[0],
                                truncated=segment.get("truncated", False),
                            )
                        pbar.update(1)
            else:
                # For multi-batch audio use a background thread so feature extraction
                # for batch N+1 overlaps with GPU inference on batch N.  For single-batch
                # audio the executor adds thread-creation overhead with no pipeline benefit.
                if len(batch_starts) > 1:
                    # numpy's FFT releases the GIL, so 2 workers genuinely
                    # parallelize STFT across chunks while staying memory-light.
                    _pool = ThreadPoolExecutor(max_workers=2)
                    _futures = [_pool.submit(_extract_and_cache, i) for i in batch_starts]

                    def _get_features(fut):
                        return fut.result()
                else:
                    _pool = None
                    _futures = [_extract_and_cache(batch_starts[0])]

                    def _get_features(feat):
                        return feat

                try:
                    for i, future in zip(batch_starts, _futures, strict=False):
                        features = _get_features(future)
                        results = self.forward(
                            features,
                            tokenizer,
                            chunks_metadata[i : i + batch_size],
                            options,
                        )
                        for result in results:
                            for segment in result:
                                seg_idx += 1
                                yield Segment(
                                    seek=segment["seek"],
                                    id=seg_idx,
                                    text=segment["text"],
                                    start=round(segment["start"], 3),
                                    end=round(segment["end"], 3),
                                    words=(
                                        None
                                        if not options.word_timestamps
                                        else [Word(**word) for word in segment["words"]]
                                    ),
                                    tokens=segment["tokens"],
                                    avg_logprob=segment["avg_logprob"],
                                    no_speech_prob=segment["no_speech_prob"],
                                    compression_ratio=segment["compression_ratio"],
                                    temperature=options.temperatures[0],
                                    truncated=segment.get("truncated", False),
                                )
                            pbar.update(1)
                finally:
                    if _pool is not None:
                        _pool.shutdown(wait=True)

        pbar.close()
        self.last_speech_timestamp = 0.0

    def _super_chunk_transcribe(
        self,
        streamer,
        tokenizer,
        batch_size,
        options,
        log_progress,
        chunk_length,
        sampling_rate,
        vad_options,
    ):
        """Transcribe while decode + VAD still run on background threads.

        The caller starts decode (``streamer.start(audio_path)``) and this
        method starts a worker that runs VAD inference over each block as its
        audio is decoded.  On the caller's thread it waits for each processed
        block and feeds its probabilities into an
        ``IncrementalSpeechSegmenter``, which carries the Silero state machine
        across blocks and returns only the newly-final segments (the last one
        is deferred: its end, and the pad-split of the boundary before it, are
        only final once the following segment appears — and at file end
        ``finish`` closes it).  Those segments are sliced out of the
        already-decoded audio, merged into ``chunk_length``-sized super-chunks
        exactly like ``collect_chunks`` does, and transcribed with ``forward()``
        in ``batch_size`` batches, overlapping GPU inference with the CPU
        decode/VAD work.

        Returns the list of transcribed Segment objects (absolute timestamps
        are restored by the caller once the VAD has fully finished).
        """
        block = streamer.block_windows

        self.last_speech_timestamp = 0.0
        pbar = tqdm(total=None, disable=not log_progress, position=0)
        segments_out = []

        def _run_vad_worker(streamer):
            try:
                while streamer.process_next_block():
                    pass
                streamer.drain_remaining_blocks()
            except BaseException as exc:
                streamer.set_error(exc)

        threading.Thread(
            target=_run_vad_worker, args=(streamer,), daemon=True, name="vad-worker"
        ).start()

        seg_idx = 0

        def _make_segment(segment):
            nonlocal seg_idx
            seg_idx += 1
            return Segment(
                seek=segment["seek"],
                id=seg_idx,
                text=segment["text"],
                start=round(segment["start"], 3),
                end=round(segment["end"], 3),
                words=(
                    None
                    if not options.word_timestamps
                    else [Word(**word) for word in segment["words"]]
                ),
                tokens=segment["tokens"],
                avg_logprob=segment["avg_logprob"],
                no_speech_prob=segment["no_speech_prob"],
                compression_ratio=segment["compression_ratio"],
                temperature=options.temperatures[0],
                truncated=segment.get("truncated", False),
            )

        # Super-chunk accumulator (mirrors collect_chunks() exactly).
        cur_slices = []
        cur_segments = []
        cur_duration = 0
        total_duration = 0
        super_chunks = []  # (audio_slice, metadata)
        clip_timestamps = []  # every VAD speech segment, in order

        # Feature extraction runs on a worker thread so batch N+1's mel is
        # computed while the GPU generates batch N.  This only pays off when
        # extraction is CPU-bound (numpy fallback): the work then genuinely
        # overlaps with ctranslate2's GPU generate.  When GPU mel is active the
        # extraction already runs on the GPU and a worker would only contend
        # with generate for SMs, so we keep it synchronous.  _compute_features
        # is deterministic either way, so the forwarded features — and the
        # transcript — are byte-identical to the synchronous version.
        from concurrent.futures import ThreadPoolExecutor

        _prefetch_extract = self._get_gpu_mel_extractor() is None

        def _prepare_batch(batch):
            slices = [b[0] for b in batch]
            metas = [b[1] for b in batch]
            frames = [max(1, min((len(c) + 160) // 160 - 1, 3000)) for c in slices]
            max_frames = max(frames) + max(frames) % 2
            features = self._compute_features(slices, max_frames)
            return features, metas

        def _forward_features(features, metas):
            results = self.forward(features, tokenizer, metas, options)
            for result in results:
                for segment in result:
                    segments_out.append(_make_segment(segment))
                pbar.update(1)

        extract_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="mel-extract")
            if _prefetch_extract
            else None
        )
        # Future of (features, metas) for the next batch to forward.
        pending = None

        def _drain_ready():
            nonlocal pending
            while len(super_chunks) >= batch_size:
                batch = super_chunks[:batch_size]
                del super_chunks[:batch_size]
                if not _prefetch_extract:
                    features, metas = _prepare_batch(batch)
                    _forward_features(features, metas)
                    continue
                # Start extracting the newly-formed batch on the worker while we
                # generate the one extracted during the previous forward.
                new_pending = extract_pool.submit(_prepare_batch, batch)
                if pending is not None:
                    features, metas = pending.result()
                    _forward_features(features, metas)
                pending = new_pending

        def _accumulate(segments):
            nonlocal cur_slices, cur_segments, cur_duration, total_duration
            for seg in segments:
                clip_timestamps.append(seg)
                seg_dur = seg["end"] - seg["start"]
                if cur_duration and cur_duration + seg_dur > chunk_length * sampling_rate:
                    super_chunks.append(
                        (
                            np.concatenate(cur_slices),
                            {
                                "offset": total_duration / sampling_rate,
                                "duration": cur_duration / sampling_rate,
                                "segments": cur_segments,
                            },
                        )
                    )
                    total_duration += cur_duration
                    cur_slices = [streamer._read(seg["start"], seg_dur)]
                    cur_segments = [seg]
                    cur_duration = seg_dur
                else:
                    cur_slices.append(streamer._read(seg["start"], seg_dur))
                    cur_segments.append(seg)
                    cur_duration += seg_dur

        # The segmentizer state machine runs incrementally: feed each block's
        # probabilities as they become ready and forward the newly-finalized
        # segments.  This avoids re-running the O(n) segmentizer over the whole
        # prefix on every block.
        segz = IncrementalSpeechSegmenter(vad_options, sampling_rate)
        try:
            block_index = 0
            while True:
                streamer.wait_processed_blocks(block_index + 1)
                if streamer._vad_done:
                    total_windows = streamer.num_segments_windows()
                    if total_windows > block_index * block:
                        _accumulate(
                            segz.feed(
                                streamer.probs_for_windows(block_index * block, total_windows)
                            )
                        )
                    _accumulate(segz.finish(streamer._total_samples))
                    _drain_ready()
                    break
                _accumulate(
                    segz.feed(
                        streamer.probs_for_windows(block_index * block, (block_index + 1) * block)
                    )
                )
                _drain_ready()
                block_index += 1

            if cur_slices:
                super_chunks.append(
                    (
                        np.concatenate(cur_slices),
                        {
                            "offset": total_duration / sampling_rate,
                            "duration": cur_duration / sampling_rate,
                            "segments": cur_segments,
                        },
                    )
                )
            # Drain the remaining super-chunks through the same extraction
            # pipeline, then forward the last batch the worker prepared.
            while super_chunks:
                batch = super_chunks[:batch_size]
                del super_chunks[:batch_size]
                if not _prefetch_extract:
                    features, metas = _prepare_batch(batch)
                    _forward_features(features, metas)
                    continue
                new_pending = extract_pool.submit(_prepare_batch, batch)
                if pending is not None:
                    features, metas = pending.result()
                    _forward_features(features, metas)
                pending = new_pending
            if _prefetch_extract and pending is not None:
                features, metas = pending.result()
                _forward_features(features, metas)
        finally:
            if extract_pool is not None:
                extract_pool.shutdown(wait=True)

        pbar.close()
        return segments_out, clip_timestamps


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: int | list[int] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: str | None = None,
        local_files_only: bool = False,
        files: dict = None,
        revision: str | None = None,
        use_auth_token: str | bool | None = None,
        flash_attention: bool = False,
        **model_kwargs,
    ):
        """Initializes the Whisper model.

        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
            large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo),
            a path to a converted model directory, or a CTranslate2-converted Whisper model ID from
            the HF Hub. When a size or a model ID is configured, the converted model is downloaded
            from the Hugging Face Hub.
          device: Device to use for computation ("cpu", "cuda", "auto").
          device_index: Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of IDs
            (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
            when transcribe() is called from multiple Python threads (see also num_workers).
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
          num_workers: When transcribe() is called from multiple Python threads,
            having multiple workers enables true parallelism when running the model
            (concurrent calls to self.model.generate() will run in parallel).
            This can improve the global throughput at the cost of increased memory usage.
          download_root: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Fire cache directory.
          local_files_only:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
          files: Load model files from the memory. This argument is a dictionary mapping file names
            to file contents as file-like or bytes objects. If this is set, model_path acts as an
            identifier for this model.
          revision:
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
          use_auth_token: HuggingFace authentication token or True to use the
            token stored by the HuggingFace config folder.
          flash_attention: Enable Flash Attention for the encoder and decoder when running
            on GPU with FP16 compute type (requires CUDA compute capability >= 7.5).
            Reduces attention memory from O(n²) to O(n) and speeds up long-context decoding.
            NOTE: only works with a ctranslate2 build that ships the FA2 kernels;
            the official PyPI wheels do NOT include them (source build with
            WITH_FLASH_ATTN=ON required, Linux + CUDA only). Default: False.
        """
        self.logger = get_logger()

        tokenizer_bytes, preprocessor_bytes = None, None
        if files:
            model_path = model_size_or_path
            tokenizer_bytes = files.pop("tokenizer.json", None)
            preprocessor_bytes = files.pop("preprocessor_config.json", None)
        elif os.path.isdir(model_size_or_path):
            model_path = model_size_or_path
        else:
            model_path = download_model(
                model_size_or_path,
                local_files_only=local_files_only,
                cache_dir=download_root,
                revision=revision,
                use_auth_token=use_auth_token,
            )

        # Note: flash_attention=True only works with ctranslate2 builds that
        # include the FA2 kernels. The official PyPI wheels (all platforms)
        # are built WITHOUT Flash Attention; enabling it with a stock wheel
        # raises "Flash attention 2 is not supported" at generation time.
        # It requires a source build with WITH_FLASH_ATTN=ON (Linux + CUDA).
        self.model = ctranslate2.models.Whisper(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
            files=files,
            flash_attention=flash_attention,
            **model_kwargs,
        )

        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if tokenizer_bytes:
            self.hf_tokenizer = tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
        elif os.path.isfile(tokenizer_file):
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )
        self.feat_kwargs = self._get_feature_kwargs(model_path, preprocessor_bytes)
        self.feature_extractor = FeatureExtractor(**self.feat_kwargs)
        self.input_stride = 2
        self.num_samples_per_token = self.feature_extractor.hop_length * self.input_stride
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = self.feature_extractor.sampling_rate // self.num_samples_per_token
        self.time_precision = 0.02
        self.max_length = 448

    @property
    def supported_languages(self) -> list[str]:
        """The languages supported by the model."""
        return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

    def _get_feature_kwargs(self, model_path, preprocessor_bytes=None) -> dict:
        config = {}
        try:
            config_path = os.path.join(model_path, "preprocessor_config.json")
            if preprocessor_bytes:
                config = json.loads(preprocessor_bytes)
            elif os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as file:
                    config = json.load(file)
            else:
                return config
            valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
            return {k: v for k, v in config.items() if k in valid_keys}
        except json.JSONDecodeError as e:
            self.logger.warning("Could not load preprocessor config: %s", e)

        return config

    def transcribe(
        self,
        audio: str | BinaryIO | np.ndarray,
        language: str | None = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: float | list[float] | tuple[float, ...] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: str | Iterable[int] | None = None,
        prefix: str | None = None,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'¿([{-",
        append_punctuations: str = "\"'.。,,!!??::”)]}、",
        multilingual: bool = False,
        vad_filter: bool = False,
        vad_parameters: dict | VadOptions | None = None,
        max_new_tokens: int | None = None,
        chunk_length: int | None = None,
        clip_timestamps: str | list[float] = "0",
        hallucination_silence_threshold: float | None = None,
        hotwords: str | None = None,
        language_detection_threshold: float | None = 0.5,
        language_detection_segments: int = 1,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          log_progress: whether to show progress bar or not.
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in `tokenizer.non_speech_tokens()`.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          multilingual: Perform language detection on every segment.
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
          max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
            the maximum will be set by the default max_length.
          chunk_length: The length of audio segments. If it is not None, it will overwrite the
            default chunk_length of the FeatureExtractor.
          clip_timestamps:
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to
             process. The last end timestamp defaults to the end of the file.
             vad_filter will be ignored if clip_timestamps is used.
          hallucination_silence_threshold:
            When word_timestamps is True, skip silent periods longer than this threshold
             (in seconds) when a possible hallucination is detected
          hotwords:
            Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.
          language_detection_threshold: If the maximum probability of the language tokens is higher
           than this value, the language is detected.
          language_detection_segments: Number of segments to consider for the language detection.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        sampling_rate = self.feature_extractor.sampling_rate

        if multilingual and not self.model.is_multilingual:
            self.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        self.logger.info("Processing audio with duration %s", format_timestamp(duration))

        if vad_filter and clip_timestamps == "0":
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters, progress=log_progress)
            audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
            audio = np.concatenate(audio_chunks, axis=0)
            duration_after_vad = audio.shape[0] / sampling_rate

            self.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "VAD filter kept the following audio segments: %s",
                    ", ".join(
                        f"[{format_timestamp(chunk['start'] / sampling_rate)} -> "
                        f"{format_timestamp(chunk['end'] / sampling_rate)}]"
                        for chunk in speech_chunks
                    ),
                )

        else:
            speech_chunks = None

        features = self.feature_extractor(audio, chunk_length=chunk_length)

        encoder_output = None
        all_language_probs = None

        # detecting the language if not provided
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                start_timestamp = (
                    float(clip_timestamps.split(",")[0])
                    if isinstance(clip_timestamps, str)
                    else clip_timestamps[0]
                )
                content_frames = features.shape[-1] - 1
                seek = (
                    int(start_timestamp * self.frames_per_second)
                    if start_timestamp * self.frames_per_second < content_frames
                    else 0
                )
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.detect_language(
                    features=features[..., seek:],
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.is_multilingual and language != "en":
                self.logger.warning(
                    "The current model is English-only but the language "
                    f"parameter is set to '{language}'; using 'en' instead."
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(temperature if isinstance(temperature, (list, tuple)) else [temperature]),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )

        segments = self.generate_segments(
            features, tokenizer, options, log_progress, encoder_output
        )

        if speech_chunks:
            segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        return segments, info

    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: list[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> list[list[int]]:
        current_segments = []
        single_timestamp_ending = (
            len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        )

        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0
            and tokens[i] >= tokenizer.timestamp_begin
            and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                start_time = time_offset + start_timestamp_position * self.time_precision
                end_time = time_offset + end_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=start_time,
                        end=end_time,
                        tokens=sliced_tokens,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = tokens[last_slice - 1] - tokenizer.timestamp_begin
                seek += last_timestamp_position * self.input_stride

        else:
            duration = segment_duration
            timestamps = [token for token in tokens if token >= tokenizer.timestamp_begin]
            if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            seek += segment_size

        return current_segments, seek, single_timestamp_ending

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        log_progress,
        encoder_output: ctranslate2.StorageView | None = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - 1
        content_duration = float(content_frames * self.feature_extractor.time_per_frame)

        if isinstance(options.clip_timestamps, str):
            options.clip_timestamps = [
                float(ts)
                for ts in (options.clip_timestamps.split(",") if options.clip_timestamps else [])
            ]

        seek_points: list[int] = [
            round(ts * self.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: list[tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2], strict=False)
        )

        punctuation = "\"'¿([{-\"'.。,，!！?？:：”)]}、"

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = seek * self.feature_extractor.time_per_frame
            window_end_time = float(
                (seek + self.feature_extractor.nb_max_frames)
                * self.feature_extractor.time_per_frame
            )
            segment_size = min(
                self.feature_extractor.nb_max_frames,
                content_frames - seek,
                seek_clip_end - seek,
            )
            segment = features[:, seek : seek + segment_size]
            segment_duration = segment_size * self.feature_extractor.time_per_frame
            segment = pad_or_trim(segment)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Processing segment at %s", format_timestamp(time_offset))

            previous_tokens = all_tokens[prompt_reset_since:]

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            if options.multilingual:
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]

                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language

            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
                hotwords=options.hotwords,
            )

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if (
                    options.log_prob_threshold is not None
                    and avg_logprob > options.log_prob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: dict | None) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: list[dict]) -> dict | None:
                return next((s for s in segments if s["words"]), None)

            (
                current_segments,
                seek,
                single_timestamp_ending,
            ) = self._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=tokens,
                time_offset=time_offset,
                segment_size=segment_size,
                segment_duration=segment_duration,
                seek=seek,
            )

            if options.word_timestamps:
                self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * self.frames_per_second)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(current_segments[si + 1 :])
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"]) * self.frames_per_second
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end
            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=previous_seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment["words"]]
                        if options.word_timestamps
                        else None
                    ),
                )

            if (
                not options.condition_on_previous_text
                or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

            pbar.update(
                (min(content_frames, seek) - previous_seek) * self.feature_extractor.time_per_frame,
            )
        pbar.close()

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        prompt: list[int],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = min(len(prompt) + options.max_new_tokens, self.max_length)
        else:
            max_length = self.max_length

        if max_length <= len(prompt):
            raise ValueError(
                f"Prompt ({len(prompt)} tokens) leaves no room for generation "
                f"(max_length={max_length}). Shorten prefix/hotwords/initial_prompt "
                "or raise max_new_tokens."
            )

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)

            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            decode_result = (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            )
            all_results.append(decode_result)

            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True  # too repetitive

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        options.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if options.log_prob_threshold is not None and avg_logprob < options.log_prob_threshold:
                needs_fallback = True  # average log probability is too low

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    options.log_prob_threshold,
                )

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(below_cr_threshold_results or all_results, key=lambda x: x[1])
            # to pass final temperature for prompt_reset_on_temperature
            decode_result = (
                decode_result[0],
                decode_result[1],
                temperature,
                decode_result[3],
            )

        return decode_result

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: list[int],
        without_timestamps: bool = False,
        prefix: str | None = None,
        hotwords: str | None = None,
    ) -> list[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend(tokenizer.sot_sequence)

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt

    def add_word_timestamps(
        self,
        segments: list[dict],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        if len(segments) == 0:
            return

        text_tokens = []
        text_tokens_per_segment = []
        for segment in segments:
            segment_tokens = [
                [token for token in subsegment["tokens"] if token < tokenizer.eot]
                for subsegment in segment
            ]
            text_tokens.append(list(itertools.chain.from_iterable(segment_tokens)))
            text_tokens_per_segment.append(segment_tokens)

        alignments = self.find_alignment(tokenizer, text_tokens, encoder_output, num_frames)
        median_max_durations = []
        for alignment in alignments:
            word_durations = np.array([word["end"] - word["start"] for word in alignment])
            word_durations = word_durations[word_durations.nonzero()]
            median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
            median_duration = min(0.7, float(median_duration))
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(alignment)):
                    if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                        if alignment[i]["word"] in sentence_end_marks:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration
                        elif alignment[i - 1]["word"] in sentence_end_marks:
                            alignment[i]["start"] = alignment[i]["end"] - max_duration

            merge_punctuations(alignment, prepend_punctuations, append_punctuations)
            median_max_durations.append((median_duration, max_duration))

        for segment_idx, segment in enumerate(segments):
            word_index = 0
            time_offset = segment[0]["seek"] / self.frames_per_second
            median_duration, max_duration = median_max_durations[segment_idx]
            for subsegment_idx, subsegment in enumerate(segment):
                saved_tokens = 0
                words = []

                while word_index < len(alignments[segment_idx]) and saved_tokens < len(
                    text_tokens_per_segment[segment_idx][subsegment_idx]
                ):
                    timing = alignments[segment_idx][word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(time_offset + timing["start"], 2),
                                end=round(time_offset + timing["end"], 2),
                                probability=timing["probability"],
                            )
                        )

                    saved_tokens += len(timing["tokens"])
                    word_index += 1

                # hack: truncate long words at segment boundaries.
                # a better segmentation algorithm based on VAD should be able to replace this.
                if len(words) > 0:
                    # ensure the first and second word after a pause is not longer than
                    # twice the median word duration.
                    if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                        words[0]["end"] - words[0]["start"] > max_duration
                        or (
                            len(words) > 1
                            and words[1]["end"] - words[0]["start"] > max_duration * 2
                        )
                    ):
                        if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
                            boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                            words[0]["end"] = words[1]["start"] = boundary
                        words[0]["start"] = max(0, words[0]["end"] - max_duration)

                    # prefer the segment-level start timestamp if the first word is too long.
                    if (
                        subsegment["start"] < words[0]["end"]
                        and subsegment["start"] - 0.5 > words[0]["start"]
                    ):
                        words[0]["start"] = max(
                            0,
                            min(words[0]["end"] - median_duration, subsegment["start"]),
                        )
                    else:
                        subsegment["start"] = words[0]["start"]

                    # prefer the segment-level end timestamp if the last word is too long.
                    if (
                        subsegment["end"] > words[-1]["start"]
                        and subsegment["end"] + 0.5 < words[-1]["end"]
                    ):
                        words[-1]["end"] = max(
                            words[-1]["start"] + median_duration, subsegment["end"]
                        )
                    else:
                        subsegment["end"] = words[-1]["end"]

                    last_speech_timestamp = subsegment["end"]
                segments[segment_idx][subsegment_idx]["words"] = words
        return last_speech_timestamp

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        text_tokens: list[int],
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> list[dict]:
        if len(text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        return_list = []
        for result, text_token in zip(results, text_tokens, strict=False):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(text_token + [tokenizer.eot])
            if len(word_tokens) <= 1:
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return_list.append([])
                continue
            word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
            if len(word_boundaries) <= 1:
                return_list.append([])
                continue

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]
            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:], strict=False)
            ]

            return_list.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words,
                        word_tokens,
                        start_times,
                        end_times,
                        word_probabilities,
                        strict=False,
                    )
                ]
            )
        return return_list

    def detect_language(
        self,
        audio: np.ndarray | None = None,
        features: np.ndarray | None = None,
        vad_filter: bool = False,
        vad_parameters: dict | VadOptions = None,
        language_detection_segments: int = 1,
        language_detection_threshold: float = 0.5,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """
        Use Whisper to detect the language of the input audio or features.

        Arguments:
            audio: Input audio signal, must be a 1D float array sampled at 16khz.
            features: Input Mel spectrogram features, must be a float array with
                shape (n_mels, n_frames), if `audio` is provided, the features will be ignored.
                Either `audio` or `features` must be provided.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Returns:
            language: Detected language.
            language_probability: Probability of the detected language.
            all_language_probs: List of tuples with all language names and probabilities.
        """
        assert audio is not None or features is not None, (
            "Either `audio` or `features` must be provided."
        )

        if audio is not None:
            if vad_filter:
                speech_chunks = get_speech_timestamps(audio, vad_parameters)
                audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
                audio = np.concatenate(audio_chunks, axis=0)

            audio = audio[: language_detection_segments * self.feature_extractor.n_samples]
            features = self.feature_extractor(audio)

        features = features[
            ..., : language_detection_segments * self.feature_extractor.nb_max_frames
        ]

        detected_language_info = {}
        for i in range(0, features.shape[-1], self.feature_extractor.nb_max_frames):
            encoder_output = self.encode(
                pad_or_trim(features[..., i : i + self.feature_extractor.nb_max_frames])
            )
            # results is a list of tuple[str, float] with language names and probabilities.
            results = self.model.detect_language(encoder_output)[0]

            # Parse language names to strip out markers
            all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
            # Get top language token and probability
            language, language_probability = all_language_probs[0]
            if language_probability > language_detection_threshold:
                break
            detected_language_info.setdefault(language, []).append(language_probability)
        else:
            # If no language detected for all segments, the majority vote of the highest
            # projected languages for all segments is used to determine the language.
            language = max(
                detected_language_info,
                key=lambda lang: len(detected_language_info[lang]),
            )
            language_probability = max(detected_language_info[language])

        return language, language_probability, all_language_probs


def restore_speech_timestamps(
    segments: Iterable[Segment],
    speech_chunks: list[dict],
    sampling_rate: int,
) -> Iterable[Segment]:
    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    for segment in segments:
        if segment.words:
            words = []
            for word in segment.words:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word.start + word.end) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word.start = ts_map.get_original_time(word.start, chunk_index)
                word.end = ts_map.get_original_time(word.end, chunk_index)
                words.append(word)

            segment.start = words[0].start
            segment.end = words[-1].end
            segment.words = words

        else:
            segment.start = ts_map.get_original_time(segment.start)
            segment.end = ts_map.get_original_time(segment.end, is_end=True)

        yield segment


def get_ctranslate2_storage(segment: np.ndarray) -> ctranslate2.StorageView:
    segment = np.ascontiguousarray(segment)
    segment = ctranslate2.StorageView.from_array(segment)
    return segment


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: tuple[int],
) -> list[int] | None:
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
            tokenizer.no_speech,
        ]
    )

    return tuple(sorted(set(suppress_tokens)))


def merge_punctuations(alignment: list[dict], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            # prepend it to the following word
            following["word"] = previous["word"] + following["word"]
            following["tokens"] = previous["tokens"] + following["tokens"]
            previous["word"] = ""
            previous["tokens"] = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            # append it to the previous word
            previous["word"] = previous["word"] + following["word"]
            previous["tokens"] = previous["tokens"] + following["tokens"]
            following["word"] = ""
            following["tokens"] = []
        else:
            i = j
        j += 1
