import bisect
import functools
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from faster_whisper.utils import get_assets_path


# The code below is adapted from https://github.com/snakers4/silero-vad.
@dataclass
class VadOptions:
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      neg_threshold: Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than min_silence_at_max_speech (if any), to prevent aggressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
      min_silence_at_max_speech: Minimum silence duration in ms which is used to avoid abrupt cuts
          when max_speech_duration_s is reached.
      use_max_poss_sil_at_max_speech: Whether to use the maximum possible silence at
          max_speech_duration_s or not. If not, the last silence is used.
    """

    threshold: float = 0.5
    neg_threshold: float = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    min_silence_at_max_speech: int = 98
    use_max_poss_sil_at_max_speech: bool = True


def compute_speech_probs(
    audio: np.ndarray,
    vad_options: VadOptions | None = None,
    sampling_rate: int = 16000,
    device_index: int = 0,
    progress: bool = False,
    **kwargs,
) -> np.ndarray:
    """Run Silero VAD and return one speech probability per 512-sample window.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      sampling_rate: Sampling rate of the audio.
      device_index: CUDA device index for GPU-accelerated VAD (if available).
      progress: Show a tqdm progress bar while the VAD runs (useful for long files).
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      A 1-D float array of speech probabilities, one per window.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    model = get_vad_model(device_index)
    return model(audio, progress=progress)


def segments_from_speech_probs(
    speech_probs: np.ndarray,
    audio_length_samples: int,
    vad_options: VadOptions,
    sampling_rate: int = 16000,
) -> list[dict]:
    """Convert VAD per-window probabilities into padded speech segments.

    Args:
      speech_probs: One speech probability per 512-sample window (see
        compute_speech_probs).
      audio_length_samples: Length of the source audio in samples.
      vad_options: Options for VAD processing.
      sampling_rate: Sampling rate of the audio.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    threshold = vad_options.threshold
    neg_threshold = vad_options.neg_threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512
    speech_pad_ms = vad_options.speech_pad_ms
    min_silence_at_max_speech = vad_options.min_silence_at_max_speech
    use_max_poss_sil_at_max_speech = vad_options.use_max_poss_sil_at_max_speech

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    triggered = False
    speeches = []
    current_speech = {}
    possible_ends = []

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (
                not use_max_poss_sil_at_max_speech
                and sil_dur_now > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    return speeches


class IncrementalSpeechSegmenter:
    """Incremental version of ``segments_from_speech_probs``.

    Drives the Silero VAD segmentizer with per-block probabilities instead of
    the whole array at once, so a streaming pipeline can discover newly-final
    speech segments without re-running the O(n) state machine over every window
    seen so far.  ``feed`` returns the segments that can no longer change
    (their end — and the pad-split of the boundary before them — is final once
    the following segment appears); ``finish`` closes the trailing segment(s)
    at the end of the audio.  Output is byte-for-byte identical to
    ``segments_from_speech_probs`` on the same probabilities.
    """

    def __init__(self, vad_options, sampling_rate=16000):
        window_size_samples = 512

        self.sampling_rate = sampling_rate
        self.threshold = vad_options.threshold
        self.neg_threshold = (
            vad_options.neg_threshold
            if vad_options.neg_threshold is not None
            else max(vad_options.threshold - 0.15, 0.01)
        )
        self.min_speech_samples = sampling_rate * vad_options.min_speech_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * vad_options.speech_pad_ms / 1000
        self.max_speech_samples = (
            sampling_rate * vad_options.max_speech_duration_s
            - window_size_samples
            - 2 * self.speech_pad_samples
        )
        self.min_silence_samples = sampling_rate * vad_options.min_silence_duration_ms / 1000
        self.min_silence_samples_at_max_speech = (
            sampling_rate * vad_options.min_silence_at_max_speech / 1000
        )
        self.use_max_poss_sil_at_max_speech = vad_options.use_max_poss_sil_at_max_speech

        # State machine state (mirrors the locals of segments_from_speech_probs).
        self._triggered = False
        self._current_speech = {}
        self._possible_ends = []
        self._temp_end = 0
        self._prev_end = 0
        self._next_start = 0
        self._window_index = 0
        self._speeches = []  # closed segments awaiting finalization (kept <= 1)
        self._first_finalized = True
        self._done = False

    def _append(self, check_min_speech, audio_length):
        """Append ``_current_speech`` (already closed) and finalize the segment
        before it, replicating the pad-split post-pass of
        ``segments_from_speech_probs``.  Returns newly-finalized segments."""
        out = []
        cs = self._current_speech
        self._current_speech = {}
        if not check_min_speech or (cs["end"] - cs["start"]) > self.min_speech_samples:
            self._speeches.append(cs)
            if len(self._speeches) >= 2:
                prev = self._speeches.pop(-2)
                nxt = self._speeches[-1]
                silence_duration = nxt["start"] - prev["end"]
                if silence_duration < 2 * self.speech_pad_samples:
                    prev["end"] += int(silence_duration // 2)
                    nxt["start"] = int(max(0, nxt["start"] - silence_duration // 2))
                else:
                    prev["end"] = int(min(audio_length, prev["end"] + self.speech_pad_samples))
                    nxt["start"] = int(max(0, nxt["start"] - self.speech_pad_samples))
                if self._first_finalized:
                    prev["start"] = int(max(0, prev["start"] - self.speech_pad_samples))
                    self._first_finalized = False
                out.append(prev)
        return out

    def feed(self, speech_probs, audio_length=float("inf")):
        """Process new per-window probabilities; return newly-finalized segments.

        ``audio_length`` is only consulted for the pad-split of a segment whose
        end lies within a padding window of the end of the file, so infinity is
        correct during streaming (the real total is used by ``finish``)."""
        out = []
        window_size_samples = 512
        for prob in speech_probs:
            cur_sample = window_size_samples * self._window_index
            self._window_index += 1

            if (prob >= self.threshold) and self._temp_end:
                sil_dur = cur_sample - self._temp_end
                if sil_dur > self.min_silence_samples_at_max_speech:
                    self._possible_ends.append((self._temp_end, sil_dur))
                self._temp_end = 0
                if self._next_start < self._prev_end:
                    self._next_start = cur_sample

            if (prob >= self.threshold) and not self._triggered:
                self._triggered = True
                self._current_speech["start"] = cur_sample
                continue

            if self._triggered and (
                cur_sample - self._current_speech["start"] > self.max_speech_samples
            ):
                if self.use_max_poss_sil_at_max_speech and self._possible_ends:
                    self._prev_end, dur = max(self._possible_ends, key=lambda x: x[1])
                    self._current_speech["end"] = self._prev_end
                    out.extend(self._append(False, audio_length))
                    self._next_start = self._prev_end + dur

                    if self._next_start < self._prev_end + cur_sample:
                        self._current_speech["start"] = self._next_start
                    else:
                        self._triggered = False
                    self._prev_end = self._next_start = self._temp_end = 0
                    self._possible_ends = []
                else:
                    if self._prev_end:
                        self._current_speech["end"] = self._prev_end
                        out.extend(self._append(False, audio_length))
                        if self._next_start < self._prev_end:
                            self._triggered = False
                        else:
                            self._current_speech["start"] = self._next_start
                        self._prev_end = self._next_start = self._temp_end = 0
                        self._possible_ends = []
                    else:
                        self._current_speech["end"] = cur_sample
                        out.extend(self._append(False, audio_length))
                        self._prev_end = self._next_start = self._temp_end = 0
                        self._triggered = False
                        self._possible_ends = []
                        continue

            if (prob < self.neg_threshold) and self._triggered:
                if not self._temp_end:
                    self._temp_end = cur_sample
                sil_dur_now = cur_sample - self._temp_end

                if (
                    not self.use_max_poss_sil_at_max_speech
                    and sil_dur_now > self.min_silence_samples_at_max_speech
                ):
                    self._prev_end = self._temp_end

                if sil_dur_now < self.min_silence_samples:
                    continue
                else:
                    self._current_speech["end"] = self._temp_end
                    out.extend(self._append(True, audio_length))
                    self._prev_end = self._next_start = self._temp_end = 0
                    self._triggered = False
                    self._possible_ends = []
                    continue
        return out

    def finish(self, audio_length_samples):
        """Close the trailing open segment (if any) and emit the remaining
        finalized segments."""
        if self._done:
            return []
        self._done = True
        out = []
        if (
            self._current_speech
            and (audio_length_samples - self._current_speech["start"]) > self.min_speech_samples
        ):
            self._current_speech["end"] = audio_length_samples
            out.extend(self._append(False, audio_length_samples))
        if self._speeches:
            last = self._speeches.pop()
            if self._first_finalized:
                last["start"] = int(max(0, last["start"] - self.speech_pad_samples))
                self._first_finalized = False
            last["end"] = int(min(audio_length_samples, last["end"] + self.speech_pad_samples))
            out.append(last)
        return out


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: VadOptions | None = None,
    sampling_rate: int = 16000,
    device_index: int = 0,
    progress: bool = False,
    **kwargs,
) -> list[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      sampling_rate: Sampling rate of the audio.
      device_index: CUDA device index for GPU-accelerated VAD (if available).
      progress: Show a tqdm progress bar while the VAD runs (useful for long files).
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    speech_probs = compute_speech_probs(
        audio,
        vad_options=vad_options,
        sampling_rate=sampling_rate,
        device_index=device_index,
        progress=progress,
        **kwargs,
    )
    return segments_from_speech_probs(
        speech_probs, len(audio), vad_options, sampling_rate=sampling_rate
    )


class StreamingVad:
    """Decodes audio and runs Silero VAD concurrently.

    A producer thread decodes the input in chunks while the caller consumes
    complete VAD blocks (block_windows windows each) as soon as their samples
    are available.  VAD blocks are independent (each window is inferred with a
    zero initial state), so processing them incrementally is numerically
    identical to processing the whole file at once — but the decode and the
    VAD work overlap instead of running back-to-back.

    Usage:

        sv = StreamingVad(vad_options, sampling_rate)
        sv.start(input_file)
        while sv.process_next_block():   # caller drives VAD inference
            pass
        audio, segments = sv.finish()    # full audio + speech segments

    or simply call ``run(input_file)`` which does all of the above.
    """

    def __init__(
        self,
        vad_options: VadOptions,
        sampling_rate: int = 16000,
        progress: bool = False,
        device_index: int = 0,
        decode_chunk_samples: int = 500000,
    ):
        self.vad_options = vad_options
        self.sampling_rate = sampling_rate
        self.progress = progress
        self.decode_chunk_samples = decode_chunk_samples

        self.model = get_vad_model(device_index)
        self.window_size = 512
        self.context = 64
        self.block_windows = SileroVADModel.block_windows

        # Shared state between the decode thread and the consumer.
        self._parts: list[np.ndarray] = []
        self._total_samples = 0
        self._done = False
        self._error: BaseException | None = None
        self._probs: list[np.ndarray] = []
        self._next_block = 0  # next block start, in window units
        self._vad_done = False
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None

        # Running blake2b over the decoded audio so callers that only need a
        # stable fingerprint of the waveform (e.g. a VAD cache key) never have
        # to materialize the full concatenated buffer.
        import hashlib as _hashlib

        self._audio_fingerprint = _hashlib.blake2b(digest_size=16)

        # Wall-clock work accounting (used by benchmarks / profiling).
        self.decode_time = 0.0
        self.vad_time = 0.0

    # --- Producer side (decode thread) -------------------------------------

    def _producer(self, input_file):
        from faster_whisper.audio import decode_audio_chunks

        t0 = time.perf_counter()
        try:
            for chunk in decode_audio_chunks(input_file, sampling_rate=self.sampling_rate):
                self._audio_fingerprint.update(chunk.view(np.uint8))
                with self._cond:
                    self._parts.append(chunk)
                    self._total_samples += chunk.shape[0]
                    self._cond.notify_all()
        except BaseException as exc:  # propagate any decode failure
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        else:
            with self._cond:
                self._done = True
                self._cond.notify_all()
        finally:
            self.decode_time = time.perf_counter() - t0

    def start(self, input_file):
        self._thread = threading.Thread(
            target=self._producer, args=(input_file,), name="decode-vad"
        )
        self._thread.start()

    def join(self):
        if self._thread is not None:
            self._thread.join()

    # --- Consumer side -----------------------------------------------------

    def _read(self, offset: int, count: int) -> np.ndarray:
        """Copy ``count`` samples starting at ``offset`` from the decoded parts."""
        result = np.empty(count, dtype=np.float32)
        cursor = 0
        part_start = 0
        for part in self._parts:
            part_end = part_start + part.shape[0]
            if part_end > offset:
                local = max(0, offset - part_start)
                take = min(part.shape[0] - local, count - cursor)
                if take > 0:
                    result[cursor : cursor + take] = part[local : local + take]
                    cursor += take
                    if cursor >= count:
                        return result
            part_start = part_end
            if part_start > offset + count:
                break
        raise ValueError(f"VAD read past end of audio (offset={offset}, count={count})")

    def _wait_samples(self, needed: int):
        with self._cond:
            while not self._done and self._error is None and self._total_samples < needed:
                self._cond.wait()
            if self._error is not None:
                raise self._error
            return self._total_samples

    def set_error(self, exc):
        with self._cond:
            self._error = exc
            self._cond.notify_all()

    def wait_audio(self, n_samples: int) -> int:
        """Block until at least ``n_samples`` are decoded; return the total."""
        return self._wait_samples(n_samples)

    def num_segments_windows(self) -> int:
        """Total VAD windows (including the trailing padding window).

        Only meaningful once decoding has finished."""
        num_samples = self.window_size
        n = self._total_samples
        pad_n = num_samples - n % num_samples
        return (n + pad_n) // num_samples

    def wait_processed_blocks(self, n_blocks: int):
        """Block until ``n_blocks`` VAD blocks have been processed."""
        with self._cond:
            while len(self._probs) < n_blocks and not self._vad_done and self._error is None:
                self._cond.wait()
            if self._error is not None:
                raise self._error

    def probs_for_windows(self, win_start: int, win_end: int) -> np.ndarray:
        """Return the speech probabilities for windows [win_start, win_end).

        Blocks are independent, so this can be called before the whole file is
        decoded as long as the covering blocks have been processed.
        """
        block = self.block_windows
        b0 = win_start // block
        b1 = (win_end - 1) // block
        self.wait_processed_blocks(b1 + 1)
        if b1 >= len(self._probs):
            raise ValueError(
                f"probs_for_windows: window {win_end} beyond processed "
                f"blocks (have {len(self._probs)})"
            )
        parts = []
        for bi in range(b0, b1 + 1):
            p = self._probs[bi]
            s = max(win_start, bi * block) - bi * block
            e = min(win_end, (bi + 1) * block) - bi * block
            parts.append(p[s:e])
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def _process_block_range(self, next_start: int, count: int):
        """Infer VAD block for windows [next_start, next_start+count) and store
        its probabilities.  ``count < block_windows`` uses the file-end padding
        semantics of SileroVADModel (only valid for the trailing block)."""
        num_samples = self.window_size
        context = self.context
        block = self.block_windows
        n = self._total_samples
        t0 = time.perf_counter()
        if count == block:
            # Row 0's context is the previous window's tail (zeros for block 0);
            # the read only covers the leading context + the block's own windows,
            # so it stays within decoded audio even on exactly-aligned files.
            if next_start == 0:
                block_audio = np.concatenate(
                    (
                        np.zeros(context, dtype=np.float32),
                        self._read(0, count * num_samples),
                    )
                )
            else:
                block_audio = self._read(
                    next_start * num_samples - context,
                    count * num_samples + context,
                )
            b = np.empty((count, context + num_samples), dtype=np.float32)
            self._fill_block_parts(b, next_start, block_audio, num_samples, context)
        else:
            # Trailing block: reuse the exact original fill logic on the
            # concatenated audio so padding matches byte-for-byte.
            full_segs = n // num_samples
            remainder = n % num_samples
            full_audio = np.concatenate(self._parts)
            b = np.empty((count, context + num_samples), dtype=np.float32)
            SileroVADModel._fill_block(
                b,
                next_start,
                full_audio,
                full_segs,
                remainder,
                num_samples,
                context,
            )
        prob = self.model._infer_block(b)
        self.vad_time += time.perf_counter() - t0
        with self._cond:
            self._probs.append(prob)
            self._next_block = next_start + count
            self._cond.notify_all()

    def process_next_block(self) -> bool:
        """Process the next complete VAD block if its audio is available.

        Returns False when there is nothing left to do (decode finished and all
        full blocks processed).  The last (partial) block is handled by
        ``drain_remaining_blocks``.
        """
        num_samples = self.window_size
        block = self.block_windows
        next_start = self._next_block

        needed = (next_start + block) * num_samples
        if self._wait_samples(needed) < needed:
            return False  # decode finished before a full block was available
        self._process_block_range(next_start, block)
        return True

    def drain_remaining_blocks(self):
        """Process the trailing partial block (file-end padding), if any.

        Runs once, after ``process_next_block`` has returned False (i.e. after
        the decode thread finished)."""
        if self._vad_done:
            return
        num_samples = self.window_size
        block = self.block_windows
        n = self._total_samples
        pad_n = num_samples - n % num_samples
        num_segments = (n + pad_n) // num_samples

        while self._next_block < num_segments:
            next_start = self._next_block
            count = min(block, num_segments - next_start)
            self._process_block_range(next_start, count)
        with self._cond:
            self._vad_done = True
            self._cond.notify_all()

    def _fill_block_parts(self, block, start, block_audio, num_samples, context):
        """Fill a VAD block from ``block_audio``.

        ``block_audio`` holds `[leading context (unless start == 0) | windows
        [start, start+rows) | trailing context]` laid out exactly like the
        single-buffer path: row r's context is the last `context` samples of
        window start+r-1, followed by the `num_samples` samples of window
        start+r.
        """
        rows = block.shape[0]
        for r in range(rows):
            base = r * num_samples
            if start == 0 and r == 0:
                block[0, :context] = 0.0
            else:
                block[r, :context] = block_audio[base : base + context]
            block[r, context:] = block_audio[base + context : base + context + num_samples]

    def finish(self, need_audio=True):
        """Finish VAD processing and return (full_audio, speech_segments).

        ``need_audio=False`` skips concatenating the decoded waveform (the
        caller only wants the speech segments); ``full_audio`` is then ``None``.
        """
        self.join()
        self.drain_remaining_blocks()
        n = self._total_samples
        speech_probs = np.concatenate(self._probs, axis=0)
        full_audio = np.concatenate(self._parts) if need_audio else None
        segments = segments_from_speech_probs(
            speech_probs, n, self.vad_options, sampling_rate=self.sampling_rate
        )
        return full_audio, segments

    def audio_fingerprint(self) -> bytes:
        """Stable 16-byte blake2b of the decoded audio (hash of ``full_audio``).

        Only valid once decoding has finished (``join``/``finish``).  Equivalent
        to ``hashlib.blake2b(full_audio.tobytes(), digest_size=16).digest()``
        but computed incrementally during decode, so it never materializes the
        whole waveform."""
        return self._audio_fingerprint.digest()

    def run(self, input_file):
        """Decode + VAD concurrently; returns (full_audio, speech_segments)."""
        self.start(input_file)
        while self.process_next_block():
            pass
        return self.finish()


def collect_chunks(
    audio: np.ndarray,
    chunks: list[dict],
    sampling_rate: int = 16000,
    max_duration: float = float("inf"),
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """This function merges the chunks of audio into chunks of max_duration (s) length."""
    if not chunks:
        chunk_metadata = {
            "offset": 0,
            "duration": 0,
            "segments": [],
        }
        return [np.array([], dtype=np.float32)], [chunk_metadata]

    audio_chunks = []
    chunks_metadata = []

    current_segments = []
    current_duration = 0
    total_duration = 0
    # Collect slices into a list; concatenate once per output chunk instead of
    # once per input chunk.  The naive approach copies O(n²) samples in total
    # because every np.concatenate allocates a fresh array and copies all prior
    # data.  Deferring to a single np.concatenate reduces that to O(n).
    current_slices: list[np.ndarray] = []

    for chunk in chunks:
        if current_duration + chunk["end"] - chunk["start"] > max_duration * sampling_rate:
            audio_chunks.append(
                np.concatenate(current_slices) if current_slices else np.array([], dtype=np.float32)
            )
            chunk_metadata = {
                "offset": total_duration / sampling_rate,
                "duration": current_duration / sampling_rate,
                "segments": current_segments,
            }
            total_duration += current_duration
            chunks_metadata.append(chunk_metadata)

            current_segments = [chunk]
            current_slices = [audio[chunk["start"] : chunk["end"]]]
            current_duration = chunk["end"] - chunk["start"]
        else:
            current_segments.append(chunk)
            current_slices.append(audio[chunk["start"] : chunk["end"]])
            current_duration += chunk["end"] - chunk["start"]

    audio_chunks.append(
        np.concatenate(current_slices) if current_slices else np.array([], dtype=np.float32)
    )

    chunk_metadata = {
        "offset": total_duration / sampling_rate,
        "duration": current_duration / sampling_rate,
        "segments": current_segments,
    }
    chunks_metadata.append(chunk_metadata)
    return audio_chunks, chunks_metadata


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: list[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: int | None = None,
        is_end: bool = False,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time, is_end)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float, is_end: bool = False) -> int:
        sample = int(time * self.sampling_rate)

        if is_end:
            # bisect_left finds the leftmost position where sample could be
            # inserted to keep the list sorted.  If chunk_end_sample[idx] ==
            # sample the sample lands exactly on a chunk boundary, which is the
            # condition the original code tested with a linear `.index()` call.
            idx = bisect.bisect_left(self.chunk_end_sample, sample)
            if idx < len(self.chunk_end_sample) and self.chunk_end_sample[idx] == sample:
                return idx

        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )


@functools.lru_cache
def get_vad_model(device_index: int = 0):
    """Returns the VAD model instance, preferring CUDA when available."""
    path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
    return SileroVADModel(path, device_index=device_index)


class SileroVADModel:
    def __init__(self, path, device_index: int = 0):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError("Applying the VAD filter requires the onnxruntime package") from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        # The VAD runs as one large batched inference, not many small
        # sequential calls, so a few intra-op threads and the memory arena
        # are a clear win instead of contention.
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.enable_cpu_mem_arena = True
        opts.log_severity_level = 4
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_provider = (
            "CUDAExecutionProvider",
            {"device_id": device_index},
        )
        providers = (
            [cuda_provider, "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = onnxruntime.InferenceSession(
            path,
            providers=providers,
            sess_options=opts,
        )

    # Windows per inference block. Each block is block_windows x 576 float32
    # (~4.6 MB at 2000), so peak memory stays flat no matter how long the audio
    # is. Previously the whole file was materialized as one (num_windows, 576)
    # buffer plus the matching ONNX input/output tensors — ~700 MB for a 2.7 h
    # file and multiple GB for longer ones, which could exhaust RAM.
    # Blocking is numerically safe: every window is inferred independently
    # (each carries its own 64-sample context and a zero initial state), so
    # splitting the batch does not change any window's output.
    block_windows = 2000

    def __call__(
        self,
        audio: np.ndarray,
        num_samples: int = 512,
        context_size_samples: int = 64,
        progress: bool = False,
    ):
        assert audio.ndim == 1, "Input should be a 1D array"

        n = len(audio)
        # Preserve original padding semantics: always add (num_samples - n % num_samples)
        # zeros at the end, which appends a full extra window when audio is already aligned.
        pad_n = num_samples - n % num_samples
        num_segments = (n + pad_n) // num_samples
        full_segs = n // num_samples  # segments fully covered by audio (no padding needed)
        remainder = n % num_samples

        outputs = []
        # Silero operates at 16 kHz: num_samples samples per window.
        pbar = tqdm(
            total=round(num_segments * num_samples / 16000.0, 1),
            desc="VAD",
            unit="s",
            disable=not progress,
            mininterval=0.5,
        )
        try:
            for s in range(0, num_segments, self.block_windows):
                e = min(s + self.block_windows, num_segments)
                block = self._make_block(
                    s,
                    e,
                    audio,
                    full_segs,
                    remainder,
                    num_samples,
                    context_size_samples,
                )
                outputs.append(self._infer_block(block))
                pbar.update(round((e - s) * num_samples / 16000.0, 1))
        finally:
            pbar.close()

        return np.concatenate(outputs, axis=0)

    def _make_block(
        self,
        start,
        end,
        audio,
        full_segs,
        remainder,
        num_samples,
        context_size_samples,
    ):
        block = np.empty((end - start, context_size_samples + num_samples), dtype=np.float32)
        self._fill_block(
            block,
            start,
            audio,
            full_segs,
            remainder,
            num_samples,
            context_size_samples,
        )
        return block

    def _infer_block(self, block):
        state = np.zeros((2, block.shape[0], 128), dtype="float32")
        return self.session.run(
            None,
            {
                "input": block,
                "state": state,
                "sr": np.array(16000, dtype="int64"),
            },
        )[0]

    @staticmethod
    def _fill_block(block, start, audio, full_segs, remainder, num_samples, context):
        """Fill block rows for global window indices [start, start+len(block))
        with [context | window] samples, exactly matching the layout the old
        single-buffer implementation produced."""
        rows = block.shape[0]
        end = start + rows

        # Audio region of fully-covered windows.
        full_end = min(end, full_segs)
        if start < full_end:
            r = full_end - start
            block[:r, context:] = audio[start * num_samples : full_end * num_samples].reshape(
                r, num_samples
            )

        # Trailing window (partial tail, or all zeros when audio is aligned).
        if end > full_segs:
            row = full_segs - start
            if 0 <= row < rows:
                block[row, context:] = 0.0
                if remainder:
                    block[row, context : context + remainder] = audio[full_segs * num_samples :]

        # Context region: window i gets the last `context` samples of window i-1.
        if start == 0:
            block[0, :context] = 0.0
        i0 = max(start, 1)
        ctx_full_end = min(end, full_segs)
        if i0 < ctx_full_end:
            cnt = ctx_full_end - i0
            src = audio[i0 * num_samples - context : i0 * num_samples - context + cnt * num_samples]
            block[i0 - start : ctx_full_end - start, :context] = src.reshape(cnt, num_samples)[
                :, :context
            ]
        # Context of the trailing window: the last `context` samples of the audio.
        if end > full_segs and full_segs > 0:
            row = full_segs - start
            if 0 <= row < rows:
                block[row, :context] = audio[
                    full_segs * num_samples - context : full_segs * num_samples
                ]
