"""We use the PyAV library to decode the audio: https://github.com/PyAV-Org/PyAV

The advantage of PyAV is that it bundles the FFmpeg libraries so there is no additional
system dependencies. FFmpeg does not need to be installed on the system.

However, the API is quite low-level so we need to manipulate audio frames directly.
"""

import gc
import itertools
from typing import BinaryIO

import av
import numpy as np


def decode_audio(
    input_file: str | BinaryIO,
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    chunks = list(
        decode_audio_chunks(input_file, sampling_rate=sampling_rate, split_stereo=split_stereo)
    )
    if not chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(chunks)

    if split_stereo:
        # fltp stereo is planar: all left samples first, then all right.
        half = len(audio) // 2
        return audio[:half], audio[half:]

    return audio


def decode_audio_chunks(
    input_file: str | BinaryIO,
    sampling_rate: int = 16000,
    split_stereo: bool = False,
    chunk_samples: int = 500000,
):
    """Decodes the audio in chunks, yielding one float32 Numpy array per chunk.

    Yields chunks of roughly ``chunk_samples`` samples (16 kHz mono, ~31 s each).
    This lets a consumer (e.g. a streaming VAD) process audio while it is still
    being decoded.  The concatenation of all yielded arrays is byte-identical to
    the array returned by :func:`decode_audio`.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels (planar layout).
      chunk_samples: Approximate number of samples per yielded chunk.
    """
    resampler = av.audio.resampler.AudioResampler(
        # fltp: decode straight to float32 planar, skipping the s16
        # intermediate and the extra astype/scale pass over the whole buffer.
        format="fltp",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    try:
        with av.open(input_file, mode="r", metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, chunk_samples)
            frames = _resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                if array.dtype != np.float32:
                    # Fallback for any future resampler format change.
                    array = array.astype(np.float32) / 32768.0
                yield array.reshape(-1)
    finally:
        # It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        # https://github.com/SYSTRAN/faster-whisper/issues/390
        del resampler
        gc.collect(0)


def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
    """
    Pad or trim the Mel features array to 3000, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array
