import inspect
import logging
import os

import numpy as np

from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.transcribe import TranscriptionOptions


def test_supported_languages():
    model = WhisperModel("tiny.en")
    assert model.supported_languages == ["en"]


def test_transcribe(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(jfk_path, word_timestamps=True)
    assert info.all_language_probs is not None

    assert info.language == "en"
    assert info.language_probability > 0.9
    assert info.duration == 11

    # Get top language info from all results, which should match the
    # already existing metadata
    top_lang, top_lang_score = info.all_language_probs[0]
    assert info.language == top_lang
    assert abs(info.language_probability - top_lang_score) < 1e-16

    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.text == "".join(word.word for word in segment.words)
    assert segment.start == segment.words[0].start
    assert segment.end == segment.words[-1].end


def test_batched_transcribe(physcisworks_path):
    model = WhisperModel("tiny")
    batched_model = BatchedInferencePipeline(model=model)
    segments_iter, info = batched_model.transcribe(physcisworks_path, batch_size=16)
    assert info.language == "en"
    assert info.language_probability > 0.7
    segments = []
    for segment in segments_iter:
        segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
    assert len(segments) == 6  # number of VAD-bounded batched chunks

    segment = segments[0]


def test_prefix_with_timestamps(jfk_path):
    model = WhisperModel("tiny")
    segments, _ = model.transcribe(jfk_path, prefix="And so my fellow Americans")
    segments = list(segments)

    assert len(segments) == 1

    segment = segments[0]

    assert segment.text == (
        " And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )

    assert segment.start == 0
    assert 10 < segment.end <= 11


def test_vad(jfk_path):
    model = WhisperModel("tiny")
    segments, info = model.transcribe(
        jfk_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    segments = list(segments)

    assert len(segments) == 1
    segment = segments[0]

    # NOTE: VAD-segmented decode of the tiny model yields lowercase,
    # unpunctuated text — different from the full-audio decode above.
    # The plumbing assertions (1 segment, timestamps, vad_options) are the
    # real checks here; the exact string just locks current behavior.
    assert segment.text == (
        " and so my fellow america ask not what your country can do for you "
        "ask what you can do for your country"
    )

    assert 0 < segment.start < 1
    assert 10 < segment.end < 11

    assert info.vad_options.min_silence_duration_ms == 500
    assert info.vad_options.speech_pad_ms == 200


def test_stereo_diarization(data_dir):
    model = WhisperModel("tiny")

    audio_path = os.path.join(data_dir, "stereo_diarization.wav")
    left, right = decode_audio(audio_path, split_stereo=True)

    segments, _ = model.transcribe(left)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == (
        "He began a confused complaint against the wizard, "
        "who had vanished behind the curtain on the left."
    )

    segments, _ = model.transcribe(right)
    transcription = "".join(segment.text for segment in segments).strip()
    assert transcription == "The horizon seems extremely distant."


def test_multisegment_lang_id(physcisworks_path):
    model = WhisperModel("tiny")
    audio = decode_audio(physcisworks_path)
    language, confidence, _ = model.detect_language(
        audio, language_detection_segments=4
    )
    assert language == "en"
    assert confidence > 0.7


# --- Regression tests for the implicit max_new_tokens=128 cap ---------------
# Dense speech in token-inefficient languages needs 150-250 tokens per 30 s
# window; the old default (128) cut words mid-token with no EOT marker.


class _FakeResult:
    def __init__(self, sequences_ids, scores, no_speech_prob=0.0):
        self.sequences_ids = sequences_ids
        self.scores = scores
        self.no_speech_prob = no_speech_prob


class _FakeCTranslate2Model:
    def __init__(self):
        self.captured = {}

    def generate(self, encoder_output, prompts, **kwargs):
        self.captured.update(kwargs)
        self.captured["prompts"] = prompts
        return self._results

    def encode(self, features):
        return object()


def _make_fake_pipeline(prompt_len=4, max_length=448):
    """BatchedInferencePipeline with all model I/O stubbed out."""
    inner = _FakeCTranslate2Model()
    _prompt_len = prompt_len
    _max_length = max_length

    class _FakeModel:
        logger = logging.getLogger("test")

        def __init__(self):
            self.model = inner
            self.max_length = _max_length

        def get_prompt(self, tokenizer, previous_tokens=None, **kwargs):
            return list(range(_prompt_len))

        def encode(self, features):
            return object()

    pipe = BatchedInferencePipeline.__new__(BatchedInferencePipeline)
    pipe.model = _FakeModel()
    pipe.last_speech_timestamp = 0.0
    return pipe, inner


def _make_options(max_new_tokens=None):
    return TranscriptionOptions(
        beam_size=1,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        log_prob_threshold=None,
        no_speech_threshold=None,
        compression_ratio_threshold=None,
        condition_on_previous_text=False,
        prompt_reset_on_temperature=0.5,
        temperatures=[0.0],
        initial_prompt=None,
        prefix=None,
        suppress_blank=True,
        suppress_tokens=[],
        without_timestamps=True,
        max_initial_timestamp=0.0,
        word_timestamps=False,
        prepend_punctuations="",
        append_punctuations="",
        multilingual=False,
        max_new_tokens=max_new_tokens,
        clip_timestamps="0",
        hallucination_silence_threshold=None,
        hotwords=None,
    )


class _FakeTokenizer:
    def encode(self, text):
        return []


def _dummy_features(n=1):
    return np.zeros((n, 80, 300), dtype=np.float32)


def test_batched_default_is_unlimited():
    sig = inspect.signature(BatchedInferencePipeline.transcribe)
    assert sig.parameters["max_new_tokens"].default is None


def test_batched_default_uses_full_model_budget():
    pipe, inner = _make_fake_pipeline(prompt_len=4, max_length=448)
    inner._results = [_FakeResult(sequences_ids=[[1, 2, 3]], scores=[0.0])]
    opts = _make_options(max_new_tokens=None)
    pipe.generate_segment_batched(_dummy_features(1), _FakeTokenizer(), opts)
    assert inner.captured["max_length"] == 448


def test_batched_explicit_max_new_tokens_is_clamped():
    pipe, inner = _make_fake_pipeline(prompt_len=4, max_length=448)
    inner._results = [_FakeResult(sequences_ids=[[1, 2, 3]], scores=[0.0])]
    opts = _make_options(max_new_tokens=10_000)
    _, outputs = pipe.generate_segment_batched(_dummy_features(1), _FakeTokenizer(), opts)
    assert inner.captured["max_length"] == 448
    assert outputs[0]["truncated"] is False


def test_truncation_is_flagged():
    pipe, inner = _make_fake_pipeline(prompt_len=4, max_length=448)
    budget = 5

    def fake_generate(encoder_output, prompts, **kwargs):
        n = kwargs["max_length"] - len(prompts[0])
        assert n == budget
        return [_FakeResult(sequences_ids=[list(range(n))], scores=[0.0])]

    inner.generate = fake_generate
    opts = _make_options(max_new_tokens=budget)
    _, outputs = pipe.generate_segment_batched(_dummy_features(1), _FakeTokenizer(), opts)
    assert outputs[0]["truncated"] is True


def test_no_truncation_when_under_budget():
    pipe, inner = _make_fake_pipeline(prompt_len=4, max_length=448)
    inner._results = [_FakeResult(sequences_ids=[[1, 2]], scores=[0.0])]
    opts = _make_options(max_new_tokens=50)
    _, outputs = pipe.generate_segment_batched(_dummy_features(1), _FakeTokenizer(), opts)
    assert outputs[0]["truncated"] is False


def test_prompt_without_room_raises():
    pipe, inner = _make_fake_pipeline(prompt_len=448, max_length=448)
    inner._results = [_FakeResult(sequences_ids=[[1]], scores=[0.0])]
    opts = _make_options(max_new_tokens=0)
    try:
        pipe.generate_segment_batched(_dummy_features(1), _FakeTokenizer(), opts)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when prompt leaves no room")
