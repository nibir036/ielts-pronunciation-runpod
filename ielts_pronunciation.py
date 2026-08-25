"""
Pronunciation scoring for real IELTS speaking test audio (no known script).

Unlike gop_scoring.py's single-clip demo (which needs a known reference_text),
real IELTS answers are spontaneous. This pipeline:

  1. Transcribes the full response with Whisper, using SEGMENT-level
     timestamps (reliable) rather than word-level timestamps (heuristic
     and known to be less accurate — see huggingface/transformers#25605,
     #36228). Segments are natural sentence/utterance chunks.
  2. For each segment, slices that piece of audio out and runs it through
     the same forced-alignment GOP scorer validated in gop_scoring.py,
     using the segment's OWN transcript as the reference text.
  3. Groups the resulting per-phoneme scores back into per-word scores
     (phonemizer gives us word boundaries if we keep them instead of
     stripping, unlike the single-word demo).
  4. Aggregates into an overall response-level score plus a flagged list
     of the lowest-scoring words, which is what's actually useful to show
     a learner ("these specific words were unclear") rather than one
     opaque number.

Known limitation: this is SELF-referential — the reference text comes from
what Whisper thinks was said, not a known target. If pronunciation is bad
enough that Whisper mishears the word entirely, that instance won't be
flagged (the "reference" silently absorbs the error). This is an inherent
ceiling of any ASR-derived-reference GOP approach on free speech, not a bug
to fix here.

Install:
    pip install transformers torch pydub phonemizer
    system: espeak-ng, ffmpeg
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pydub import AudioSegment
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline as hf_pipeline,
)
from phonemizer import phonemize
from phonemizer.separator import Separator

PHONEME_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ASR_MODEL_ID = "openai/whisper-small"  # swap for "openai/whisper-large-v3"
                                        # for better accented-speech accuracy
                                        # at higher compute cost
TARGET_SR = 16000
MAX_SEGMENT_SECONDS = 30  # per-segment cap (same reasoning as gop_scoring.py);
                           # whole responses can be minutes long, but each
                           # sentence-level segment should be well under this

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dtype = torch.float16 if device == "cuda" else torch.float32

_phoneme_processor = None
_phoneme_model = None
_asr_pipeline = None


def _get_phoneme_model():
    global _phoneme_processor, _phoneme_model
    if _phoneme_model is None:
        _phoneme_processor = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL_ID)
        _phoneme_model = (
            Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL_ID, torch_dtype=model_dtype)
            .to(device)
            .eval()
        )
    return _phoneme_processor, _phoneme_model


def _get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        _asr_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            torch_dtype=model_dtype,
            device=device,
        )
    return _asr_pipeline


def load_audio(path: str) -> torch.Tensor:
    """Same pydub-based loader as gop_scoring.py — handles m4a/mp3/wav/etc."""
    audio_seg = AudioSegment.from_file(path)
    audio_seg = audio_seg.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    max_val = float(1 << (8 * audio_seg.sample_width - 1))
    samples = samples / max_val
    return torch.from_numpy(samples)


def transcribe_segments(waveform: torch.Tensor) -> list[dict]:
    """Whisper transcription with SEGMENT-level timestamps (sentence-ish
    chunks), which are far more reliable than Whisper's word-level
    timestamp heuristic. Returns [{"text": ..., "start": s, "end": e}, ...]."""
    asr = _get_asr_pipeline()
    result = asr(
        {"array": waveform.numpy(), "sampling_rate": TARGET_SR},
        return_timestamps=True,
        generate_kwargs={"language": "en"},
    )
    segments = []
    for chunk in result.get("chunks", []):
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        if not text or start is None or end is None:
            continue
        segments.append({"text": text, "start": start, "end": end})
    return segments


def text_to_phonemes_by_word(text: str) -> list[list[str]]:
    """Like gop_scoring.text_to_phonemes, but keeps word boundaries instead
    of flattening — returns one phoneme list PER WORD, so per-phoneme GOP
    scores can be grouped back into per-word scores afterward."""
    ph_string = phonemize(
        text, language="en-us", backend="espeak",
        separator=Separator(phone=" ", word="| ", syllable=""),
        strip=True, preserve_punctuation=False, with_stress=False,
    )
    words = [w.strip() for w in ph_string.split("|") if w.strip()]
    return [w.split() for w in words]


def score_segment(waveform: torch.Tensor, text: str) -> dict:
    """Runs forced-alignment GOP scoring on one short audio segment against
    its own transcript, and groups the result by word. Same core algorithm
    as gop_scoring.gop_score, extended with word grouping."""
    duration_seconds = waveform.shape[-1] / TARGET_SR
    if duration_seconds > MAX_SEGMENT_SECONDS:
        return {"error": f"segment is {duration_seconds:.1f}s, exceeds cap; split further upstream"}

    if waveform.abs().max() < 1e-4:
        return {"words": [], "note": "segment is silent"}

    processor, model = _get_phoneme_model()
    inputs = processor(waveform.numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = F.log_softmax(logits.float(), dim=-1).squeeze(0)

    words_phonemes = text_to_phonemes_by_word(text)
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id or 0

    # Flatten to one target sequence for a single forced_align call across
    # the whole segment, but remember each phoneme's (word_index, phoneme)
    # so we can regroup the per-phoneme results by word afterward.
    target_ids = []
    owner = []  # owner[i] = word index that target_ids[i] belongs to
    word_results = [{"word": None, "phonemes": []} for _ in words_phonemes]
    for wi, ph_list in enumerate(words_phonemes):
        word_results[wi]["word"] = "".join(ph_list)  # placeholder if we need it; real word text set below
        for ph in ph_list:
            ph_id = vocab.get(ph)
            if ph_id is None:
                word_results[wi]["phonemes"].append({"phoneme": ph, "score": None, "note": "OOV"})
                continue
            target_ids.append(ph_id)
            owner.append((wi, len(word_results[wi]["phonemes"])))
            word_results[wi]["phonemes"].append({"phoneme": ph, "score": None})  # filled below

    # Re-attach the ORIGINAL word text (not the phoneme placeholder) by
    # re-splitting the segment text the same way phonemizer word-split it.
    # (word count from phonemizer's "|" split should match text.split() for
    # plain English sentences; fall back gracefully if counts differ.)
    plain_words = text.split()
    if len(plain_words) == len(word_results):
        for wr, w in zip(word_results, plain_words):
            wr["word"] = w

    if not target_ids:
        return {"words": word_results, "error": "no alignable phonemes in this segment"}

    targets_tensor = torch.tensor([target_ids], dtype=torch.int32, device=device)
    try:
        aligned_tokens, _ = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), targets_tensor, blank=blank_id
        )
    except RuntimeError as e:
        return {"words": word_results, "error": f"forced alignment failed: {e}"}
    aligned_tokens = aligned_tokens.squeeze(0)
    top_logprob_per_frame = log_probs.max(dim=-1).values

    for target_pos, ph_id in enumerate(target_ids):
        wi, pi = owner[target_pos]
        frames = (aligned_tokens == ph_id).nonzero(as_tuple=True)[0]
        if len(frames) == 0:
            word_results[wi]["phonemes"][pi]["note"] = "no aligned frames"
            continue
        expected_logprob = log_probs[frames, ph_id].mean().item()
        top_logprob = top_logprob_per_frame[frames].mean().item()
        word_results[wi]["phonemes"][pi]["score"] = round(expected_logprob - top_logprob, 3)

    for wr in word_results:
        valid = [p["score"] for p in wr["phonemes"] if p.get("score") is not None]
        wr["word_score"] = round(sum(valid) / len(valid), 3) if valid else None

    return {"words": word_results}


def score_ielts_response(audio_path: str) -> dict:
    """Full pipeline entry point: transcribe an IELTS speaking answer and
    score pronunciation per word and per segment."""
    waveform = load_audio(audio_path)
    segments = transcribe_segments(waveform)

    scored_segments = []
    all_word_scores = []
    for seg in segments:
        start_sample = int(seg["start"] * TARGET_SR)
        end_sample = int(seg["end"] * TARGET_SR)
        segment_audio = waveform[start_sample:end_sample]

        result = score_segment(segment_audio, seg["text"])
        scored_segments.append({
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            **result,
        })
        for w in result.get("words", []):
            if w.get("word_score") is not None:
                all_word_scores.append((w["word"], w["word_score"], seg["start"]))

    overall_score = (
        round(sum(s for _, s, _ in all_word_scores) / len(all_word_scores), 3)
        if all_word_scores else None
    )
    # Lowest-scoring words = most likely pronunciation issues, most useful
    # thing to actually show a learner rather than one aggregate number.
    weakest_words = sorted(all_word_scores, key=lambda x: x[1])[:10]

    return {
        "full_transcript": " ".join(s["text"] for s in segments),
        "overall_pronunciation_score": overall_score,
        "weakest_words": [
            {"word": w, "score": s, "at_seconds": round(t, 1)} for w, s, t in weakest_words
        ],
        "segments": scored_segments,
    }


if __name__ == "__main__":
    import json
    result = score_ielts_response("speaking_response.wav")
    print(json.dumps(result, indent=2))
