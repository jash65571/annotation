from pathlib import Path
import json

import opensmile
import soundfile as sf
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

ROOT = Path(__file__).resolve().parent
WHISPERX_JSON = ROOT / "output" / "VIDEO.json"
AUDIO_WAV = ROOT / "analysis" / "audio.wav"
EVIDENCE_JSON = ROOT / "analysis" / "manuscript_audio_evidence.json"
EMOTION_MODEL = "superb/wav2vec2-base-superb-er"


def build_whisperx_evidence(data):
    results = []

    for index, segment in enumerate(data.get("segments", [])):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start

        words = [
            w for w in segment.get("words", [])
            if "start" in w and "end" in w
        ]

        scores = [
            float(w["score"])
            for w in words
            if "score" in w
        ]

        low_confidence = [
            {
                "word": w.get("word", ""),
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "score": round(float(w["score"]), 3),
            }
            for w in words
            if w.get("score", 1.0) < 0.50
        ]

        gaps = []
        for left, right in zip(words, words[1:]):
            gap = float(right["start"]) - float(left["end"])
            if gap > 0:
                gaps.append(gap)

        word_count = len(words)
        raw_wpm = (word_count / duration) * 60 if duration else 0.0

        results.append({
            "segment": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration_sec": round(duration, 3),
            "text": segment.get("text", "").strip(),
            "word_count": word_count,
            "raw_wpm": round(raw_wpm, 1),
            "speed_measurement_reliable": duration >= 2.0 and word_count >= 5,
            "mean_word_confidence": (
                round(sum(scores) / len(scores), 3) if scores else None
            ),
            "lowest_word_confidence": (
                round(min(scores), 3) if scores else None
            ),
            "low_confidence_words": low_confidence,
            "largest_between_word_gap_sec": (
                round(max(gaps), 3) if gaps else 0.0
            ),
        })

    return results


def semitone_to_hz(value):
    return 27.5 * (2.0 ** (value / 12.0))


def build_acoustic_evidence(data, audio, sr):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    results = []

    for index, segment in enumerate(data.get("segments", [])):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start

        clip = audio[
            max(0, int(start * sr)):
            min(len(audio), int(end * sr))
        ]

        features = smile.process_signal(clip, sr).iloc[0]

        f0_st = float(
            features["F0semitoneFrom27.5Hz_sma3nz_amean"]
        )

        results.append({
            "segment": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "mean_pitch_semitones": round(f0_st, 3),
            "mean_pitch_hz": round(semitone_to_hz(f0_st), 1),
            "pitch_variation_norm": round(
                float(features["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"]),
                4,
            ),
            "pitch_range_semitones": round(
                float(features["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"]),
                3,
            ),
            "mean_loudness": round(
                float(features["loudness_sma3_amean"]),
                4,
            ),
            "loudness_variation_norm": round(
                float(features["loudness_sma3_stddevNorm"]),
                4,
            ),
            "equivalent_sound_level_db": round(
                float(features["equivalentSoundLevel_dBp"]),
                2,
            ),
            "jitter_local": round(
                float(features["jitterLocal_sma3nz_amean"]),
                5,
            ),
            "shimmer_local_db": round(
                float(features["shimmerLocaldB_sma3nz_amean"]),
                4,
            ),
            "hnr_db": round(
                float(features["HNRdBACF_sma3nz_amean"]),
                3,
            ),
            "loudness_peaks_per_sec": round(
                float(features["loudnessPeaksPerSec"]),
                3,
            ),
            "voiced_segments_per_sec": round(
                float(features["VoicedSegmentsPerSec"]),
                3,
            ),
            "mean_voiced_segment_sec": round(
                float(features["MeanVoicedSegmentLengthSec"]),
                3,
            ),
            "mean_unvoiced_segment_sec": round(
                float(features["MeanUnvoicedSegmentLength"]),
                3,
            ),
            "voice_quality_measurement_reliable": duration >= 1.5,
        })

    return results


def build_emotion_evidence(data, audio, sr):
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(EMOTION_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.eval()

    results = []

    for index, segment in enumerate(data.get("segments", [])):
        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start

        clip = audio[
            max(0, int(start * sr)):
            min(len(audio), int(end * sr))
        ]

        inputs = extractor(
            clip,
            sampling_rate=sr,
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        scores = {
            model.config.id2label[i]: round(
                float(probabilities[i].item()),
                4,
            )
            for i in range(len(probabilities))
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1]

        results.append({
            "segment": index,
            "emotion_probabilities": scores,
            "top_emotion_candidate": top_label,
            "top_emotion_score": top_score,
            "second_emotion_candidate": second_label,
            "second_emotion_score": second_score,
            "candidate_margin": round(top_score - second_score, 4),
            "emotion_measurement_reliable": duration >= 2.0,
            "use_as_final_tone": False,
        })

    return results


def build_continuity_evidence(whisperx_segments, acoustic_segments):
    results = []

    for i in range(len(acoustic_segments) - 1):
        left_audio = acoustic_segments[i]
        right_audio = acoustic_segments[i + 1]
        left_text = whisperx_segments[i]
        right_text = whisperx_segments[i + 1]

        left_pitch = float(left_audio["mean_pitch_hz"])
        right_pitch = float(right_audio["mean_pitch_hz"])

        low_pitch = min(left_pitch, right_pitch)
        high_pitch = max(left_pitch, right_pitch)

        pitch_ratio = (
            high_pitch / low_pitch
            if low_pitch > 0
            else None
        )

        gap = float(right_text["start"]) - float(left_text["end"])

        large_pitch_jump = (
            pitch_ratio is not None
            and pitch_ratio >= 1.45
        )

        results.append({
            "boundary": f"{i}->{i + 1}",
            "left_segment": i,
            "right_segment": i + 1,
            "gap_sec": round(gap, 3),
            "left_pitch_hz": round(left_pitch, 1),
            "right_pitch_hz": round(right_pitch, 1),
            "pitch_ratio": (
                round(pitch_ratio, 2)
                if pitch_ratio is not None
                else None
            ),
            "large_pitch_discontinuity": large_pitch_jump,
            "possible_speaker_or_source_change": large_pitch_jump,
            "manual_review_required": large_pitch_jump,
            "automatic_speaker_change": False,
        })

    return results


def build_review_synthesis(
    whisperx_segments,
    acoustic_segments,
    emotion_segments,
    continuity_boundaries,
):
    continuity_by_segment = {}

    for boundary in continuity_boundaries:
        if boundary.get("manual_review_required"):
            left = boundary["left_segment"]
            right = boundary["right_segment"]

            continuity_by_segment.setdefault(left, []).append(
                boundary["boundary"]
            )
            continuity_by_segment.setdefault(right, []).append(
                boundary["boundary"]
            )

    results = []

    for i in range(len(whisperx_segments)):
        speech = whisperx_segments[i]
        acoustic = acoustic_segments[i]
        emotion = emotion_segments[i]

        reasons = []
        evidence_warnings = []

        low_words = speech.get("low_confidence_words", [])
        if low_words:
            reasons.append("low_confidence_transcript_word")

        if not speech.get("speed_measurement_reliable", False):
            evidence_warnings.append(
                "segment_too_short_for_reliable_speed_label"
            )

        if not acoustic.get("voice_quality_measurement_reliable", False):
            evidence_warnings.append(
                "segment_too_short_for_reliable_voice_quality"
            )

        emotion_reliable = emotion.get(
            "emotion_measurement_reliable",
            False,
        )
        emotion_margin = float(emotion.get("candidate_margin", 0.0))

        if not emotion_reliable:
            evidence_warnings.append(
                "segment_too_short_for_reliable_emotion"
            )

        if emotion_reliable and emotion_margin < 0.15:
            reasons.append("emotion_model_ambiguous")

        boundary_flags = continuity_by_segment.get(i, [])
        if boundary_flags:
            reasons.append(
                "possible_speaker_or_source_change_near_segment"
            )

        if boundary_flags:
            priority = "high"
        elif reasons:
            priority = "medium"
        else:
            priority = "normal"

        results.append({
            "segment": i,
            "start": speech["start"],
            "end": speech["end"],
            "text": speech["text"],
            "review_priority": priority,
            "manual_review_reasons": reasons,
            "evidence_warnings": evidence_warnings,
            "transcript_mean_confidence": speech.get(
                "mean_word_confidence"
            ),
            "low_confidence_words": low_words,
            "raw_wpm": speech.get("raw_wpm"),
            "speed_evidence_reliable": speech.get(
                "speed_measurement_reliable"
            ),
            "mean_pitch_hz": acoustic.get("mean_pitch_hz"),
            "pitch_range_semitones": acoustic.get(
                "pitch_range_semitones"
            ),
            "mean_loudness": acoustic.get("mean_loudness"),
            "hnr_db": acoustic.get("hnr_db"),
            "emotion_candidate": emotion.get(
                "top_emotion_candidate"
            ),
            "emotion_score": emotion.get(
                "top_emotion_score"
            ),
            "emotion_margin": emotion.get(
                "candidate_margin"
            ),
            "emotion_evidence_reliable": emotion_reliable,
            "near_flagged_boundary": boundary_flags,
            "automatic_tone_label": None,
            "automatic_speaker_identity": None,
            "requires_human_audio_check": bool(
                reasons or evidence_warnings
            ),
        })

    return results


def main():
    print("=== Manuscript Audio Reviewer ===")

    if not WHISPERX_JSON.exists():
        raise FileNotFoundError(f"Missing WhisperX JSON: {WHISPERX_JSON}")

    if not AUDIO_WAV.exists():
        raise FileNotFoundError(f"Missing WAV: {AUDIO_WAV}")

    with WHISPERX_JSON.open("r", encoding="utf-8") as f:
        whisperx_data = json.load(f)

    audio, sr = sf.read(AUDIO_WAV)

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    evidence = {
        "media": {
            "sample_rate": sr,
            "duration_sec": round(len(audio) / sr, 3),
        },
        "whisperx_segments": build_whisperx_evidence(whisperx_data),
        "acoustic_segments": build_acoustic_evidence(
            whisperx_data,
            audio,
            sr,
        ),
        "emotion_segments": build_emotion_evidence(
            whisperx_data,
            audio,
            sr,
        ),
    }

    evidence["continuity_boundaries"] = build_continuity_evidence(
        evidence["whisperx_segments"],
        evidence["acoustic_segments"],
    )

    evidence["review_synthesis"] = build_review_synthesis(
        evidence["whisperx_segments"],
        evidence["acoustic_segments"],
        evidence["emotion_segments"],
        evidence["continuity_boundaries"],
    )

    EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)

    with EVIDENCE_JSON.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print("Evidence file:", EVIDENCE_JSON)
    print()

    for s in evidence["emotion_segments"]:
        print(f"SEGMENT {s['segment']}")
        print("Probabilities:", s["emotion_probabilities"])
        print(
            "Candidate:",
            s["top_emotion_candidate"],
            s["top_emotion_score"],
        )
        print("Margin:", s["candidate_margin"])
        print("Reliable:", s["emotion_measurement_reliable"])
        print("Final tone automatically chosen: NO")
        print()

    print("EMOTION EVIDENCE LAYER: PASS")


if __name__ == "__main__":
    main()
