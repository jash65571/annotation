from pathlib import Path
import json
import math

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent

AUDIO = ROOT / "analysis" / "audio.wav"

OUTPUT = (
    ROOT
    / "analysis"
    / "recording_defect_evidence.json"
)


def rms_dbfs(audio):
    if len(audio) == 0:
        return -120.0

    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )

    if rms <= 1e-12:
        return -120.0

    return 20.0 * math.log10(rms)


def low_frequency_ratio(
    audio,
    sample_rate,
    cutoff_hz=200.0,
):
    if len(audio) < 32:
        return 0.0

    centered = (
        audio
        - np.mean(audio)
    )

    windowed = (
        centered
        * np.hanning(
            len(centered)
        )
    )

    spectrum = np.fft.rfft(
        windowed
    )

    power = (
        np.abs(spectrum) ** 2
    )

    frequencies = np.fft.rfftfreq(
        len(windowed),
        1.0 / sample_rate,
    )

    useful = (
        (frequencies >= 20.0)
        & (frequencies <= 8000.0)
    )

    low = (
        (frequencies >= 20.0)
        & (frequencies <= cutoff_hz)
    )

    total_power = float(
        np.sum(
            power[useful]
        )
    )

    if total_power <= 0:
        return 0.0

    return float(
        np.sum(
            power[low]
        )
        / total_power
    )


def echo_correlation_score(
    audio,
    sample_rate,
):
    if len(audio) < int(
        0.5 * sample_rate
    ):
        return 0.0

    signal = (
        audio
        - np.mean(audio)
    )

    energy = float(
        np.dot(
            signal,
            signal,
        )
    )

    if energy <= 1e-12:
        return 0.0

    n = len(signal)

    fft_size = 1

    while fft_size < 2 * n:
        fft_size *= 2

    spectrum = np.fft.rfft(
        signal,
        n=fft_size,
    )

    autocorrelation = np.fft.irfft(
        spectrum
        * np.conjugate(spectrum),
        n=fft_size,
    )[:n]

    autocorrelation = (
        autocorrelation
        / max(
            autocorrelation[0],
            1e-12,
        )
    )

    minimum_lag = int(
        0.05 * sample_rate
    )

    maximum_lag = min(
        len(autocorrelation),
        int(
            0.25
            * sample_rate
        ),
    )

    if maximum_lag <= minimum_lag:
        return 0.0

    return float(
        np.max(
            autocorrelation[
                minimum_lag:
                maximum_lag
            ]
        )
    )


def analyze_recording_defects(
    audio_path,
    window_sec=1.0,
):
    audio, sample_rate = sf.read(
        audio_path
    )

    if getattr(
        audio,
        "ndim",
        1,
    ) > 1:
        audio = audio.mean(
            axis=1
        )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    samples_per_window = max(
        1,
        int(
            window_sec
            * sample_rate
        ),
    )

    windows = []
    review_windows = []

    for start_sample in range(
        0,
        len(audio),
        samples_per_window,
    ):
        end_sample = min(
            len(audio),
            start_sample
            + samples_per_window,
        )

        clip = audio[
            start_sample:
            end_sample
        ]

        if len(clip) < int(
            0.20 * sample_rate
        ):
            continue

        start = (
            start_sample
            / sample_rate
        )

        end = (
            end_sample
            / sample_rate
        )

        absolute = np.abs(
            clip
        )

        peak = float(
            np.max(absolute)
        )

        clipped_ratio = float(
            np.mean(
                absolute
                >= 0.999
            )
        )

        near_clip_ratio = float(
            np.mean(
                absolute
                >= 0.98
            )
        )

        level = rms_dbfs(
            clip
        )

        low_ratio = (
            low_frequency_ratio(
                clip,
                sample_rate,
            )
        )

        echo_score = (
            echo_correlation_score(
                clip,
                sample_rate,
            )
        )

        clipping_candidate = (
            clipped_ratio
            >= 0.0005
        )

        wind_candidate = (
            low_ratio >= 0.55
            and level > -45.0
        )

        echo_candidate = (
            echo_score >= 0.45
            and level > -45.0
        )

        item = {
            "start": round(
                start,
                3,
            ),
            "end": round(
                end,
                3,
            ),
            "peak_amplitude": round(
                peak,
                5,
            ),
            "rms_dbfs": round(
                level,
                2,
            ),
            "clipped_sample_ratio":
                round(
                    clipped_ratio,
                    6,
                ),
            "near_clip_sample_ratio":
                round(
                    near_clip_ratio,
                    6,
                ),
            "low_frequency_ratio":
                round(
                    low_ratio,
                    4,
                ),
            "echo_correlation_score":
                round(
                    echo_score,
                    4,
                ),
            "possible_clipping":
                clipping_candidate,
            "possible_wind_noise":
                wind_candidate,
            "possible_echo":
                echo_candidate,
        }

        windows.append(item)

        if clipping_candidate:
            review_windows.append({
                "priority": "high",
                "type":
                    "recording_defect_check",
                "defect":
                    "Obvious clipping",
                "start": round(
                    max(
                        0.0,
                        start - 0.25,
                    ),
                    3,
                ),
                "end": round(
                    end + 0.25,
                    3,
                ),
                "description":
                    "Possible clipped samples detected; listen for obvious clipping.",
            })

        if wind_candidate:
            review_windows.append({
                "priority": "medium",
                "type":
                    "recording_defect_check",
                "defect":
                    "Wind noise",
                "start": round(
                    max(
                        0.0,
                        start - 0.25,
                    ),
                    3,
                ),
                "end": round(
                    end + 0.25,
                    3,
                ),
                "description":
                    "Strong low-frequency energy detected; verify possible wind noise by listening.",
            })

        if echo_candidate:
            review_windows.append({
                "priority": "medium",
                "type":
                    "recording_defect_check",
                "defect":
                    "Excessive echo",
                "start": round(
                    max(
                        0.0,
                        start - 0.25,
                    ),
                    3,
                ),
                "end": round(
                    end + 0.25,
                    3,
                ),
                "description":
                    "Repeated delayed energy detected; verify whether excessive echo is audible.",
            })

    possible_clipping = any(
        item[
            "possible_clipping"
        ]
        for item in windows
    )

    possible_wind = any(
        item[
            "possible_wind_noise"
        ]
        for item in windows
    )

    possible_echo = any(
        item[
            "possible_echo"
        ]
        for item in windows
    )

    defects = {
        "Obvious clipping": {
            "evidence_candidate":
                "possible"
                if possible_clipping
                else "not_detected",
            "recommended_ui_answer":
                None,
            "human_listening_required":
                True,
        },

        "Distortion": {
            "evidence_candidate":
                "manual_only",
            "recommended_ui_answer":
                None,
            "human_listening_required":
                True,
        },

        "Excessive echo": {
            "evidence_candidate":
                "possible"
                if possible_echo
                else "not_detected",
            "recommended_ui_answer":
                None,
            "human_listening_required":
                True,
        },

        "Wind noise": {
            "evidence_candidate":
                "possible"
                if possible_wind
                else "not_detected",
            "recommended_ui_answer":
                None,
            "human_listening_required":
                True,
        },

        "Other recording defect": {
            "evidence_candidate":
                "manual_only",
            "recommended_ui_answer":
                None,
            "human_listening_required":
                True,
        },
    }

    return {
        "recording_defects":
            defects,
        "windows":
            windows,
        "review_windows":
            review_windows,
        "policy": {
            "automatic_yes_no_answers":
                False,
            "human_listening_required":
                True,
            "not_detected_does_not_mean_no":
                True,
        },
    }


def main():
    print(
        "=== RECORDING DEFECT EVIDENCE ==="
    )

    if not AUDIO.exists():
        raise FileNotFoundError(
            f"Missing audio: {AUDIO}"
        )

    result = (
        analyze_recording_defects(
            AUDIO
        )
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "RECORDING DEFECT EVIDENCE: PASS |",
        len(
            result["windows"]
        ),
        "windows |",
        len(
            result[
                "review_windows"
            ]
        ),
        "review cues",
    )

    print()

    for name, evidence in (
        result[
            "recording_defects"
        ].items()
    ):
        print(
            name,
            "|",
            evidence[
                "evidence_candidate"
            ],
            "| UI answer:",
            evidence[
                "recommended_ui_answer"
            ],
        )


if __name__ == "__main__":
    main()
