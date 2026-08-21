"""Face tracking + mouth-activity worker (Phase 3B).

Runs under the isolated `.venv-vision` environment (mediapipe + opencv) so
the base `.venv-review` interpreter never needs a vision stack (design
rule 5). JSON on stdout only; every failure path degrades to a well-formed
`status: "unavailable"/"failed"` result instead of crashing (design rule 4)
-- a video with no visible faces, or a missing/broken vision environment,
must never break the rest of the packet.

Scope decision (recorded here since it shapes every consumer of this
evidence): LR-ASD/TalkNet-style active-speaker networks were evaluated and
are not practical in this CPU-only, no-CUDA environment -- both hardcode
`.cuda()` calls throughout, are not pip-installable, and getting either
running would mean patching vendored research code with no guarantee of a
correct CPU result. This worker instead produces a *mouth-motion* signal
(mediapipe Face Mesh landmarks -> mouth-aspect-ratio time series per
tracked face) rather than genuine audiovisual sync. That is intentionally a
weaker signal, and downstream fusion logic (manuscript_audio_speaker_mapping.py)
must never let a mouth-motion-only candidate reach STRONG -- STRONG is
reserved for real audiovisual sync evidence this worker cannot produce.

What this worker does NOT do:
- It never assigns a character (C#) identity. It only produces anonymous
  face-track ids (F1, F2, ...).
- It never claims a face is "the" active speaker. It only reports a
  mouth-motion time series per track; fusion/scoring happens downstream.

Usage:
    python manuscript_audio_face_worker.py VIDEO.mp4 OUTPUT.json [--fps 5]
"""

import argparse
import json
import sys
import time
from pathlib import Path


# Mediapipe Face Mesh landmark indices for a simple mouth-aspect-ratio.
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291

_IOU_MATCH_THRESHOLD = 0.25
_MAX_TRACK_GAP_SEC = 0.6  # a gap longer than this ends a track (likely a cut)


def diagnostic_code_for_exception(exc):
    """Map worker failures to an actionable setup/operational code."""
    module = getattr(exc, "name", "") or ""
    message = str(exc).lower()
    if module == "mediapipe" or "mediapipe" in message:
        return "mediapipe_import_failed"
    if module in ("cv2", "opencv", "opencv_python") or "cv2" in message:
        return "opencv_import_failed"
    return "worker_failed"


def _bbox_from_landmarks(landmarks, width, height):
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    x0, x1 = min(xs) * width, max(xs) * width
    y0, y1 = min(ys) * height, max(ys) * height
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _mouth_aspect_ratio(landmarks, width, height):
    top = landmarks[_MOUTH_TOP]
    bottom = landmarks[_MOUTH_BOTTOM]
    left = landmarks[_MOUTH_LEFT]
    right = landmarks[_MOUTH_RIGHT]

    vertical = ((top.x - bottom.x) * width) ** 2 + ((top.y - bottom.y) * height) ** 2
    horizontal = ((left.x - right.x) * width) ** 2 + ((left.y - right.y) * height) ** 2

    vertical = vertical ** 0.5
    horizontal = horizontal ** 0.5

    return round(vertical / horizontal, 4) if horizontal > 0 else 0.0


def _extract_frame_faces(video_path, fps, max_num_faces=4, min_detection_confidence=0.5):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / native_fps if native_fps else None

    step = max(1, round(native_fps / fps))

    frames = []
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    ) as face_mesh:
        frame_index = 0

        while True:
            ok = cap.grab()
            if not ok:
                break

            if frame_index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    frame_index += 1
                    continue

                t = round(frame_index / native_fps, 3)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                faces = []
                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        faces.append({
                            "bbox": _bbox_from_landmarks(landmarks, width, height),
                            "mar": _mouth_aspect_ratio(landmarks, width, height),
                        })

                frames.append({"time": t, "faces": faces})

            frame_index += 1

    cap.release()

    return frames, {
        "native_fps": round(native_fps, 3),
        "sampled_fps": fps,
        "duration_sec": round(duration, 3) if duration else None,
        "width": width,
        "height": height,
        "frames_sampled": len(frames),
    }


def _track_faces(frames):
    """Simple greedy IoU tracker across sampled frames.

    A track ends (and a new one begins on the next match) whenever no
    detection overlaps it above the IoU threshold, or the time gap since it
    was last seen exceeds `_MAX_TRACK_GAP_SEC` -- e.g. across a shot cut, a
    face reappearing gets a new id rather than being silently assumed to be
    the same person. Track identity is purely visual/positional; it is
    never a character (C#) claim.
    """
    active = []  # list of {"id","last_time","last_bbox","points":[...]}
    next_id = 1

    for frame in frames:
        t = frame["time"]
        unmatched = list(range(len(frame["faces"])))
        used_tracks = set()

        for det_index in list(unmatched):
            det = frame["faces"][det_index]
            best_track = None
            best_iou = 0.0

            for track in active:
                if track["id"] in used_tracks:
                    continue
                if t - track["last_time"] > _MAX_TRACK_GAP_SEC:
                    continue

                iou = _iou(det["bbox"], track["last_bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= _IOU_MATCH_THRESHOLD:
                best_track["last_time"] = t
                best_track["last_bbox"] = det["bbox"]
                best_track["points"].append({
                    "time": t, "bbox": det["bbox"], "mar": det["mar"],
                })
                used_tracks.add(best_track["id"])
                unmatched.remove(det_index)

        for det_index in unmatched:
            det = frame["faces"][det_index]
            track_id = f"F{next_id}"
            next_id += 1
            active.append({
                "id": track_id,
                "last_time": t,
                "last_bbox": det["bbox"],
                "points": [{"time": t, "bbox": det["bbox"], "mar": det["mar"]}],
            })

    # Drop tracks that only ever had one detection -- almost always a false
    # positive (a single stray frame), not a real visible face worth
    # reporting as evidence.
    tracks = [t for t in active if len(t["points"]) >= 2]

    results = []
    for track in tracks:
        points = track["points"]
        mars = [p["mar"] for p in points]

        results.append({
            "face_id": track["id"],
            "first_seen": points[0]["time"],
            "last_seen": points[-1]["time"],
            "frame_count": len(points),
            "mean_mar": round(sum(mars) / len(mars), 4),
            "mar_range": round(max(mars) - min(mars), 4),
            "points": points,
        })

    results.sort(key=lambda r: r["first_seen"])
    return results


def write_face_track_evidence(video_path, output_path, fps=5):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()

    try:
        frames, media_info = _extract_frame_faces(video_path, fps=fps)
        face_tracks = _track_faces(frames)

        result = {
            "status": "complete",
            "runtime_sec": round(time.time() - started, 2),
            "media": media_info,
            "face_tracks": face_tracks,
            "policy": (
                "Face tracks are anonymous visual identities (F1, F2, ...), "
                "never character (C#) claims. mar (mouth-aspect-ratio) is a "
                "mouth-motion proxy, not a verified audiovisual sync score."
            ),
        }

        if not face_tracks:
            result["note"] = (
                "No face detected across sampled frames (or every "
                "detection was a single-frame flicker). Downstream mapping "
                "must treat all speech as possibly off-screen."
            )

    except Exception as exc:  # noqa: BLE001 -- fail soft (design rule 4)
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "error_code": diagnostic_code_for_exception(exc),
            "runtime_sec": round(time.time() - started, 2),
            "face_tracks": [],
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if result["status"] == "complete":
        print(
            "FACE TRACKING: PASS |",
            f"{media_info['frames_sampled']} frames sampled |",
            f"{len(face_tracks)} face track(s) |",
            f"{result['runtime_sec']}s",
        )
    else:
        print("FACE TRACKING: FAILED |", result.get("error"))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("output_path")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    result = write_face_track_evidence(args.video_path, args.output_path, fps=args.fps)
    sys.exit(0 if result["status"] == "complete" else 0)  # never fail the pipeline


if __name__ == "__main__":
    main()
