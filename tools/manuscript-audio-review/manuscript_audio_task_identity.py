from pathlib import Path
import hashlib
import json


def video_fingerprint(video_path):
    video_path = Path(video_path)

    digest = hashlib.sha256()

    with video_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_bound_json(
    path,
    current_video_sha256,
    label,
):
    path = Path(path)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    bound_hash = data.get(
        "video_sha256"
    )

    if not bound_hash:
        print(
            f"{label}: IGNORED | "
            "file is not bound to the current video"
        )
        return None

    if (
        bound_hash
        != current_video_sha256
    ):
        print(
            f"{label}: IGNORED | "
            "belongs to a different video"
        )
        return None

    return data


def write_video_identity(
    video_path,
    output_path,
):
    video_path = Path(video_path)

    result = {
        "video_name":
            video_path.name,

        "video_sha256":
            video_fingerprint(
                video_path
            ),
    }

    output_path = Path(
        output_path
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return result
