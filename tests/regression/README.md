# Real-world regression fixtures

Architecture for testing the Shot Truth Engine against completed Manuscript II
tasks WITHOUT committing private task videos.

## Rules

- **Private source videos are never committed.** Everything matching
  `tests/regression/media/` is gitignored; only manifests are versioned.
- A regression manifest binds expectations to a video via its SHA-256, so a run
  against the wrong file fails immediately.
- Expected boundaries are exact frame pairs and exact PTS, in the Phase 1
  ledger's identity space — never rounded seconds.

## Layout

```
tests/regression/
├── README.md            # this file (committed)
├── media/               # local private clips (gitignored)
│   └── <clip>.mp4
└── manifests/           # expected results (committed)
    └── <clip>.expected.json
```

## Manifest format (`<clip>.expected.json`)

```json
{
  "media_sha256": "…64 hex chars…",
  "frame_count": 960,
  "expected_boundaries": [
    {
      "left_frame_index": 705,
      "right_frame_index": 706,
      "right_pts": 361472,
      "transition_type": "Hard cut",
      "notes": "verified in the live tool on 2026-08-12"
    }
  ],
  "allowed_review_pairs": [[501, 502]],
  "review_notes": "single white flash at F501 is an in-shot muzzle flash"
}
```

Semantics for a future `test_regression.py` runner (Phase 3+):

- skip the manifest when its media file is absent (developer machines without
  the private clip);
- FAIL if the present media's SHA-256 or frame count differs;
- FAIL if any `expected_boundaries` entry is missing from SUPPORTED (exact
  left/right/pts match) — false negatives are high risk;
- FAIL if a SUPPORTED boundary exists that is neither expected nor listed in
  `allowed_review_pairs`;
- REVIEW_REQUIRED candidates outside both lists are reported but do not fail.
