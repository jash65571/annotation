"""Audio Truth Engine (Phase 3): deterministic audio evidence + local ASR leads.

Actual source audio is factual truth; ASR is evidence only. Everything runs
locally — media is never uploaded anywhere, and a local ASR failure never
authorizes any cloud or Descript fallback (rules v1.2, audio_tooling).
"""
