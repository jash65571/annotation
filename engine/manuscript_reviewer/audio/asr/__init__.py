"""Local ASR adapters: isolated uv worker environments, evidence-only output.

The core engine never imports torch/ctranslate2/whisper — workers run in their
own uv-managed environments and communicate via JSON files. A worker failure
degrades to ASR_UNAVAILABLE/FAILED evidence; it never aborts the audit and
never authorizes any cloud fallback.
"""
