"""
RunPod Serverless worker for IELTS speaking pronunciation scoring.

Accepts audio (base64 or URL), transcribes it with Whisper, forced-aligns
against a wav2vec2 phoneme model, and returns per-word pronunciation
confidence scores. See ielts_pronunciation.py for the full pipeline.
"""
import base64
import tempfile

import requests
import runpod

from ielts_pronunciation import score_ielts_response


def handler(event):
    job_input = event.get("input", {}) or {}

    if "audio_base64" in job_input:
        raw = base64.b64decode(job_input["audio_base64"])
        suffix = job_input.get("audio_format", ".webm")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            audio_path = tmp.name
    elif "audio_url" in job_input:
        resp = requests.get(job_input["audio_url"], timeout=30)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(resp.content)
            audio_path = tmp.name
    else:
        return {"error": "job input must include 'audio_base64' or 'audio_url'"}

    try:
        result = score_ielts_response(audio_path)
    except Exception as e:
        return {"error": f"pronunciation scoring failed: {e}"}

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
