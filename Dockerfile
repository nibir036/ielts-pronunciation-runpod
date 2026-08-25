FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# System deps: espeak-ng (phonemizer backend), ffmpeg (pydub audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image at build time so cold starts don't
# re-download weights on every new worker.
RUN python3 -c "\
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-lv-60-espeak-cv-ft'); \
Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-lv-60-espeak-cv-ft')"
RUN python3 -c "\
from transformers import pipeline; \
pipeline('automatic-speech-recognition', model='openai/whisper-small')"

COPY ielts_pronunciation.py .
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
