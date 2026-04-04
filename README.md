## Ask-the-Image — Multimodal Mini App

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HF_Transformers-🤗-FFCC4D)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-20B2AA?logo=python&logoColor=white)](https://gradio.app/)

A lightweight demo that lets users upload an image and ask a spoken question. The app:
- Transcribes audio to text (Whisper)
- Answers about the image with a vision-language model (BLIP‑2 FLAN‑T5)
- Speaks the answer back using TTS


## Features
- Image + speech input, text + audio output
- GPU acceleration where available (CUDA), falls back to CPU
- Simple Gradio UI for local testing


## Project Structure
- `app.py` — Gradio interface wiring image, ASR, VLM, and TTS
- `asr.py` — Whisper speech-to-text (`small` model)
- `qa.py` — BLIP‑2 FLAN‑T5 vision‑language answering
- `tts.py` — gTTS text‑to‑speech to MP3
- `requirements.txt` — dependencies
- `README.md` — this document


## Setup

### Prerequisites
- Python 3.10+
- Optional CUDA GPU for faster inference (BLIP‑2 is large)

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python app.py
```
Open the local Gradio URL, upload an image, ask your question by voice, and receive text + spoken answers.


## Notes and Configuration
- Whisper ASR: `asr.py` uses the `small` model. Change to `base`/`tiny` for lower resource usage.
- BLIP‑2: `qa.py` loads `Salesforce/blip2-flan-t5-xl`. This is compute-heavy; consider smaller variants if needed.
- Mixed precision: Float tensors are cast to float16 on CUDA only; token tensors remain integer types.
- TTS: `gtts` calls a public TTS service; requires internet connectivity.
- Gradio: `interface.launch(debug=True)` shows logs and errors in the console. Switch to `debug=False` for cleaner output.


## Security and Secrets
- No API keys or tokens are hardcoded in this repository.
- gTTS uses a public endpoint; no credentials needed.
- Large model weights are pulled from public hubs on first run. Excluding caches/artifacts from Git is recommended.


## Suggested .gitignore
```
__pycache__/
*.pyc
.ipynb_checkpoints/
.cache/
.gradio/
env/
.venv/
venv/
*.pt
*.pth
*.bin
```


## License
Add an open‑source license (e.g., MIT or Apache‑2.0) appropriate for your use case.