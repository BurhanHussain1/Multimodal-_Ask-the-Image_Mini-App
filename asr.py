import whisper
asr_model = whisper.load_model("small")

def transcribe_audio(audio_path: str) -> str:
    result = asr_model.transcribe(audio_path, language='en')
    question = result["text"]
    print("Transcribed Question:", question)
    return question
