import gradio as gr
from PIL import Image

from asr import transcribe_audio
from qa import get_image_answer
from tts import text_to_speech

def multimodal_qa_app(image: Image.Image, audio_path: str):
    question_text = transcribe_audio(audio_path)
    answer = get_image_answer(image, question_text)
    audio_response = text_to_speech(answer)
    return question_text, answer, audio_response

interface = gr.Interface(
    fn=multimodal_qa_app,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Audio(type="filepath", label="Ask a Question via Mic (10s max)")
    ],
    outputs=[
        gr.Textbox(label="Transcribed Question"),
        gr.Textbox(label="Answer"),
        gr.Audio(label="Spoken Answer")
    ],
    title="Ask-the-Image: Multimodal QA",
    description="Upload an image and ask a question using your voice. The app answers and reads it out loud."
)

if __name__ == "__main__":
    interface.launch(debug=True)
