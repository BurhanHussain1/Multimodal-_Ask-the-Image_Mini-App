import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map="auto"
)

def get_image_answer(image: Image.Image, question: str) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, text=question, return_tensors="pt")

    for key in inputs:
        if inputs[key].dtype in [torch.float32, torch.float64]:
            # Cast only float tensors (like pixel values) to float16 if on CUDA
            inputs[key] = inputs[key].to(device, torch.float16 if device == "cuda" else torch.float32)
        else:
            # Leave token inputs (e.g., input_ids) as integers
            inputs[key] = inputs[key].to(device)

    print("Prompt Passed to VLM:", f"Question: {question} Answer:")
    output_ids = model.generate(**inputs)
    answer = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    print("Model Response:", answer)
    return answer
