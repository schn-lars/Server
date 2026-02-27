import io
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
#from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import transformers
print("TRANSFORMERS VERSION:", transformers.__version__)
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig, AutoProcessor

#MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on {device}...")
GENERAL_CONTEXT_PROMPT = """
Analyze this object and return JSON with:
{
  "object_type": "",
  "visible_text": "",
  "summary": "",
  "contains_qr": boolean,
  "urls_detected": [],
  "other_relevant_information": ""
}
Return ONLY valid JSON. Do not explain anything.
"""

app = FastAPI()
model, processor = None, None

@app.on_event("startup")
async def load_model():
    global model, processor
    print("Starting up...")
    print("About to create model")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("About to create processor")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    print("Startup has completed")

@app.post("/smol")
async def run_smol(object: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"In this image is a {object}. {GENERAL_CONTEXT_PROMPT}"}
            ]
        },
        ]
        chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, images=[img], return_tensors="pt")
        inputs = inputs.to(device)

    # Generate outputs
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(response)
        try:
            parsed = json.loads(response)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "Invalid JSON", "raw_output": response},
                status_code=500,
            )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)