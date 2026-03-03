import io, os
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
#from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import transformers
print("TRANSFORMERS VERSION:", transformers.__version__)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on {device}...")

app = FastAPI()
model, processor = None, None

@app.on_event("startup")
async def load_model():
    global model, processor, tokenizer
    print("Starting up...")
    print("About to create model")
    # MiniCPM
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # Use FP16 for RTX 4090
        device_map="auto"             # Splits layers across available GPU automatically
    )

    # Load processor (handles image preprocessing + tokenizer)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model.eval()

@app.post("/general")
async def run_qwen_general(object: str, file: UploadFile = File(...)):
    GENERAL_CONTEXT_PROMPT = """
    You are a visual assistant. You are given an image.
    Provide a description of this object using following attributes:
    - "object_type" = Specific object type
    - "visible_text" = Text, if any, which is placed on the object itself
    - "summary" = Description of how the object is embedded in the environment. Are there other similar objects? How do they relate to each other?
    - "locations_detected" = List of all adresses or other location based information (i.e. ["Paris", "Bern 3011"]). Please make sure we have one location per item of the list.
    - "other_relevant_information" = Anything which is worth mentioning that has not been covered.

    The result must be captured in a single JSON like that:
    {
    "object_type": "",
    "visible_text": "",
    "summary": "",
    "locations_detected": [],
    "other_relevant_information": ""
    }
    Return ONLY valid JSON.

    Example:
    {
    "object_type": "dog",
    "visible_text": "",
    "summary": "A brown dog laying on green grass.",
    "locations_detected": [],
    "other_relevant_information": "The dog's breed is a german shephard"
    }
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": GENERAL_CONTEXT_PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = processor(
            text=text,
            images=img,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        try:
            generated_json = json.loads(generated_text)
        except:
            generated_json = {"raw_output": generated_text}
        return JSONResponse(content=generated_json)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)