import io, os
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
#from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import transformers
print("TRANSFORMERS VERSION:", transformers.__version__)
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv("/app/.env")

HUGGING_TOKEN = os.getenv('HUGGING_TOKEN')
MODEL_ID = 'openbmb/MiniCPM-o-4_5' # MiniCPM-V-2_6 is Vision only apparently

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model on {device}...")

app = FastAPI()
model, processor, tokenizer = None, None, None

@app.on_event("startup")
async def load_model():
    global model, processor, tokenizer
    print("Starting up...")
    print("About to create model")
    # MiniCPM
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=False,
        token=HUGGING_TOKEN
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HUGGING_TOKEN)

@app.post("/general")
async def run_minicpm_general(object: str, file: UploadFile = File(...)):
    GENERAL_CONTEXT_PROMPT = """
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
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        msgs = [{'role': 'user', 'content': [img, GENERAL_CONTEXT_PROMPT]}]
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer
        )
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)