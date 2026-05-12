import io, os
import json
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
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


CONTEXT_SPECIFIC_PROMPT =  """
    Provide a description of this object using following attributes:
    - "summary" = What kind of object is it precisely? How does it look like?
    - "content" = In case the object has text on it, give a summary of that text here.
    - "urls": = In case the object contains either a website or QR-code, provide the websites in question.
    - "locations_detected" = List of all adresses or other location based information (i.e. ["Paris", "Bern 3011"]). Please make sure we have one location per item of the list.
    - "other_relevant_information" = Anything which is worth mentioning that has not been covered.

    The result must be captured in a single JSON like that:
    {
    "summary": "",
    "content": "",
    "urls": []
    "locations_detected": [],
    "other_relevant_information": ""
    }
    Return ONLY valid JSON.

    Example:
    {
    "summary": "A brown dog laying on green grass.",
    "content": "The collar reads 'Bello'".,
    "urls": [],
    "locations_detected": [],
    "other_relevant_information": "The dog's breed is a german shephard"
    }

    Try not to be very brief with your values. But dont overcomplicate your descriptions by large text that dont carry much meaning.
    The summary needs to be a text which can be used by a CLIP model to retrieve images of the same structure.
    """

GENERAL_CONTEXT_PROMPT =  """
    Provide a description of this environment using following attributes:
    - "summary" = Description of how the shown environment. Are there other similar objects? How do they relate to each other?
    - "locations_detected" = List of all adresses or other location based information (i.e. ["Paris", "Bern 3011"]). Please make sure we have one location per item of the list.
    - "other_relevant_information" = Anything which is worth mentioning that has not been covered.

    The result must be captured in a single JSON like that:
    {
    "summary": "",
    "locations_detected": [],
    "other_relevant_information": ""
    }
    Return ONLY valid JSON.

    Example:
    {
    "object_type": "dog",
    "visible_text": "",
    "summary": "A brown dog laying on green grass. Some other dogs are playing in the background.",
    "locations_detected": [],
    "other_relevant_information": "The dog's breed is a german shephard"
    }

    Try not to be very brief with your values. But dont overcomplicate your descriptions by large text that dont carry much meaning.
    The summary needs to be a text which can be used by a CLIP model to retrieve images of the same structure.
    """

def get_pretext(obj: str, general: bool) -> str:
    if general:
        return f"""
            You are a visual assistant. You are given an image.
            This image contains a {obj}.
            """
    else:
        return f"""
        You are a visual assistant. You are given an image.
        This image contains a {obj} in an environment.
        """


@app.post("/general")
async def run_minicpm_general(obj: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        msgs = [{'role': 'user', 'content': [img, get_pretext(obj=obj, general=True) + GENERAL_CONTEXT_PROMPT]}]
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer
        )
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/specific")
async def run_minicpm_specific(obj: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        msgs = [{'role': 'user', 'content': [img, get_pretext(obj=obj, general=False) + CONTEXT_SPECIFIC_PROMPT]}]
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer
        )
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/retrieve")
async def run_minicpm_retrieval(
    obj: str = Form(...),
    cropped: UploadFile = File(...),
    full: UploadFile = File(...)
):
    try:
        cropped_contents = await cropped.read()
        cropped_img = Image.open(io.BytesIO(cropped_contents)).convert("RGB")

        full_contents = await full.read()
        full_img = Image.open(io.BytesIO(full_contents)).convert("RGB")

        cropped_msgs = [{'role': 'user', 'content': [cropped_img, get_pretext(obj=obj, general=False) + CONTEXT_SPECIFIC_PROMPT]}]
        cropped_res = model.chat(
            msgs=cropped_msgs,
            tokenizer=tokenizer
        )

        full_msgs = [{'role': 'user', 'content': [full_img, get_pretext(obj=obj, general=True) + GENERAL_CONTEXT_PROMPT]}]
        full_res = model.chat(
            msgs=full_msgs,
            tokenizer=tokenizer
        )

        try:
            specific_generated_json = json.loads(cropped_res)
        except:
            specific_generated_json = {"raw_output": cropped_res}
        
        try:
            general_generated_json = json.loads(full_res)
        except:
            general_generated_json = {"raw_output": full_res}
        
        final_content = {"general" : general_generated_json, "specific" : specific_generated_json}
        return JSONResponse(content=final_content)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)