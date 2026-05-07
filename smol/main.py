import io, os
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

app = FastAPI()
model, processor = None, None

@app.on_event("startup")
async def load_model():
    global model, processor
    print("Starting up...")
    print("About to create model")
    # Smol
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        #quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    print("About to create processor")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    print("Startup has completed")

async def run_inference(object: str, prompt: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"In this image is a {object}. {prompt}"}
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
        return response
    except Exception as e:
        print(f"Error during inference of model: {str(e)}")
        return None

@app.post("/general")
async def run_smol_general(object: str, file: UploadFile = File(...)):
    GENERAL_CONTEXT_PROMPT =  f'''
    You are a visual assistant. You are given an image.
    This image contains a {object} in an environment.
    ''' + """
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
    try:
        try:
            response = await run_inference(object=object, prompt=GENERAL_CONTEXT_PROMPT, file=file)
            if response is None:
                print(f"run_smol_general: Response is None")
                return JSONResponse(content={"error": "Response is None"}, status_code=500)
            parsed = json.loads(response)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "Invalid JSON", "raw_output": response},
                status_code=500,
            )
    except Exception as e:
        print(f"run_smol_general: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/specific")
async def run_smol_specific(object: str, file: UploadFile = File(...)):
    CONTEXT_SPECIFIC_PROMPT =  f'''
    You are a visual assistant. You are given an image.
    This image contains a {object}.
    ''' + """
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
    try:
        try:
            response = await run_inference(object=object, prompt=CONTEXT_SPECIFIC_PROMPT, file=file)
            if response is None:
                print(f"run_smol_specific: Response is None")
                return JSONResponse(content={"error": "Response is None"}, status_code=500)
            parsed = json.loads(response)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "Invalid JSON", "raw_output": response},
                status_code=500,
            )
    except Exception as e:
        print(f"run_smol_specific: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

'''
    This endpoint is designed for cropped images of posters that could potentially contain locations.
'''
@app.post("/location")
async def run_smol_location_extraction(object: str, file: UploadFile = File(...)):
    LOCATION_EXTRACTION_PROMPT = """
    Extract any adresses that are in the image in written form. Furthermore, if the image is located at a particular well-known location,
    specify at which location that image has been taken (f.e. "London Bridge"). Do not include visual description of the area as location or adress.

    Return the result in valid JSON like:
    {
      "location": ""
      "adresses": []
    }
    """
    try:
        try:
            response = await run_inference(object=object, prompt=LOCATION_EXTRACTION_PROMPT, file=file)
            if response is None:
                print(f"run_smol_location_extraction: Response is None")
                return JSONResponse(content={"error": "Response is None"}, status_code=500)
            parsed = json.loads(response)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "Invalid JSON", "raw_output": response},
                status_code=500,
            )
    except Exception as e:
        print(f"run_smol_location_extraction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)