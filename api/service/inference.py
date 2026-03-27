from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO
import base64
from PIL import Image
import io
from fastapi import UploadFile

# https://docs.ultralytics.com/models/sam-3/#segment-with-text-prompts
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True, # Use FP16 for faster inference
    save=False
)

YOLOv26_SEG = YOLO("yolo26m-seg.pt")
YOLOv26_DET = YOLO("yolo26m.pt")
SAMv3_SEG = SAM3SemanticPredictor(overrides=overrides)

async def yolo_26_segmentation_prediction(file: UploadFile):
    print(f"Starting inference for YOLOv26 - SEGMENTATION")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = YOLOv26_SEG.predict(img)
    print(results)
    return results

async def yolo_26_detection_prediction(file: UploadFile):
    print(f"Starting inference for YOLOv26 - DETECTION")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = YOLOv26_SEG.predict(img)
    print(results)
    return results

async def sam3_segment_with_text_prompts(file: UploadFile, text: list[str]):
    print(f"Starting inference for SAM3 - TEXTUAL PROMPTS")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    SAMv3_SEG.set_image(img)
    results = SAMv3_SEG(text=text)
    print(results)
    return results

async def sam3_segment_with_bounding_boxes(file: UploadFile, boxes: list[int]):
    print(f"Starting inference for SAM3 - BOUNDING BOXES")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    SAMv3_SEG.set_image(img)
    results = SAMv3_SEG(boxes=boxes)
    print(results)
    return results


def save_result_image_to_disk():
    pass