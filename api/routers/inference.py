from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from service import inference
import json

inference_api_router = APIRouter(
    prefix="/api/inference"
)

#
#   This is supposed to make segmentation returning masks seen in an image.
#
@inference_api_router.post("/yolov26-segmentation")
async def yolo_26_segmentation_prediction(file: UploadFile = File(...)):
    try:
        results = await inference.yolo_26_segmentation_prediction(file=file)
        return JSONResponse(content={"results": results}, status_code=200)
    except Exception as e:
        print(f"Error in yolo_26_segmentation_prediction(): {str(e)}")
        raise HTTPException(status_code=500, detail="YOLOv26 Segmentation Prediction has failed!")

#
#   This is supposed to make predictions returning bounding boxes.
#
@inference_api_router.post("/yolov26-detection")
async def yolo_26_detection_prediction(file: UploadFile = File(...)):
    try:
        results = await inference.yolo_26_detection_prediction(file=file)
        return JSONResponse(content={"results": results}, status_code=200)
    except Exception as e:
        print(f"Error in yolo_26_detection_prediction(): {str(e)}")
        raise HTTPException(status_code=500, detail="YOLOv26 Detection Prediction has failed!")

@inference_api_router.post("/sam3-prompted")
async def sam3_segment_with_text_prompts(file: UploadFile = File(...), text: list[str] = Form(...)):
    try:
        text = await json.loads(text)
        results = await inference.sam3_segment_with_text_prompts(file=file, text=text)
        return JSONResponse(content={"results": results}, status_code=200)
    except Exception as e:
        print(f"Error in sam3_segment_with_text_prompts(): {str(e)}")
        raise HTTPException(status_code=500, detail="SAM3-TXT Prediction has failed!")

@inference_api_router.post("/sam3-boxed")
async def sam3_segment_with_bounding_boxes(file: UploadFile = File(...), boxes: list[int] = Form(...)):
    try:
        text = await json.loads(text)
        results = await inference.sam3_segment_with_bounding_boxes(file=file, boxes=boxes)
        return JSONResponse(content={"results": results}, status_code=200)
    except Exception as e:
        print(f"Error in sam3_segment_with_bounding_boxes(): {str(e)}")
        raise HTTPException(status_code=500, detail="SAM3-BB Prediction has failed!")