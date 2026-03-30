from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi import WebSocket
import asyncio
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


@inference_api_router.websocket("/ws/yolov26-segmentation")
async def yolov26_websocket_inference(websocket: WebSocket):
    await websocket.accept()

    latest_frame = None

    async def receiver():
        nonlocal latest_frame
        while True: # overwriting new frames
            data = await websocket.receive_bytes()
            latest_frame = data

    async def processor():
        nonlocal latest_frame
        while True:
            if latest_frame is None:
                await asyncio.sleep(0.01)
                continue

            frame = latest_frame
            latest_frame = None

            results = inference.yolo_26_segmentation_prediction(frame=frame)

            await websocket.send_json({
                "boxes": results[0].boxes.xyxy.tolist(),
                "scores": results[0].boxes.conf.tolist(),
                "classes": results[0].boxes.cls.tolist()
            })
    await asyncio.gather(receiver(), processor())

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