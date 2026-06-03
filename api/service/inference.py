from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO
import time
from PIL import Image
import io
import uuid
from service.utils import logging
from fastapi import UploadFile
import torch

YOLO_WORLD_CUSTOM = 'yolov8s-world-custom.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# https://docs.ultralytics.com/models/sam-3/#segment-with-text-prompts
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True, # Use FP16 for faster inference
    save=False
)

SAM3_DEFAULT_PROMPT= '''
    You are an agent which is used by a VR headset. Your goal is to identify as many objects as you possibly can.
    Make sure that you return only the objects you are more than 60 percent sure about.
    Try to categorize the returned objects in somewhat logical labels.
    An example for labels we are NOT interested in are: ['blue book', 'green book', 'book with colorful cover', 'poster'],
    instead we want to have the label-set ['book', 'poster'].
'''

print(f"Setup inference on {DEVICE}")

class InferenceSession:
    def __init__(self):
        self.model_type = None
        self.prompt = [SAM3_DEFAULT_PROMPT]
        self.sam_predictor = None
        self.yolo_model = None
        self.task = None
    
    # might be useful
    def has_default_prompt(self) -> bool:
        return self.prompt == [SAM3_DEFAULT_PROMPT]

    '''
        @param model_type [YOLOv26, YOLOv11, SAM3]
        @param task:      [Segmentation, Detection] 
    '''
    def load_model(self, model_type: str, task: str):
        if model_type == "YOLOv26":
            self.yolo_model = YOLO("yolo26s-seg.pt" if task == 'Segmentation' else 'yolo26s.pt')
            self.yolo_model.to(DEVICE)
            self.sam_predictor = None
            self.model_type = model_type
            self.task = task

        elif model_type == "YOLOv11":
            self.yolo_model = YOLO("yolo11s-seg.pt" if task == 'Segmentation' else 'yolo11s.pt')
            self.yolo_model.to(DEVICE)
            self.sam_predictor = None
            self.model_type = model_type
            self.task = task

        elif model_type == "SAM3":
            self.sam_predictor = SAM3SemanticPredictor(overrides={**overrides, "device": DEVICE})
            self.yolo_model = None
            self.model_type = model_type
            self.task = task
        
        elif model_type == "WORLD":
            self.yolo_model = YOLO(YOLO_WORLD_CUSTOM)
            self.yolo_model.to(DEVICE)
            self.sam_predictor = None
            self.model_type = model_type
            self.task = task

        else:
            print("load_model() ERROR - Unknown model", model_type)
            raise ValueError("Unknown model")

    def predict(self, frame: bytes):
        img = Image.open(io.BytesIO(frame)).convert("RGB")
        width, height = img.size

        if self.model_type == "SAM3":
            self.sam_predictor.set_image(img)
            # TODO: check return value here
            return self.sam_predictor(text=self.prompt)

        elif self.task == "Segmentation" and self.model_type.startswith('YOLO'):
            results = self.yolo_model.predict(img)
            r = results[0]

            # TODO: we also need to return the actual masks here. This is not yet happening.
            print(r)
            return {
                "type": self.task,
                "observations": [
                    {
                        "id": str(uuid.uuid4()),
                        "label": str(int(c)),
                        "confidence": float(s),
                        "bbox": {
                            "x": float(x1) / width,
                            "y": float(y1) / height,
                            "width": float(x2 - x1) / width,
                            "height": float(y2 - y1) / height
                        },
                        "worldPosition": None
                    }
                    for (x1, y1, x2, y2), s, c in zip(
                        r.boxes.xyxy.tolist(),
                        r.boxes.conf.tolist(),
                        r.boxes.cls.tolist()
                    )
                ]
            }


        elif self.task == "Detection" and self.model_type.startswith('YOLO'):
            result = self.yolo_model.predict(img)
            r = result[0]
            '''
                Ultralytics: boxes.xyxy = top-left-x, top-left-y, bottom-right-x, bottom-right-y
                This is however not the case. An example output is: 
                    [297.5180358886719, 389.44091796875, 425.7095642089844, 479.23895263671875]
                We can clearly see that arr[0] < arr[2] and arr[1] < arr[3]
            '''
            return {
                "type": self.task,
                "observations": [
                    {
                        "id": str(uuid.uuid4()),
                        "label": str(c),
                        "confidence": float(s),
                        "bbox": {
                            "x": float(x1) / width,
                            "y": float(y1) / height,
                            "width": float(x2 - x1) / width,
                            "height": float(y2 - y1) / height
                        },
                        "worldPosition": None
                    }
                    for (x1, y1, x2, y2), s, c in zip(
                        r.boxes.xyxy.tolist(), 
                        r.boxes.conf.tolist(),
                        [r.names[idx] for idx in r.boxes.cls.tolist()]
                    )
                ]
            }
        elif self.model_type == 'WORLD':
            result = self.yolo_model.predict(img)
            r = result[0]
            return {
                "type": self.task,
                "observations": [
                    {
                        "id": str(uuid.uuid4()),
                        "label": str(c),
                        "confidence": float(s),
                        "bbox": {
                            "x": float(x1) / width,
                            "y": float(y1) / height,
                            "width": float(x2 - x1) / width,
                            "height": float(y2 - y1) / height
                        },
                        "worldPosition": None
                    }
                    for (x1, y1, x2, y2), s, c in zip(
                        r.boxes.xyxy.tolist(), 
                        r.boxes.conf.tolist(),
                        [r.names[idx] for idx in r.boxes.cls.tolist()]
                    )
                ]
            }
        else:
            print(f"predict() ERROR - Unknown model:", self.model_type)
            raise RuntimeError("Model not loaded")


def save_result_image_to_disk(model: str, results):
    annotated = results[0].plot()
    Image.fromarray(annotated).save(f"/outputs/{model}-{time.time()}.jpg")