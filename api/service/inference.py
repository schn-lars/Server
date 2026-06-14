from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO
import time
from PIL import Image
import io
import uuid
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import base64

YOLO_WORLD_CUSTOM = 'yolov8s-world-custom.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# https://docs.ultralytics.com/models/sam-3/#segment-with-text-prompts
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3_b.pt",
    half=True, # Use FP16 for faster inference
    save=False
)

SAM3_DEFAULT_PROMPT= [
    "laptop",
    "person",
    "cup",
    "plate",
    "poster"
]

print(f"Setup inference on {DEVICE}")

class InferenceSession:
    def __init__(self):
        self.model_type = None
        self.prompt = SAM3_DEFAULT_PROMPT
        self.sam_predictor = None
        self.yolo_model = None
        self.task = None
        self.streaming = True
    
    # might be useful
    def has_default_prompt(self) -> bool:
        return self.prompt == SAM3_DEFAULT_PROMPT

    '''
        @param model_type [YOLOv26, YOLOv11, SAM3]
        @param task:      [Segmentation, Detection] 
    '''
    def load_model(self, model_type: str, task: str):
        print("load_model:", model_type, task)
        task = task if task else "Detection"
        if self.task == task and self.model_type == model_type:
            print("load_model: No changes needed. Already running this mode.")
            return

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
            self.model_type = model_type
            self.task = task

            if self.task == "Segmentation":
                sam = sam_model_registry["vit_t"](
                    checkpoint="/opt/MobileSAM/weights/mobile_sam.pt"
                )
                sam.to(DEVICE)
                self.sam_predictor = SamPredictor(sam)
            else:
                self.sam_predictor = None
        else:
            print("load_model() ERROR - Unknown model", model_type)
            raise ValueError("Unknown model")

    def predict(self, frame: bytes):
        img = Image.open(io.BytesIO(frame)).convert("RGB")
        width, height = img.size

        if self.model_type == "SAM3":
            start = time.time()
            self.sam_predictor.set_image(img)
            # TODO: check return value here
            results = self.sam_predictor.predict(
                source=np.array(img),
                texts=self.prompt
            )
            r = results[0]

            obs = []
            if r.masks is not None and r.boxes is not None:
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.tolist()
                scores = r.boxes.conf.tolist()
                classes = [r.names[idx] for idx in r.boxes.cls.tolist()]

                for mask, box, score, cls in zip(masks, boxes, scores, classes):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_img.resize((256, 256), Image.NEAREST)
                    mask_np = np.array(mask_resized)
                    mask_bytes = (mask_np > 0).astype(np.uint8).tobytes()
                    mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

                    x1, y1, x2, y2 = box
                    obs.append({
                        "label": str(cls),
                        "confidence": float(score),
                        "bbox": {
                            "x": float(x1) / width,
                            "y": float(y1) / height,
                            "width": float(x2 - x1) / width,
                            "height": float(y2 - y1) / height
                        },
                        "mask": mask_b64,
                        "mask_width": 256,
                        "mask_height": 256
                    })
            return {
                "type": self.task,
                "observations": obs,
                "time": time.time() - start
            }

        elif self.task == "Segmentation" and self.model_type.startswith('YOLO'):
            start = time.time()
            results = self.yolo_model.predict(img)
            r = results[0]

            # TODO: we also need to return the actual masks here. This is not yet happening.
            #print(r)
            obs = []
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.tolist()
                scores = r.boxes.conf.tolist()
                classes = [r.names[idx] for idx in r.boxes.cls.tolist()]

                for mask, box, score, cls in zip(masks, boxes, scores, classes):
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_img.resize((256, 256), Image.NEAREST)
                    mask_np = np.array(mask_resized)
                    mask_bytes = (mask_np > 0).astype(np.uint8).tobytes()
                    mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

                    x1, y1, x2, y2 = box
                    obs.append({
                        "label": str(cls),
                        "confidence": float(score),
                        "bbox": {
                            "x": float(x1) / width,
                            "y": float(y1) / height,
                            "width": float(x2 - x1) / width,
                            "height": float(y2 - y1) / height
                        },
                        "mask": mask_b64,
                        "mask_width": 256,
                        "mask_height": 256
                    })
            return {
                "type": self.task,
                "observations": obs,
                "time": time.time() - start
            }


        elif self.task == "Detection" and self.model_type.startswith('YOLO'):
            start = time.time()
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
                ],
                "time": time.time() - start
            }
        elif self.model_type == 'WORLD':
            start = time.time()
            if self.task == 'Segmentation':
                result = self.yolo_model.predict(img)
                r = result[0]
                boxes = r.boxes.xyxy.tolist()
                scores = r.boxes.conf.tolist()
                classes = [r.names[idx] for idx in r.boxes.cls.tolist()]
                self.sam_predictor.set_image(np.array(img))
                obs = []
                for box, score, cls in zip(boxes, scores, classes):
                    masks, _, _ = self.sam_predictor.predict(
                        box=np.array(box),
                        multimask_output=False
                    )
                    mask = masks[0]
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_resized = mask_img.resize((256, 256), Image.NEAREST)
                    mask_np = np.array(mask_resized)
                    mask_bytes = (mask_np > 0).astype(np.uint8).tobytes()
                    mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')
                    obs.append({
                        "label": str(cls),
                        "confidence": float(score),
                        "bbox": {
                                "x": float(box[0]) / width,
                                "y": float(box[1]) / height,
                                "width": float(box[2] - box[0]) / width,
                                "height": float(box[3] - box[1]) / height
                            },
                        "mask": mask_b64,
                        "mask_width": int(mask.shape[1]),
                        "mask_height": int(mask.shape[0])
                    })
                return {
                    "type": self.task,
                    "observations" : obs,
                    "time": time.time() - start
                }
            else:
                start = time.time()
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
                    ],
                    "time": time.time() - start
                }
        else:
            print(f"predict() ERROR - Unknown model:", self.model_type)
            raise RuntimeError("Model not loaded")


def save_result_image_to_disk(model: str, results):
    annotated = results[0].plot()
    Image.fromarray(annotated).save(f"/outputs/{model}-{time.time()}.jpg")