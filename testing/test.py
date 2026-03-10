import requests
from enum import Enum
from argparse import ArgumentParser

class InferenceType(Enum):
    GENERAL = 1
    SPECIFIC = 2
    LOCATION = 3

parser = ArgumentParser(
    prog='VLM-Testing',
    description='Testing performance of particular VLM in specified scenario.'
)
parser.add_argument('--type', type=int, help='(1) general (2) specific (3) location', default=1)

args = parser.parse_args()

type = InferenceType(value=args.type)

def get_url_path(type: InferenceType):
    match type:
        case InferenceType.GENERAL:
            return "general"
        case InferenceType.SPECIFIC:
            return "specific"
        case InferenceType.LOCATION:
            return "location"


from ultralytics import YOLO

model = YOLO("yolo26m-seg.pt")
model.export(format="coreml")