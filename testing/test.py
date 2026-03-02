import requests
from enum import Enum
from argparse import ArgumentParser

class Models(Enum):
    SMOL = 1
    MINI = 2

parser = ArgumentParser(
    prog='VLM-Testing',
    description='Testing performance of particular VLM in specified scenario.'
)
parser.add_argument('--model', type=int, help='(1) SMOL (2) MINI', default=1)

args = parser.parse_args()

model = Models(value=args.model)

match model:
    case Models.MINI:
        with open("dog.jpeg", "rb") as f:
            response = requests.post(
                "http://10.34.64.211:8000/minicpm-general",
                params={"object": "dog"},
                files={"file": ("dog-in-park.jpeg", f, "image/jpeg")}
            )
        print("General Request:")
        print(response.json())
    case Models.SMOL:
        with open("dog.jpeg", "rb") as f:
            response = requests.post(
                "http://10.34.64.211:8000/smol-general",
                params={"object": "dog"},
                files={"file": ("dog.jpeg", f, "image/jpeg")}
            )
        print("Specific Request:")
        print(response.json())