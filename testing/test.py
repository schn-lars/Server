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

'''
with open("dog.jpeg", "rb") as f:
    response = requests.post(
        "http://10.34.64.211:8000/" + get_url_path(type=type),
        params={"object": "dog"},
        files={"file": ("dog-in-park.jpeg", f, "image/jpeg")}
    )
    print("General Request:")
    print(response.json())
'''


##  RUN THIS WITH PYTHON VERSION 3.12.9
## coremltools==9.0
def export_yolo_model(words: list[str]):
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8s-worldv2.pt")  # or yolov8m-worldv2.pt
    model.set_classes(words)  # your vocab

    # Export to CoreML
    model.export(format="coreml", nms=True)

# Code from raphael

from pathlib import Path

# ==========================
# CONFIG
# ==========================

WORDS_FILE = Path("yolo_world_objects_2k.txt")  # your 6k-word list
OUTPUT_FILE = Path("yolo_world_objects_2k.txt")
TARGET_CLASSES = 2000

# If your API expects a different JSON shape, adjust `get_clip_embedding` below.


# ==========================
# HEURISTIC PREFILTERING
# ==========================

BANNED = {
    "fuck", "shit", "ass", "fisting",  # NSFW / offensive
    "atm", "cd", "suv", "tv",          # generic acronyms
}

ABSTRACT_SUFFIXES = (
    "ness", "ity", "ism", "ship", "ment", "ance", "ence", "hood",
    "tion", "sion", "acy", "age", "ery", "dom", "tude",
    "ology", "graphy", "ography", "work", "ware",
    "omics", "ism", "ist",
)

VERBISH_SUFFIXES = ("ing", "ment")

YOLO_KEY_OBJECTS = {
    # people / animals
    "person", "man", "woman", "boy", "girl", "baby",
    "dog", "cat", "bird", "cow", "horse", "sheep", "goat", "pig",
    "chicken", "duck", "fish", "elephant", "bear", "zebra", "giraffe",
    # vehicles
    "bicycle", "bike", "motorcycle", "scooter",
    "car", "truck", "bus", "van", "train", "tram", "subway",
    "airplane", "plane", "helicopter", "boat", "ship",
    # common indoor stuff
    "cup", "mug", "bottle", "glass", "bowl", "plate",
    "fork", "knife", "spoon", "chopsticks",
    "table", "desk", "chair", "sofa", "couch", "bed", "door", "window",
    "tv", "monitor", "screen", "phone", "cellphone", "laptop",
    "keyboard", "mouse", "remote", "book",
    # clothes & accessories
    "hat", "helmet", "shoe", "boot", "sneaker",
    "coat", "jacket", "shirt", "tshirt", "t-shirt", "pants", "jeans", "skirt",
    "backpack", "bag", "handbag", "wallet", "watch",
    # food
    "pizza", "burger", "sandwich", "hotdog", "hot-dog",
    "apple", "banana", "orange", "grape", "strawberry",
    "broccoli", "carrot", "potato", "tomato",
    "cake", "cookie", "donut", "doughnut", "icecream", "ice-cream",
    # sports / misc
    "ball", "football", "soccer", "basketball", "tennis", "frisbee",
    "skateboard", "surfboard", "kite", "bat", "racket",
    # traffic-ish
    "trafficlight", "traffic-light", "stop-sign", "stopsign",
}


def clean_word(w: str) -> str:
    w = w.strip()
    if not w:
        return ""
    # strip trailing punctuation like "zucchini."
    while w and not w[-1].isalnum():
        w = w[:-1]
    return w


def is_candidate(token: str) -> bool:
    w = token.strip()
    if not w:
        return False

    w_lower = w.lower()

    # banned words
    if w_lower in BANNED:
        return False

    # digits: ignore things like "mRNA" as labels
    if any(ch.isdigit() for ch in w_lower):
        return False

    # no spaces: we keep it single-token for now
    if " " in w_lower:
        return False

    # only letters / hyphen
    if not all(c.isalpha() or c == "-" for c in w_lower):
        return False

    # prefer lower-case common nouns (but allow whitelisted tokens)
    if w != w_lower and w_lower not in YOLO_KEY_OBJECTS:
        return False

    # length filter – avoid "to", "in", etc.
    if len(w_lower) <= 2 and w_lower not in {"ox"}:
        return False

    # soft filter: abstract suffixes
    if w_lower not in YOLO_KEY_OBJECTS:  # don't drop whitelisted
        if any(w_lower.endswith(s) for s in ABSTRACT_SUFFIXES):
            return False
        if any(w_lower.endswith(s) for s in VERBISH_SUFFIXES):
            return False

    return True


def prefer_singular(words):
    """
    Remove simple plurals when singular exists: dogs -> dog.
    """
    word_map = {w.lower(): w for w in words}
    result = []

    for original in words:
        w = original.lower()

        # never drop whitelisted classes
        if w in YOLO_KEY_OBJECTS:
            result.append(original)
            continue

        singular = None
        if w.endswith("ies") and len(w) > 3:
            singular = w[:-3] + "y"
        elif w.endswith("es") and len(w) > 3:
            singular = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            singular = w[:-1]

        if singular and singular in word_map:
            # skip plural form
            continue

        result.append(original)

    return result


def heuristic_prefilter(words):
    cleaned = [clean_word(w) for w in words]
    candidates = [w for w in cleaned if is_candidate(w)]

    # deduplicate (preserve first occurrence)
    seen = set()
    uniq = []
    for w in candidates:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            uniq.append(w)

    # prefer singular forms
    singular_pref = prefer_singular(uniq)
    return singular_pref

raw_words = [
    w.strip()
    for w in WORDS_FILE.read_text(encoding="utf-8").splitlines()
    if w.strip()
]
print(f"Loaded {len(raw_words)} raw words.")

# 2) Heuristic prefiltering
filtered = heuristic_prefilter(raw_words)
print(f"After heuristic prefilter: {len(filtered)} candidate words.")

print(filtered)
#export_yolo_model(words=filtered)