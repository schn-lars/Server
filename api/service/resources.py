from pathlib import Path
from PIL import Image
import io, os, time, base64, threading

img_folder = Path("images")
img_folder.mkdir(parents=True, exist_ok=True)

image_deletion_dict = []
deletor_lock = threading.Lock()
DELETE_TIME = 900

def save_image(id: str, image_data: str):
    global image_deletion_dict
    global deletor_lock
    try:
        image_data = base64.b64decode(image_data)
        filename = f"{id}.jpg"
        file_path = img_folder / filename
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image.save(file_path, "JPEG")
        with deletor_lock:
            image_deletion_dict.append((id, time.time() + DELETE_TIME))
    except:
        raise Exception()

def remove_image(id):
    file_path = img_folder / f"{id}.jpg"
    if file_path.exists():
        os.remove(file_path)
        jpg_count = sum(1 for f in img_folder.glob("*.jpg"))
        print(f"Removed image {id}. Now we have a count of {jpg_count} .jpg files")

def clear_images():
    for file in img_folder.iterdir():
        os.remove(file)

def delete_temp_file(path: Path, delay: int = 10):
    time.sleep(delay)
    try:
        if path.exists():
            os.remove(path)
            print(f"File deleted: {path}")
    except Exception as e:
        print(f"Error deleting file: {e}")

def deletor():
    global image_deletion_dict
    global deletor_lock
    time_to_sleep = DELETE_TIME
    while True:
        with deletor_lock:
            if len(image_deletion_dict) == 0:
                time_to_sleep = DELETE_TIME
            else:
                id, timestamp = image_deletion_dict[0]
                if time.time() > timestamp:
                    # remove this item
                    image_deletion_dict.pop(0)
                    remove_image(id)
                    if len(image_deletion_dict) == 0:
                        time_to_sleep = DELETE_TIME
                    else:
                        next_id, next_timestamp = image_deletion_dict[0]
                        time_to_sleep = min(DELETE_TIME, max(next_timestamp - time.time(), 0))
        time.sleep(time_to_sleep)