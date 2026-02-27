import requests

url = "http://10.34.64.211:8000/smol"

with open("dog.jpeg", "rb") as f:
    response = requests.post(
        url,
        params={"object": "dog"},   # ← query parameter
        files={"file": ("dog.jpeg", f, "image/jpeg")}
    )

print(response.json())