from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from service.requestforms import UploadRequest, ReverseRequest
from service import resources

img_folder = Path("images")
img_folder.mkdir(parents=True, exist_ok=True)

resources_api_router = APIRouter(
    prefix="/api/resources"
)

@resources_api_router.get("/images/{uuid}.jpg")
@resources_api_router.head("/images/{uuid}.jpg")
async def get_image(uuid: str, request: Request):
    print(f"GET image {uuid}.jpg from {request.client.host} or {request.headers}")

    try:
        file_path = img_folder / f"{uuid}.jpg"
        if file_path.exists():
            response = FileResponse(file_path, media_type="image/jpeg")
            response.headers["Content-Type"] = "image/jpeg"
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Cache-Control"] = "public, max-age=900, immutable"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Strict-Transport-Security"] = "max-age=300"
            response.headers["Accept-Ranges"] = "bytes"
            return response
        return JSONResponse(content={"message": "Image not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@resources_api_router.post("/images/upload")
async def upload_image(
        request: UploadRequest
    ):
    try:
        resources.save_image(id=request.id, image_data=request.img)
        return JSONResponse(content={"url": f"https://myurl.com/images/{request.id}.jpg"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)