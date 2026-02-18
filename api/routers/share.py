from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from service.requestforms import ShareData, UUIDPayload
import threading

share_api_router = APIRouter(
    prefix="/api/share"
)

shared_items = {}
share_lock_holmes = threading.Lock()

@share_api_router.post("/fetch", response_model=ShareData)
async def fetch(
        payload: UUIDPayload
    ):
    try:
        with share_lock_holmes:
            info = shared_items[payload.uuid]
            return info
    except KeyError:
        return JSONResponse(content={"error": "The given ID is not being shared."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@share_api_router.post("/share")
async def share(
        request: Request
    ):
    try:
        body = await request.body()
        data = ShareData.model_validate_json(body)  # now parse explicitly
        print(f"Upload for object: {data.id}")
        with share_lock_holmes:
            shared_items[data.id] = data
            print(f"Current state of shared infos: {shared_items.keys()}")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@share_api_router.post("/exitshare")
async def exitshare(
        payload: UUIDPayload
    ):
    try:
        print(f"Exit share for object: {payload.uuid}")
        with share_lock_holmes:
            shared_items.pop(payload.uuid)
            print(f"Current state of shared infos: {shared_items.keys()}")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)