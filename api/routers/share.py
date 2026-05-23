from fastapi import APIRouter, Request, Depends, Form, UploadFile, File
from fastapi.responses import JSONResponse
from service.requestforms import ShareData, UUIDPayload, FetchSharedIdsRequest
import threading
from service import share
from service.session import get_db
from service.users import CurrentUser
from sqlalchemy.orm import Session
import json, os, shutil
import traceback

share_api_router = APIRouter(
    prefix="/api/share"
)

UPLOAD_DIR_CROPPED = "uploads/shared-crops"

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
async def share_info(
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


@share_api_router.post("/unpublish-info")
async def unpublish_shared_information(
        current_user: CurrentUser,
        shared_info_id: str,
        db: Session = Depends(get_db),
    ):
    try:
        share.unpublish_shared_information(
            current_user=current_user,
            shared_info_id=shared_info_id,
            db=db,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.post("/publish-info")
async def publish_shared_information(
        current_user: CurrentUser,
        shared_info_id: str,
        db: Session = Depends(get_db),
    ):
    try:
        share.publish_shared_information(
            current_user=current_user,
            shared_info_id=shared_info_id,
            db=db,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.get("/priviledged-users")
async def fetch_priviledged_users(
        current_user: CurrentUser,
        info_id: str,
        search_query: str = "",
        db: Session = Depends(get_db)
    ):
    try:
        share.fetch_priviledged_users(
            current_user=current_user,
            info_id=info_id,
            db=db,
            search_query=search_query
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.post("/remove-priviledged")
async def remove_all_priviledged_users(
        current_user: CurrentUser,
        info_id: str,
        db: Session = Depends(get_db)
    ):
    try:
        share.remove_all_priviledged_users(
            current_user=current_user,
            info_id=info_id,
            db=db
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@share_api_router.post("/add-priviledged-user")
async def add_user_to_shared_information(
        current_user: CurrentUser,
        user_id: str,
        shared_info_id: str,
        db: Session = Depends(get_db)
    ):
    try:
        share.add_user_to_shared_information(
            current_user=current_user,
            user_id=user_id,
            shared_info_id=shared_info_id,
            db=db
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.post("/add-priviledged-user")
async def remove_user_from_shared_information(
        current_user: CurrentUser,
        user_id: str,
        shared_info_id: str,
        db: Session = Depends(get_db)
    ):
    try:
        share.remove_user_from_shared_information(
            current_user=current_user,
            user_id=user_id,
            shared_info_id=shared_info_id,
            db=db
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@share_api_router.get("/info")
async def request_info_for_id(
    current_user: CurrentUser,
    info_id: str,
    db: Session = Depends(get_db)
):
    try:
        rows = share.fetch_shared_info_by_ids([info_id], db)

        if not rows:
            return JSONResponse(status_code=404, content={"error": "Not found"})

        shared, retrieved, user = rows[0]
        return {
            "id": str(shared.id),
            "owner": user.username,
            "obj": shared.object,
            "confidence": shared.confidence,
            "json": json.loads(retrieved.content_json)
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.get("/proximity-info")
async def request_info_for_id(
    current_user: CurrentUser,
    coord_x: float,
    coord_y: float,
    db: Session = Depends(get_db)
):
    try:
        rows = share.fetch_shared_by_proximity(
            request=FetchSharedIdsRequest(
                coord_x=coord_x,
                coord_y=coord_y
            ),
            db=db
        )

        #if not rows:
        #    return JSONResponse(status_code=404, content={"error": "Not found"})

        return \
        { "rows" : [
            {
                "id": str(shared.id),
                "owner": user.username,
                "obj": shared.object,
                "confidence": shared.confidence,
                "coord_x": shared.coord_x,
                "coord_y": shared.coord_y,
                "json": json.loads(retrieved.content_json),
                "image_url": f"/static/{shared.id}.jpg"
            }
            for shared, retrieved, user in rows]
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@share_api_router.post("/share-info")
async def share_object(
    current_user: CurrentUser,
    id: str = Form(...),
    label: str = Form(...),
    confidence: float = Form(...),
    coord_x: float = Form(...),
    coord_y: float = Form(...),
    content_json: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        ext = os.path.splitext(image.filename)[1] or ".jpg"
        filename = f"{id}{ext}"
        file_path = os.path.join(UPLOAD_DIR_CROPPED, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        share.share_object(
            id=id,
            current_user=current_user,
            label=label,
            confidence=confidence,
            coord_x=coord_x,
            coord_y=coord_y,
            content_json=content_json,
            db=db
        )

        db.commit()
        return { "status": True }
    except Exception as e:
        traceback.print_exc()
        db.rollback()
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse(content={"error": str(e)}, status_code=500)