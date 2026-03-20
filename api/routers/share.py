from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from service.requestforms import ShareData, UUIDPayload
import threading
from service import share
from service.session import get_db
from service.users import CurrentUser
from sqlalchemy.orm import Session

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