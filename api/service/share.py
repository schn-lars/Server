from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID
from service.requestforms import ShareInformationRequest, FetchSharedIdsRequest
from service.entities import SharedInformation, RetrievedInformation, User, ShareMapping
from service.users import CurrentUser
import uuid
import math

def fetch_shared_info_by_ids(ids: list[str], db: Session):
    try:
        return (
            db.query(SharedInformation, RetrievedInformation, User)
            .join(RetrievedInformation, RetrievedInformation.id == SharedInformation.id)
            .join(User, User.id == SharedInformation.user_id)
            .filter(SharedInformation.public == True)
            .filter(SharedInformation.id.in_(ids))
            .all()
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def fetch_shared_by_proximity(request: FetchSharedIdsRequest, db: Session, radius_in_meters: int = 20) -> list[str]:
    try:
        center_lat = request.coord_y
        center_lon = request.coord_x

        latMetersPerDeg = 111_100
        longMetersPerDeg = 111_100 * math.cos(math.radians(center_lat))

        lat_delta = radius_in_meters / latMetersPerDeg # this is a box around essentially, however in a small scale I guess it works
        lon_delta = radius_in_meters / longMetersPerDeg
        return (
            db.query(SharedInformation, RetrievedInformation, User)
            .join(RetrievedInformation, RetrievedInformation.id == SharedInformation.id)
            .join(User, User.id == SharedInformation.user_id)
            .filter(SharedInformation.public == True)
            .filter(SharedInformation.coord_y.between(
                center_lat - lat_delta,
                center_lat + lat_delta
            ))
            .filter(SharedInformation.coord_x.between(
                center_lon - lon_delta,
                center_lon + lon_delta
            ))
            .limit(5)
            .all()
        )
    except Exception as e:
        print(f"Error in fetch_shared_by_proximity: {str(e)}")
        return []

def insert_shared_information(request: ShareInformationRequest, db: Session):
    try:
        shared_info = SharedInformation(
            id=UUID(request.id),
            coord_x= request.coord_x,
            coord_y=request.coord_y
        )
        retrieved_info = RetrievedInformation(
            id=UUID(request.id),
            content_json=request.content_json
        )
        db.add_all([shared_info, retrieved_info])
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in insert_shared_information: {str(e)}")

def remove_shared_information(id: str, db: Session):
    try:
        obj = db.query(SharedInformation).get(UUID(id))
        info = db.query(RetrievedInformation).get(UUID(id))
        if obj:
            db.delete(obj)
            db.delete(info)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error in remove_shared_information: {str(e)}")

def add_user_to_shared_information(current_user: CurrentUser, user_id: str, shared_info_id: str, db: Session):
    user = db.query(User).get(user_id)
    if user is None:
        raise Exception("User does not exist.")
    
    # Check if the current_user is allowed to do that
    shared_info = db.query(SharedInformation)\
            .filter(SharedInformation.user_id == current_user.get_uuid())\
            .filter(SharedInformation.id == uuid.UUID(shared_info_id))\
            .filter(SharedInformation.public == False)\
            .first()

    if shared_info is None:
        raise Exception("The shared information does not belong to the user or is already public.")

    new_share = ShareMapping(
        shared_to_id=user_id,
        info_id=shared_info_id,
    )
    db.add(new_share)
    db.commit()


def remove_user_from_shared_information(current_user: CurrentUser, user_id: str, shared_info_id: str, db: Session):
    user = db.query(User).get(user_id)
    if user is None:
        raise Exception("User does not exist.")
    
    # Check if the current_user is allowed to do that
    shared_info = db.query(SharedInformation)\
            .filter(SharedInformation.user_id == current_user.get_uuid())\
            .filter(SharedInformation.id == uuid.UUID(shared_info_id))\
            .filter(SharedInformation.public == False)\
            .first()

    if shared_info is None:
        raise Exception("The shared information does not belong to the user or is already public.")
    share_mapping = db.query(ShareMapping)\
            .filter(ShareMapping.shared_to_id == user_id)\
            .filter(ShareMapping.info_id == shared_info_id)\
            .first()
    if share_mapping is not None:
        db.delete(share_mapping)
        db.commit()


def publish_shared_information(current_user: CurrentUser, shared_info_id: str, db: Session):    
    # Check if the current_user is allowed to do that
    shared_info = db.query(SharedInformation)\
            .filter(SharedInformation.user_id == current_user.get_uuid())\
            .filter(SharedInformation.id == uuid.UUID(shared_info_id))\
            .filter(SharedInformation.public == False)\
            .first()
    if shared_info is not None:
        shared_info.public = True
        db.refresh(shared_info)
        db.commit()
    else:
        print("Information is either already public or does not belong to this user.")


def unpublish_shared_information(current_user: CurrentUser, shared_info_id: str, db: Session):
    # Check if the current_user is allowed to do that
    shared_info = db.query(SharedInformation)\
            .filter(SharedInformation.user_id == current_user.get_uuid())\
            .filter(SharedInformation.id == uuid.UUID(shared_info_id))\
            .filter(SharedInformation.public == True)\
            .first()
    if shared_info is not None:
        shared_info.public = False
        db.refresh(shared_info)
        db.commit()
    else:
        print("Information is either already not public or does not belong to this user.")


def fetch_priviledged_users(
        current_user: CurrentUser,
        info_id: str,
        db: Session,
        search_query: str = "",
    ):
    info = db.query(SharedInformation).get(info_id)
    if info is None or info.user_id != current_user.get_uuid():
        raise Exception("This information does not exist or does not belong to you.")

    rows = db.execute(
        select(User.username, User.id)
        .join(User, User.id == ShareMapping.shared_to_id)
        .where(ShareMapping.info_id == info_id)
        .where(User.username.ilike(f"{search_query}%"))
        .limit(20)
    )

    privledged_users = [
        {"username" : username, "user_id" : id } for username, id in rows
    ]
    return { "priviledged_users" : privledged_users }


def remove_all_priviledged_users(
        current_user: CurrentUser,
        info_id: str,
        db: Session
    ):
    info = db.query(SharedInformation).get(info_id)
    if info is None or info.user_id != current_user.get_uuid():
        raise Exception("This information does not exist or does not belong to you.")
    
    db.execute(
        delete(ShareMapping)
        .where(ShareMapping.info_id == info_id)
    )
    db.commit()

def request_info_for_id(
        current_user: CurrentUser,
        info_id: str,
        db: Session
    ):
    info = fetch_shared_info_by_ids(ids=[info_id], db=db)
    return info

def share_object(
    current_user: CurrentUser,
    id: str,
    label: str,
    confidence: float,
    coord_x: float,
    coord_y: float,
    content_json: str,
    db: Session
) -> str | None:
    shared = SharedInformation(
        id=id,
        user_id=current_user.user_id,
        object=label,
        confidence=confidence,
        coord_x=coord_x,
        coord_y=coord_y
    )
    db.add(shared)
    db.flush()

    retrieved = RetrievedInformation(
        id=shared.id,
        content_json=content_json
    )
    db.add(retrieved)
    return str(id)