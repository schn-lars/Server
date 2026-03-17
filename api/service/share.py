from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID
from service.requestforms import ShareInformationRequest, FetchSharedIdsRequest
from service.entities import SharedInformation, RetrievedInformation, User, ShareMapping
from service.users import CurrentUser
import uuid

def fetch_shared_info_by_ids(ids: list[str], db: Session):
    try:
        return db.query(RetrievedInformation).where(RetrievedInformation.id.in_(ids)).all()
    except Exception as e:
        print(f"Error in fetch_shared_info_by_id: {str(e)}")

def fetch_shared_by_proximity(request: FetchSharedIdsRequest, db: Session, radius_in_meters: int = 20) -> list[str]:
    try:
        return db.query(SharedInformation.id).where(SharedInformation.coord_x.between(SharedInformation.coord_x))
    except Exception as e:
        print(f"Error in fetch_shared_by_proximity: {str(e)}")

def insert_shared_information(request: ShareInformationRequest, db: Session):
    try:
        shared_info = SharedInformation(
            id=UUID(request.id),
            coord_x= request.coord_x,
            coord_y=request.coord_y
        )
        retrieved_info = RetrievedInformation(
            id=UUID(request.id),
            json=request.json
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