from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID
from service.requestforms import ShareInformationRequest, FetchSharedIdsRequest
from service.entities import SharedInformation, RetrievedInformation

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
