from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from typing import Annotated
from fastapi import Depends

db_name = os.getenv("LOCATION_POSTGRES_DB", "default_db")
db_user = os.getenv("LOCATION_POSTGRES_USER", "default_user")
db_password = os.getenv("LOCATION_POSTGRES_PASSWORD", "default_password")
db_host = os.getenv("LOCATION_API_HOST", "default_host")
db_port = os.getenv("LOCATION_DB_PORT", "5432")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

LocationBase = declarative_base()

def get_location_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DBSession = Annotated[Session, Depends(get_location_db)]