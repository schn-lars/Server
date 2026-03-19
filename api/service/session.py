from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from typing import Annotated
from fastapi import Depends
from dotenv import load_dotenv
import time

load_dotenv("/app/.env")

db_name = os.getenv("POSTGRES_DB", "default_db")
db_user = os.getenv("POSTGRES_USER", "default_user")
db_password = os.getenv("POSTGRES_PASSWORD", "default_password")
db_host = os.getenv("POSTGRES_HOST", "default_host")
db_port = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = None
for i in range(10):
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            print("Connected to DB")
        break
    except Exception as e:
        print(f"DB not ready, retrying... ({i})", e)
        time.sleep(3)

if engine is None:
    raise RuntimeError("Could not connect to database after retries")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DBSession = Annotated[Session, Depends(get_db)]