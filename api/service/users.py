from .entities import User
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID
from .requestforms import TokenData, Token
from typing import Annotated
from fastapi import Depends, status, HTTPException

import jwt
from jwt import PyJWTError
from dotenv import load_dotenv

load_dotenv("/app/.env")

JWT_ENCODING = os.getenv("JWT_ENCODING")
ALGORITHM = "HS256"
TOKEN_TTL_IN_MINS = 131400 # 6 Months, do not do this usually

oauth_bearer = OAuth2PasswordBearer(tokenUrl="api/auth/token")
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(raw_password: str, hash_password: str) -> bool:
    return bcrypt_context.verify(raw_password, hash_password)


def get_password_hash(raw_password: str) -> str:
    return bcrypt_context.hash(raw_password)

def authenticate_user_with_username(
    username: str, raw_password: str, db: Session
) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        print(f"No matching user has been found for {username}")
        return None
    if not verify_password(raw_password=raw_password, hash_password=user.password_hash):
        print(f"Password does not match for {username}")
        return None
    return user


def create_access_token(userid: UUID, expiration: timedelta) -> str:
    if JWT_ENCODING is None:
        raise ValueError("JWT_ENCODING environment variable is not set")
    encode = {
        "id": str(userid),
        "exp": datetime.now(timezone.utc) + expiration,
    }
    return jwt.encode(encode, JWT_ENCODING, algorithm=ALGORITHM)


def register_user(username: str, password: str, db: Session):
    if 6 > len(password) or 20 < len(password):
        raise Exception("Password must be between 6 and 20 characters.")
    
    other_user = db.query(User).filter(User.username == username).first()
    if other_user is not None:
        raise Exception("This username already exists!")
    user = User(
        username=username,
        hashed_password=get_password_hash(raw_password=password),
    )
    db.add(user)
    db.commit()


def verify_token(token: str) -> TokenData:
    try:
        if JWT_ENCODING is None:
            raise ValueError("JWT_ENCODING environment variable is not set")
        payload = jwt.decode(token, JWT_ENCODING, algorithms=[ALGORITHM])
        userid: str = payload["id"]
        return TokenData(user_id=userid)
    except (PyJWTError, Exception) as e:
        print(f"verify_token: {str(e)}")
        raise Exception("Unable to authenticate user.")
    

def get_current_user(token: Annotated[str, Depends(oauth_bearer)]) -> TokenData:
    try:
        return verify_token(token)
    except Exception:
        # Calmly respond with 401 instead of full stack trace
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You are not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )


CurrentUser = Annotated[TokenData, Depends(get_current_user)]


def login_to_get_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session
) -> Token:
    try:
        user = authenticate_user_with_username(
            form_data.username, form_data.password, db
        )
        if not user:
            print(f"{form_data.username} is not authorized!")
            raise Exception()
        token = create_access_token(
            user.id, timedelta(minutes=TOKEN_TTL_IN_MINS)
        )
        return Token(access_token=token, token_type="bearer")
    except Exception as e:
        print(str(e))
        raise Exception("Error fetching access token!")