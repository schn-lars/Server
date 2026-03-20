from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated
from fastapi.responses import JSONResponse
from service.session import get_db
from service.users import CurrentUser
from sqlalchemy.orm import Session
from service.requestforms import UserRegistrationRequest, Token
from service import users

users_api_router = APIRouter(
    prefix="/api/users"
)

@users_api_router.post("/register")
def register_user(
    user_registration_request: UserRegistrationRequest,
    db: Session = Depends(get_db),
):
    try:
        users.register_user(
            username=user_registration_request.username,
            password=user_registration_request.password,
            db=db)
        return {
            "status": True,
            "message": "Registration was successful.",
        }
    except Exception as e:
        print(f"register_user: ERROR {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@users_api_router.post("/login", response_model=Token)
async def login_to_get_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db),
):
    try:
        return users.login_to_get_access_token(form_data=form_data, db=db)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Username and password do not match!",
        )


@users_api_router.get("/search")
async def get_users(
        current_user: CurrentUser,
        search_query: str,
        db: Session = Depends(get_db)
    ):
    try:
        return users.get_users(
            current_user=current_user,
            search_query=search_query,
            db=db
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
        )