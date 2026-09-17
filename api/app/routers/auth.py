from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.models import User
from app.schemas import Credentials, Token, UserOut
from app.security import DB, CurrentUser, create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _authenticate(db, username: str, password: str) -> User:
    user = db.scalar(select(User).where(User.username == username.lower()))
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")
    return user


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(body: Credentials, db: DB) -> Token:
    username = body.username.lower()
    if db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(status.HTTP_409_CONFLICT, "That username is taken")
    user = User(username=username, password_hash=hash_password(body.password), favorite_genres=[])
    db.add(user)
    db.commit()
    return Token(access_token=create_access_token(user.id))


@router.post("/login", response_model=Token)
def login(body: Credentials, db: DB) -> Token:
    return Token(access_token=create_access_token(_authenticate(db, body.username, body.password).id))


@router.post("/token", response_model=Token, include_in_schema=False)
def token(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DB) -> Token:
    """OAuth2 form endpoint so the "Authorize" button in /docs works."""
    return Token(access_token=create_access_token(_authenticate(db, form.username, form.password).id))


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
