import hmac
import os
import time

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2 = OAuth2PasswordBearer(tokenUrl="auth/login")
ALGORITHM = "HS256"


def _secret() -> str:
    return os.getenv("JWT_SECRET", "dev-only-change-me")


def check_credentials(username: str, password: str) -> bool:
    u, p = os.getenv("DEMO_USER", "analyst"), os.getenv("DEMO_PASSWORD", "fraud-demo")
    return hmac.compare_digest(username, u) and hmac.compare_digest(password, p)


def create_token(username: str, minutes: int = 60) -> str:
    return jwt.encode({"sub": username, "exp": int(time.time()) + minutes * 60}, _secret(), algorithm=ALGORITHM)


def current_user(token: str = Depends(oauth2)) -> str:
    try:
        return jwt.decode(token, _secret(), algorithms=[ALGORITHM])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
