from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from db_models import User


def build_auth_router(db):

    router = APIRouter(
        prefix="/auth",
        tags=["auth"],
    )

    @router.get("/me")
    async def me(
        authorization: Optional[str] = Header(None),
    ):
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Not logged in"
            )

        email = authorization.replace("Bearer ", "")
        user_id = email.split("@")[0]

        profile = await db.patient_profiles.find_one(
            {"user_id": user_id},
            {"_id": 0},
        )

        return {
            "user_id": user_id,
            "email": email,
            "name": user_id,
            "picture": None,
            "profile": profile,
        }

    @router.post("/logout")
    async def logout():
        return {"ok": True}

    async def _get_user_from_session():
        return None

    return router, _get_user_from_session


async def require_user(
    db,
    session_token=None,
    authorization=None,
):
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    email = authorization.replace("Bearer ", "")
    user_id = email.split("@")[0]

    return User(
        user_id=user_id,
        email=email,
        name=user_id,
        picture=None,
    )