"""API 라우터 패키지."""

from fastapi import APIRouter

from .auth_routes import router as auth_router
from .routes import router as public_router

router = APIRouter()
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(public_router)

__all__ = ["router"]
