"""API route handlers."""
from .videos import router as videos_router
from .metrics import router as metrics_router
from .chat import router as chat_router
from .settings import router as settings_router

__all__ = ["videos_router", "metrics_router", "chat_router", "settings_router"]
