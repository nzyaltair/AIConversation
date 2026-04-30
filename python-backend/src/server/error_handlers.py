from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "server_error",
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.error_type = error_type


def not_found(message: str = "Resource not found") -> ApiError:
    return ApiError(404, message, "not_found_error")


def invalid_request(message: str = "Invalid request") -> ApiError:
    return ApiError(400, message, "invalid_request_error")


def server_error(message: str = "Internal server error") -> ApiError:
    return ApiError(500, message, "server_error")


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    detail = ErrorDetail(
        message=exc.message,
        type=exc.error_type,
        code=str(exc.status_code),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": detail.model_dump()},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    detail = ErrorDetail(
        message="Internal server error",
        type="server_error",
        code="500",
    )
    return JSONResponse(
        status_code=500,
        content={"error": detail.model_dump()},
    )
