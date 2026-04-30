"""
FastAPI 应用工厂模块。

使用工厂模式创建应用实例：配置 → AppState → 中间件 → 路由 → 异常处理器。
模块级 `app = create_app()` 供 uvicorn 直接导入使用。
"""

from __future__ import annotations

import os
import sys
import logging

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── 修复 Windows 下 llama-cpp-python / ONNX Runtime CUDA 构建找不到 CUDA 运行时 DLL 的问题 ──
# llama.dll 依赖 cudart64_12.dll / cublas64_12.dll 等，需将 CUDA bin 目录加入 DLL 搜索路径
if sys.platform == "win32":
    import glob as _glob
    _cuda_found = False
    _cuda_bin_dir: str | None = None
    _logger = logging.getLogger(__name__)

    # 1) 检查 CUDA 环境变量（优先 bin\x64 — CUDA 13.x 目录结构）
    for _env_key in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        _val = os.environ.get(_env_key)
        if _val:
            _cuda_root = _val
            for _sub in (os.path.join("bin", "x64"), "bin"):
                _bin = os.path.join(_cuda_root, _sub)
                if os.path.isdir(_bin) and any(
                    _f.startswith("cudart") and _f.endswith(".dll")
                    for _f in os.listdir(_bin)
                ):
                    _cuda_bin_dir = _bin
                    _cuda_found = True
                    break
        if _cuda_found:
            break

    # 2) Glob 扫描 Program Files 下的 CUDA Toolkit 目录
    # CUDA 12.x 及更早：DLL 在 bin\；CUDA 13.x：DLL 在 bin\x64\
    if not _cuda_found:
        _base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        for _cuda_dir in sorted(
            _glob.glob(os.path.join(_base, "v*")), reverse=True
        ):
            for _sub in ("bin", os.path.join("bin", "x64")):
                _bin = os.path.join(_cuda_dir, _sub)
                if os.path.isdir(_bin) and any(
                    _f.startswith("cudart") and _f.endswith(".dll")
                    for _f in os.listdir(_bin)
                ):
                    _cuda_bin_dir = _bin
                    _cuda_found = True
                    break
            if _cuda_found:
                break

    if _cuda_found and _cuda_bin_dir:
        os.add_dll_directory(_cuda_bin_dir)
        os.environ["PATH"] = _cuda_bin_dir + os.pathsep + os.environ.get("PATH", "")
        _logger.info("已添加 CUDA DLL 搜索路径: %s", _cuda_bin_dir)
        # CUDA 13.x: bin\x64 含主要运行时 DLL，bin\ 也需加入以覆盖所有依赖
        if _cuda_bin_dir.endswith(os.path.join("bin", "x64")):
            _parent_bin = os.path.dirname(_cuda_bin_dir)  # .../bin
            if os.path.isdir(_parent_bin):
                os.add_dll_directory(_parent_bin)
                os.environ["PATH"] = _parent_bin + os.pathsep + os.environ.get("PATH", "")
    else:
        _logger.warning(
            "未找到 CUDA Toolkit bin 目录（检查了 CUDA_PATH/CUDA_HOME/CUDA_ROOT 环境变量 "
            "及 C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin）。"
            "llama-cpp-python GPU 后端可能不可用。"
        )

    # 3) 启动时报告 ONNX Runtime 可用的执行提供器（用于 GPU 调试）
    try:
        import onnxruntime as _ort
        _logger.info("ONNX Runtime 可用 providers: %s", _ort.get_available_providers())
    except Exception:
        pass

import asyncio
import json
import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.config import build_config
from server.app_state import AppState
from server.error_handlers import ApiError, api_error_handler, general_exception_handler
from server.api.router import build_router


async def release_semaphore(request: Request) -> None:
    app_state: AppState = request.app.state.app_state
    if getattr(request.state, "_semaphore_acquired", False):
        app_state.request_semaphore.release()


def create_app() -> FastAPI:
    config = build_config()
    app_state = AppState(config)

    origins = list(config.cors_origins)

    # ── 应用生命周期 ──
    # 启动时：初始化 AppState（Store + 种子模型目录）
    # 关闭时：释放所有 Store 的数据库连接
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app_state.initialize()
        app.state.app_state = app_state
        yield
        await app_state.shutdown()

    app = FastAPI(title="AI Conversation Server", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 请求追踪中间件 ──
    # 为每个请求分配或继承 X-Request-ID，用于日志关联和分布式追踪
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response

    # ── 请求并发控制中间件 ──
    # 通过 asyncio.Semaphore 限制最大并发请求数，超时返回 HTTP 503（服务降级）
    @app.middleware("http")
    async def semaphore_middleware(request: Request, call_next):
        sem = app_state.request_semaphore
        try:
            await asyncio.wait_for(sem.acquire(), timeout=config.request_timeout_secs)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Server is busy, try again", "type": "server_error", "code": "503"}},
            )
        try:
            response = await call_next(request)
        finally:
            sem.release()
        return response

    # ── 健康检查 & 监控端点 ──
    @app.get("/health")
    async def health():
        """存活检查端点，返回服务版本信息。"""
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/metrics")
    async def metrics():
        """基础监控端点（待完善 Prometheus 指标）。"""
        return {"uptime": time.time(), "requests": 0}

    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    build_router(app, app_state)

    return app


# 模块级应用实例，供 `uvicorn server.main:app` 直接导入
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = build_config()
    uvicorn.run(
        "server.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )
