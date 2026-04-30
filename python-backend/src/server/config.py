"""
应用配置模块。

通过环境变量（前缀 `AI_SERVER_`）注入配置，支持各平台的默认路径策略。
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServeConfig(BaseSettings):
    """应用运行时配置，所有字段可通过环境变量 `AI_SERVER_<FIELD>` 覆盖。"""
    model_config = SettingsConfigDict(
        env_prefix="AI_SERVER_",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    backend: str = "auto"
    db_path: str = ""
    media_dir: str = ""
    cors_origins: list[str] = [
        "http://localhost:3000",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ]
    models_dir: str = ""
    max_concurrent_requests: int = 32
    request_timeout_secs: int = 300


def _default_db_path() -> str:
    """计算 SQLite 数据库文件的默认路径。

    各平台路径：
    - Windows: %LOCALAPPDATA%/AIConversation/
    - macOS:   ~/Library/Application Support/AIConversation/
    - Linux:   $XDG_DATA_HOME/AIConversation/（回退 ~/.local/share/AIConversation/）
    """
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return str(base / "AIConversation" / "ai-conversation.sqlite3")


def _default_media_dir() -> str:
    """计算媒体文件（录音、合成语音等）的默认存储目录，路径策略同 `_default_db_path`。"""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.uname().sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return str(base / "AIConversation" / "media")


def _default_models_dir() -> str:
    """计算模型文件存储目录，默认位于项目根目录下的 `models/` 文件夹。"""
    return str(Path(__file__).resolve().parent.parent.parent.parent / "models")


def build_config() -> ServeConfig:
    """构建配置实例，填充所有缺失的默认值并确保目录存在。"""
    config = ServeConfig()
    if not config.db_path:
        config.db_path = _default_db_path()
    if not config.media_dir:
        config.media_dir = _default_media_dir()
    if not config.models_dir:
        config.models_dir = _default_models_dir()
    os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
    os.makedirs(config.media_dir, exist_ok=True)
    os.makedirs(config.models_dir, exist_ok=True)
    return config
