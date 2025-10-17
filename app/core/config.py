"""
应用配置
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """应用配置类"""
    
    # API配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "模型服务API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "基于FastAPI的模型推理服务"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS配置
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # 模型配置
    MODEL_PATH: Optional[str] = None
    MODEL_DEVICE: str = "cuda"  # cuda或cpu
    BATCH_SIZE: int = 32
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
