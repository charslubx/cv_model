"""
服务配置文件
"""
import os
from pathlib import Path


class Settings:
    """服务配置类"""
    
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # 模型配置
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", 
        str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    )
    DEVICE: str = os.getenv("DEVICE", "cuda")
    NUM_CLASSES: int = int(os.getenv("NUM_CLASSES", "26"))
    
    # 推理配置
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "224"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
