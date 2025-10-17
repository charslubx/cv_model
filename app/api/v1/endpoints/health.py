"""
健康检查端点
"""
from fastapi import APIRouter
from datetime import datetime
from app.schemas.response import HealthResponse
from app.services.model_service import model_service
from app.core.config import settings

router = APIRouter()


@router.get("", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    检查服务健康状态
    
    返回:
        - status: 服务状态
        - timestamp: 当前时间戳
        - version: 服务版本
        - model_loaded: 模型是否已加载
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.VERSION,
        model_loaded=model_service.is_loaded
    )
