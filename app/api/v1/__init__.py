"""
API v1版本
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, model

api_router = APIRouter()

# 注册路由
api_router.include_router(health.router, prefix="/health", tags=["健康检查"])
api_router.include_router(model.router, prefix="/model", tags=["模型推理"])
