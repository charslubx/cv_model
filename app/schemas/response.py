"""
响应数据模型
"""
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from datetime import datetime


class PredictResponse(BaseModel):
    """模型推理响应"""
    
    success: bool = Field(..., description="请求是否成功")
    result: Any = Field(..., description="推理结果")
    message: Optional[str] = Field(None, description="响应消息")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {"prediction": "result"},
                "message": "推理成功",
                "processing_time": 0.123
            }
        }


class BatchPredictResponse(BaseModel):
    """批量推理响应"""
    
    success: bool = Field(..., description="请求是否成功")
    results: List[Any] = Field(..., description="批量推理结果列表")
    message: Optional[str] = Field(None, description="响应消息")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    total_count: int = Field(..., description="总数量")


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(..., description="时间戳")
    version: str = Field(..., description="版本号")
    model_loaded: bool = Field(..., description="模型是否已加载")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00",
                "version": "0.1.0",
                "model_loaded": True
            }
        }
