"""
模型推理端点
"""
from fastapi import APIRouter, HTTPException
from app.schemas.request import PredictRequest, BatchPredictRequest
from app.schemas.response import PredictResponse, BatchPredictResponse
from app.services.model_service import model_service
from app.core.logging import logger
import time

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="单个样本推理")
async def predict(request: PredictRequest):
    """
    对单个样本进行推理
    
    参数:
        - data: 输入数据
        - batch_size: 批处理大小（可选）
        
    返回:
        - success: 是否成功
        - result: 推理结果
        - message: 响应消息
        - processing_time: 处理时间
    """
    try:
        start_time = time.time()
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        result = model_service.predict(request.data)
        processing_time = time.time() - start_time
        
        return PredictResponse(
            success=True,
            result=result,
            message="推理成功",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@router.post("/batch_predict", response_model=BatchPredictResponse, summary="批量推理")
async def batch_predict(request: BatchPredictRequest):
    """
    对批量样本进行推理
    
    参数:
        - data_list: 输入数据列表
        - batch_size: 批处理大小（可选）
        
    返回:
        - success: 是否成功
        - results: 推理结果列表
        - message: 响应消息
        - processing_time: 处理时间
        - total_count: 总数量
    """
    try:
        start_time = time.time()
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        results = model_service.batch_predict(request.data_list, request.batch_size)
        processing_time = time.time() - start_time
        
        return BatchPredictResponse(
            success=True,
            results=results,
            message="批量推理成功",
            processing_time=processing_time,
            total_count=len(results)
        )
        
    except Exception as e:
        logger.error(f"批量推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


@router.post("/load", summary="加载模型")
async def load_model(model_path: str = None):
    """
    加载模型
    
    参数:
        - model_path: 模型路径（可选，默认使用配置中的路径）
        
    返回:
        成功或失败信息
    """
    try:
        model_service.load_model(model_path)
        return {
            "success": True,
            "message": "模型加载成功",
            "model_loaded": model_service.is_loaded
        }
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@router.post("/unload", summary="卸载模型")
async def unload_model():
    """
    卸载模型，释放资源
    
    返回:
        成功信息
    """
    try:
        model_service.unload_model()
        return {
            "success": True,
            "message": "模型已卸载",
            "model_loaded": model_service.is_loaded
        }
    except Exception as e:
        logger.error(f"模型卸载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")
