"""
FastAPI服务主文件
用于模型推理服务
"""
import io
import os
import sys
import logging
from typing import Dict, List
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from service.model_loader import ModelInference
from service.config import Settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载配置
settings = Settings()

# 创建FastAPI应用
app = FastAPI(
    title="服装属性多标签分类服务",
    description="基于GAT+GCN的服装属性多标签分类模型推理服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例
model_inference = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_inference
    try:
        logger.info("正在加载模型...")
        model_inference = ModelInference(
            model_path=settings.MODEL_PATH,
            device=settings.DEVICE,
            num_classes=settings.NUM_CLASSES
        )
        logger.info("模型加载成功！")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理资源"""
    logger.info("正在关闭服务...")


@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "服装属性识别服务",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model_inference is not None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5
):
    """
    图片多标签分类接口
    
    Args:
        file: 上传的图片文件
        threshold: 分类阈值（0-1之间），默认0.5
        
    Returns:
        分类结果，包含：
        - attributes: 所有属性的置信度
        - classifications: 所有属性的分类结果（0或1）
        - positive_attributes: 预测为正的属性列表
        - top_k_attributes: Top-K置信度最高的属性
        - statistics: 统计信息
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")
    
    # 验证阈值范围
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="阈值必须在0-1之间")
    
    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 进行分类
        logger.info(f"正在分类图片: {file.filename}, 阈值: {threshold}")
        predictions = model_inference.predict(image, threshold=threshold)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"分类失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold: float = 0.5
):
    """
    批量图片多标签分类接口
    
    Args:
        files: 上传的图片文件列表
        threshold: 分类阈值（0-1之间），默认0.5
        
    Returns:
        分类结果列表
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"批量分类最多支持{settings.MAX_BATCH_SIZE}张图片"
        )
    
    # 验证阈值范围
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="阈值必须在0-1之间")
    
    results = []
    
    for file in files:
        # 验证文件类型
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "不是图片文件"
            })
            continue
        
        try:
            # 读取图片
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # 进行分类
            predictions = model_inference.predict(image, threshold=threshold)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions
            })
            
        except Exception as e:
            logger.error(f"分类失败 ({file.filename}): {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total": len(files),
        "results": results
    })


@app.get("/attributes")
async def get_attributes():
    """
    获取所有属性名称
    
    Returns:
        属性名称列表
    """
    if model_inference is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "attributes": model_inference.get_attribute_names(),
        "num_classes": settings.NUM_CLASSES
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )
