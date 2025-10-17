"""
FastAPI主应用
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.api.v1 import api_router


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    @app.on_event("startup")
    async def startup_event():
        """应用启动事件"""
        logger.info(f"启动 {settings.PROJECT_NAME} v{settings.VERSION}")
        logger.info(f"文档地址: http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}/docs")
        
        # 可选：启动时自动加载模型
        # from app.services.model_service import model_service
        # if settings.MODEL_PATH:
        #     try:
        #         model_service.load_model()
        #     except Exception as e:
        #         logger.error(f"启动时加载模型失败: {str(e)}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭事件"""
        logger.info("关闭应用...")
        # 清理资源
        from app.services.model_service import model_service
        if model_service.is_loaded:
            model_service.unload_model()
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
