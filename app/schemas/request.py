"""
请求数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any


class PredictRequest(BaseModel):
    """模型推理请求"""
    
    data: Any = Field(..., description="输入数据，可以是图片URL、base64编码或其他格式")
    batch_size: Optional[int] = Field(None, description="批处理大小")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": "image_url_or_base64",
                "batch_size": 1
            }
        }


class BatchPredictRequest(BaseModel):
    """批量推理请求"""
    
    data_list: List[Any] = Field(..., description="批量输入数据列表")
    batch_size: Optional[int] = Field(None, description="批处理大小")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_list": ["image1", "image2", "image3"],
                "batch_size": 32
            }
        }
