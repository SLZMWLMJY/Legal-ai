import os
import base64
import json
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from openai import OpenAI


# ============ 定义数据模型 ============
class ImageAnalysisRequest(BaseModel):
    """图像分析请求模型"""
    image_url: str = Field(..., description="图像文件的路径")
    analysis_type: str = Field(
        default="general",
        description="分析类型：general（通用分析）、detailed（详细分析）、text（文字提取）、scene（场景理解)"
    )


class ImageAnalysisResponse(BaseModel):
    """图像分析响应模型"""
    success: bool = Field(..., description="是否成功")
    description: str = Field(..., description="图像描述")
    metadata: Dict = Field(default_factory=dict, description="元数据")


