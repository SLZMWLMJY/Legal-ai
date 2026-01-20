from pydantic import BaseModel
from typing import Any, Optional, Literal


class JsonData(BaseModel):
    """通用响应数据模型"""
    code: int = 0
    data: Optional[Any] = None
    msg: Optional[str] = None
    type: Literal["stream", "text"] = "stream"

    @classmethod
    def success(cls, data: Any) -> "JsonData":
        """创建成功响应"""
        return cls(code=0, data=data, type="text")

    @classmethod
    def error(cls, msg: str = "error", code: int = -1) -> "JsonData":
        """创建错误响应"""
        return cls(code=code, msg=msg, type="text")

    @classmethod
    def stream_data(cls, data: Any, msg: str = "") -> "JsonData":
        """创建流式数据响应"""
        return cls(code=0, data=data, msg=msg, type="stream")
