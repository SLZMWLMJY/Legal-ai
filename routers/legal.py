import os
import uuid

from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from services.legal_service import LegalService
from models.legal_schemas import ChatRequest
from agents.legal_agent import generate_stream_response
from core.auth import get_current_user
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["法律智能问答助手"],
)

# 创建上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/stream")
async def chat_stream(request: ChatRequest,
                      legal_service: LegalService = Depends(LegalService),
                      current_user: Dict[str, Any] = Depends(get_current_user)):
    account_id = current_user['account_id']
    logger.info(f"用户{account_id}开始对话")

    return StreamingResponse(
        generate_stream_response(
            legal_service = legal_service,
            account_id = account_id,
            input_text = request.message),
        media_type="text/event-stream"
    )


@router.post("/image_analysis")
async def chat_with_multi_agent(message: str = Form(..., description="用户消息"),
                                account_id: str = Form("account_id", description="用户ID"),
                                image: Optional[UploadFile] = File(None, description="可选图像文件"),
                                legal_service: LegalService = Depends(LegalService)
                                ):
    """
    与多模态智能体对话（支持图像上传）
    """
    try:
        logger.info(f"用户 {account_id} 发送消息: {message}")

        # 处理图像上传
        image_path = None
        if image:
            # 验证文件类型
            allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
            file_extension = image.filename.split('.')[-1].lower() if image.filename else ''

            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {file_extension}。支持的格式: {', '.join(allowed_extensions)}"
                )

            # 生成唯一文件名
            filename = f"{uuid.uuid4().hex}.{file_extension}"
            image_path = os.path.join(UPLOAD_DIR, filename)

            # 保存图像文件
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)

            logger.info(f"图像已保存: {image_path}")

            # 将图像路径包含在消息中，让智能体来处理图像
            full_message = f"{message}\n\n[图像文件: {image_path}]"
        else:
            full_message = message

        # 异步生成流式响应
        async def generate():
            async for chunk in generate_stream_response(
                    legal_service=legal_service,
                    account_id=account_id,
                    input_text=full_message
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")