from openai import OpenAI

from core.config import settings
from langchain_openai import ChatOpenAI

from core.multi_llm import OpenAISDKWrapper


def get_default_llm():
    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        streaming=settings.LLM_STREAMING
    )


def get_multimodal_llm():
    """获取多模态LLM（使用OpenAISDKWrapper保持LangChain兼容性）"""
    if settings.QWEN_API_KEY and settings.QWEN_MODEL_NAME:
        return OpenAISDKWrapper()  # 使用包装器
    else:
        return get_default_llm()
