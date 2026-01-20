import base64
import logging
import os

from langchain_community.utilities import SearchApiAPIWrapper
from langchain_core.tools import tool
from openai import OpenAI

from core.config import settings
from core.llm import get_multimodal_llm
from models.multi_schemas import ImageAnalysisRequest

os.environ["SEARCHAPI_API_KEY"] = settings.SEARCHAPI_API_KEY

logger = logging.getLogger(__name__)


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """
    使用此工具搜索最新的互联网信息。当你需要获取实时信息或不确定某个事实时使用
    """
    try:
        search = SearchApiAPIWrapper()
        results = search.results(query)
        return "\n\n".join([
            f"来源：{res['title']}\n内容：{res['snippet']}"
            for res in results['organic_results']
        ])
    except Exception as e:
        return f"搜索失败：{str(e)}"


# 图像分析工具
@tool("image_analysis", return_direct=False)
def image_analysis(request_data) -> str:
    """
    分析图像内容并返回详细描述。当你需要理解图像中有什么内容时使用
    """
    try:

        # 使用Pydantic模型验证输入
        request = ImageAnalysisRequest(**request_data)

        # 验证文件是否存在
        if not os.path.exists(request.image_url):
            return f"错误：图像文件不存在 - {request.image_url}"

        # 直接使用 OpenAI SDK 调用千问 VL
        client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL
        )

        # 读取并编码图像
        with open(request.image_url, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 调用千问VL模型，不使用流式响应以避免处理复杂的异步生成器
        response = client.chat.completions.create(
            model="qwen3-vl-plus",  # 确保使用正确的多模态模型名称
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请详细描述这张图像的内容，包括场景、物体、人物、颜色、氛围、构图等所有细节。用中文回答。"
                        }
                    ]
                }
            ],
            max_tokens=2000
        )

        description = response.choices[0].message.content
        return f"📷 图像分析结果：\n\n{description}"

    except Exception as e:
        logger.error(f"图像分析详细错误: {str(e)}")
        return f"图像分析失败：{str(e)}"


def get_tools():
    """
    获取可用的工具列表
    """
    tools = [
        web_search,
        image_analysis
    ]
    return tools
