import asyncio
from typing import List, Any, Dict, Optional, TypeVar, Generic, Union, Callable, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output
from openai import OpenAI, AsyncOpenAI

from core.config import settings
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class OpenAISDKWrapper(Runnable):
    """OpenAI SDK 包装器，实现 LangChain Runnable 接口"""

    model_name: str = "qwen-vl-plus"

    def __init__(self):
        super().__init__()
        # 同步客户端
        self.client = OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL
        )
        # 异步客户端
        self.async_client = AsyncOpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL
        )
        self.model = settings.QWEN_MODEL_NAME
        self.temperature = getattr(settings, 'QWEN_TEMPERATURE', 0.1)

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "qwen_openai"

    def _call(
            self,
            prompt: str,
            stop: List[str] = None,
            run_manager: CallbackManagerForLLMRun = None,
            **kwargs: Any,
    ) -> str:
        """同步调用方法"""
        try:
            # 构建消息
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Qwen LLM 调用失败: {e}")
            raise

    async def _acall(
            self,
            prompt: str,
            stop: List[str] = None,
            run_manager: AsyncCallbackManagerForLLMRun = None,
            **kwargs: Any,
    ) -> str:
        """异步调用方法"""
        try:
            # 构建消息
            messages = [{"role": "user", "content": prompt}]

            params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 2000,
                **kwargs
            }

            if stop:
                params["stop"] = stop

            # 使用异步客户端
            response = await self.async_client.chat.completions.create(**params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Qwen LLM 异步调用失败: {e}")
            raise

    def invoke(self, input: Union[List[Any], Dict], config=None, **kwargs) -> str:
        """调用 Qwen 模型，实现 LangChain Runnable 接口的 invoke 方法，返回字符串内容"""
        try:
            # 处理不同类型的输入
            if isinstance(input, dict):
                # 处理字典类型输入（LangChain通常传入包含'messages'键的字典）
                messages = input.get('messages', input)
            else:
                messages = input

            # 转换消息格式为 OpenAI 格式
            openai_messages = self._convert_messages(messages)

            logger.info(f"调用 Qwen 模型: {self.model}")

            # 准备调用参数
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": 2000
            }

            # 如果绑定了函数，添加到参数中
            if hasattr(self, 'functions') and self.functions:
                params["functions"] = self.functions

            response = self.client.chat.completions.create(**params)

            # 直接返回字符串内容，而不是包装为ChatGeneration对象
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Qwen LLM 调用失败: {e}")
            raise

    async def ainvoke(self, input: Union[List[Any], Dict], config=None, **kwargs) -> str:
        """异步调用，实现 LangChain Runnable 接口，返回字符串内容"""
        try:
            # 处理不同类型的输入
            if isinstance(input, dict):
                messages = input.get('messages', input)
            else:
                messages = input

            # 转换消息格式为 OpenAI 格式
            openai_messages = self._convert_messages(messages)

            logger.info(f"异步调用 Qwen 模型: {self.model}")

            # 准备调用参数
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": 2000
            }

            # 如果绑定了函数，添加到参数中
            if hasattr(self, 'functions') and self.functions:
                params["functions"] = self.functions

            # 使用异步客户端
            response = await self.async_client.chat.completions.create(**params)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Qwen LLM 异步调用失败: {e}")
            raise

    def batch(self, inputs: Sequence[Union[List[Any], Dict]], config=None, **kwargs):
        """批量调用，实现 LangChain Runnable 接口"""
        return [self.invoke(input, config, **kwargs) for input in inputs]

    async def abatch(self, inputs: Sequence[Union[List[Any], Dict]], config=None, **kwargs):
        """异步批量调用，实现 LangChain Runnable 接口"""
        try:
            tasks = []
            for input in inputs:
                # 为每个输入创建异步任务
                task = self.ainvoke(input, config, **kwargs)
                tasks.append(task)

            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"批量调用中的任务失败: {result}")
                    processed_results.append(f"错误: {str(result)}")
                else:
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"异步批量调用失败: {e}")
            raise

    def stream(self, input: Union[List[Any], Dict], config=None, **kwargs):
        """流式调用，实现 LangChain Runnable 接口，返回字符串内容"""
        # 简化实现，返回字符串内容
        result = self.invoke(input, config, **kwargs)
        yield result

    async def astream(self, input: Union[List[Any], Dict], config=None, **kwargs):
        """真正的异步流式调用，实现 LangChain Runnable 接口"""
        try:
            # 处理输入
            if isinstance(input, dict):
                messages = input.get('messages', input)
            else:
                messages = input

            # 转换消息格式
            openai_messages = self._convert_messages(messages)

            logger.info(f"异步流式调用 Qwen 模型: {self.model}")

            # 准备调用参数，启用流式
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "max_tokens": 2000,
                "stream": True,  # 启用流式
                **kwargs
            }

            # 如果绑定了函数，添加到参数中
            if hasattr(self, 'functions') and self.functions:
                params["functions"] = self.functions

            # 使用异步客户端进行流式调用
            response_stream = await self.async_client.chat.completions.create(**params)

            # 异步迭代流式响应
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"异步流式调用失败: {e}")
            raise

    def bind(self, functions=None, **kwargs):
        """为LLM绑定函数，兼容LangChain接口"""
        # 创建新实例以避免修改原实例
        new_instance = OpenAISDKWrapper()
        new_instance.client = self.client
        new_instance.model = self.model
        new_instance.temperature = self.temperature
        new_instance.functions = functions
        return new_instance

    def _convert_messages(self, messages: List[Any]) -> List[Dict]:
        """转换消息格式为 OpenAI 格式"""
        openai_messages = []

        for msg in messages:
            if hasattr(msg, 'content'):
                # LangChain 消息对象
                content = msg.content

                # 判断角色
                role = "user"
                if hasattr(msg, '__class__'):
                    class_name = msg.__class__.__name__
                    if 'AI' in class_name or 'Assistant' in class_name:
                        role = "assistant"
                    elif 'System' in class_name:
                        role = "system"

                # 处理内容
                if isinstance(content, list):
                    # 多模态消息
                    formatted_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if 'type' in item:
                                formatted_content.append(item)
                            else:
                                if 'image_url' in item:
                                    formatted_content.append({
                                        "type": "image_url",
                                        "image_url": item['image_url']
                                    })
                                elif 'text' in item:
                                    formatted_content.append({
                                        "type": "text",
                                        "text": item['text']
                                    })
                                else:
                                    formatted_content.append({
                                        "type": "text",
                                        "text": str(item)
                                    })
                        else:
                            formatted_content.append({
                                "type": "text",
                                "text": str(item)
                            })

                    openai_messages.append({
                        "role": role,
                        "content": formatted_content
                    })
                else:
                    # 纯文本消息
                    openai_messages.append({
                        "role": role,
                        "content": str(content)
                    })

            elif isinstance(msg, tuple) and len(msg) == 2:
                # (role, content) 格式
                role, content = msg
                role_map = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system"
                }
                openai_messages.append({
                    "role": role_map.get(role, "user"),
                    "content": str(content)
                })
            elif isinstance(msg, dict):
                # 已经是 OpenAI 格式
                if 'content' in msg and isinstance(msg['content'], list):
                    formatted_content = []
                    for item in msg['content']:
                        if isinstance(item, dict) and 'type' in item:
                            formatted_content.append(item)
                        else:
                            formatted_content.append({
                                "type": "text",
                                "text": str(item)
                            })
                    msg['content'] = formatted_content
                openai_messages.append(msg)
            else:
                # 默认作为用户消息
                openai_messages.append({
                    "role": "user",
                    "content": str(msg)
                })

        return openai_messages
