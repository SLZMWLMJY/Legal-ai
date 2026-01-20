from typing import List, Dict, Optional, Any
import redis
from datetime import datetime
from core.config import settings
from core.llm import get_default_llm
import logging
import json, hashlib

logger = logging.getLogger(__name__)


class LegalService:

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True
        )
        self.llm = get_default_llm()
        # 上下文管理配置
        self.context_config = {
            "max_history_messages": 10,  # 最大历史消息数
            "summary_update_threshold": 5,  # 更新摘要的对话轮数阈值
            "context_token_limit": 2000,  # 上下文token限制
            "summary_ttl": 86400,  # 摘要缓存时间（24小时）
        }

    def _get_chat_key(self, account_id: str) -> str:
        """获取用户对话历史key"""
        return f"chat_history:{account_id}"

    def _get_summary_key(self, account_id: str) -> str:
        """获取摘要key"""
        return f"chat_summary:{account_id}"

    def _get_profile_key(self, account_id: str) -> str:
        """获取用户画像key"""
        return f"user_profile:{account_id}"

    def _get_metadata_key(self, account_id: str, session_id: Optional[str] = None) -> str:
        """获取对话元数据key"""
        if not session_id:
            # 生成基于时间的会话ID
            session_id = datetime.now().strftime("%Y%m%d")
        return f"conv_metadata:{account_id}:{session_id}"

    async def _update_user_profile_stats(self, account_id: str):
        """更新用户画像统计信息"""
        try:
            # 获取当前用户画像
            profile = await self.get_user_profile(account_id)

            # 更新统计信息
            profile["total_conversations"] = profile.get("total_conversations", 0) + 1
            profile["last_active"] = datetime.now().isoformat()

            # 保存更新后的画像
            profile_key = self._get_profile_key(account_id)
            self.redis_client.setex(profile_key, 86400 * 7, json.dumps(profile))

            logger.info(f"更新用户画像统计: account={account_id}, 对话次数={profile['total_conversations']}")

        except Exception as e:
            logger.error(f"更新用户画像统计失败: {e}")

    def save_chat_history(self, account_id: str, messages: List[Dict]):
        """保存用户对话历史"""
        key = self._get_chat_key(account_id)
        # 只保留最新的N条消息
        if len(messages) > self.context_config["max_history_messages"]:
            messages = messages[-self.context_config["max_history_messages"]:]
        self.redis_client.set(key, json.dumps(messages))

    def get_chat_history(self, account_id: str) -> List[Dict]:
        """获取用户对话历史"""
        key = self._get_chat_key(account_id)
        messages = self.redis_client.get(key)
        if messages:
            return json.loads(messages)
        return []

    def add_message(self, account_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """添加一条消息到聊天历史"""
        messages = self.get_chat_history(account_id)
        message_data = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            message_data["metadata"] = metadata
        messages.append(message_data)
        self.save_chat_history(account_id, messages)

    def save_chat_messages(self, account_id: str, user_message: str, assistant_message: str):
        """保存用户对话消息"""
        self.add_message(account_id, "user", user_message)
        self.add_message(account_id, "assistant", assistant_message)

    def clear_chat_history(self, account_id: str):
        """清空用户对话历史"""
        key = self._get_chat_key(account_id)
        self.redis_client.delete(key)
        # 同时清除相关数据
        self.redis_client.delete(self._get_summary_key(account_id))

    async def generate_summary(self, account_id: str) -> str:
        """生成摘要（优化版）"""
        try:
            # 检查缓存摘要是否有效
            summary_key = self._get_summary_key(account_id)
            cached_summary = self.redis_client.get(summary_key)

            # 如果缓存摘要存在且未过期，直接返回
            if cached_summary and self.redis_client.ttl(summary_key) > 0:
                return cached_summary

            # 获取最新聊天记录
            messages = self.get_chat_history(account_id)
            if not messages:
                return ""

            # 计算消息哈希，判断是否需要重新生成
            messages_hash = hashlib.md5(json.dumps(messages).encode()).hexdigest()
            last_hash_key = f"summary_hash:{account_id}"
            last_hash = self.redis_client.get(last_hash_key)

            if last_hash == messages_hash:
                # 消息未变化，返回缓存
                return cached_summary or ""

            # 构建提示词
            prompt = f"""请根据以下对话历史生成一个简洁的核心摘要，突出主要话题和关键信息：
            
            {json.dumps(messages, ensure_ascii=False, indent=2)}
            
            摘要要求：
            1. 突出对话的主要话题和关键信息
            2. 使用第三人称描述，提取重要数据/时间节点/待办事项
            3. 保留原始对话中的重要细节
            4. 确保包含最新的对话内容
            5. 长度不超过200字
            """

            # 生成摘要
            response = await self.llm.ainvoke(prompt)
            new_summary = response.content

            # 更新缓存和哈希
            self.redis_client.setex(
                summary_key,
                self.context_config["summary_ttl"],
                new_summary
            )
            self.redis_client.setex(last_hash_key, self.context_config["summary_ttl"], messages_hash)

            return new_summary

        except Exception as e:
            logger.error(f"生成摘要失败:{e}")
            return ""

    # ============ 新增方法 ============

    async def get_conversation_context(
            self,
            account_id: str,
            strategy: str = "hybrid",
            max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        获取完整的对话上下文
        
        Args:
            account_id: 用户ID
            strategy: 上下文策略 (full/summary/hybrid/window)
            max_tokens: token限制
        
        Returns:
            Dict包含历史、摘要和元数据
        """
        context = {
            "history": [],
            "summary": "",
            "strategy": strategy,
            "tokens_used": 0
        }

        # 获取基础数据
        messages = self.get_chat_history(account_id)
        summary = await self.generate_summary(account_id)

        # 根据策略构建上下文
        if strategy == "full":
            # 完整历史（受token限制）
            context["history"] = messages
            # 这里可以添加token计数逻辑

        elif strategy == "summary":
            # 仅摘要
            context["summary"] = summary

        elif strategy == "hybrid":
            # 混合模式：最近几条完整 + 更早摘要
            recent_count = 3
            if len(messages) > recent_count:
                context["history"] = messages[-recent_count:]
                context["summary"] = summary
            else:
                context["history"] = messages

        elif strategy == "window":
            # 滑动窗口：固定条数
            window_size = 5
            context["history"] = messages[-window_size:] if len(messages) > window_size else messages

        # 添加用户画像信息
        user_profile = await self.get_user_profile(account_id)
        if user_profile:
            context["user_profile"] = user_profile

        return context

    async def get_user_profile(self, account_id: str) -> Dict[str, Any]:
        """
        获取用户画像/档案
        
        Args:
            account_id: 用户ID
            
        Returns:
            用户画像信息
        """
        profile_key = self._get_profile_key(account_id)
        profile_data = self.redis_client.get(profile_key)

        if profile_data:
            return json.loads(profile_data)

        # 如果没有存储的画像，从对话历史中推断
        default_profile = {
            "preferred_language": "zh-CN",
            "legal_interests": [],  # 法律兴趣领域
            "conversation_style": "formal",  # 对话风格
            "last_active": datetime.now().isoformat(),
            "total_conversations": 0
        }

        # 从历史对话中提取画像信息
        messages = self.get_chat_history(account_id)
        if messages:
            # 分析对话内容，提取用户特征
            profile_prompt = f"""根据以下对话历史，分析用户特征：
            
            对话历史：
            {json.dumps(messages, ensure_ascii=False, indent=2)}
            
            请分析：
            1. 用户可能的法律领域兴趣
            2. 用户的对话风格偏好（正式/随意）
            3. 用户的身份特征（个人/企业/学生等）
            4. 用户关心的主要法律问题类型
            
            返回JSON格式：
            {{
                "legal_interests": [],
                "conversation_style": "",
                "user_type": "",
                "concerned_areas": []
            }}
            """

            try:
                response = await self.llm.ainvoke(profile_prompt)
                extracted_profile = json.loads(response.content)
                default_profile.update(extracted_profile)
            except Exception as e:
                logger.warning(f"用户画像分析失败: {e}")

        # 保存并返回
        self.redis_client.setex(profile_key, 86400 * 7, json.dumps(default_profile))  # 缓存7天
        return default_profile

    async def save_conversation(
            self,
            account_id: str,
            user_input: str,
            agent_response: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        保存完整对话（带元数据）
        
        Args:
            account_id: 用户ID
            user_input: 用户输入
            agent_response: 智能体响应
            metadata: 附加元数据
        """
        try:
            # 保存消息内容
            self.save_chat_messages(account_id, user_input, agent_response)

            # 保存元数据
            if metadata:
                session_id = datetime.now().strftime("%Y%m%d")
                metadata_key = self._get_metadata_key(account_id, session_id)

                # 获取现有元数据
                existing_metadata = self.redis_client.get(metadata_key)
                if existing_metadata:
                    metadata_list = json.loads(existing_metadata)
                else:
                    metadata_list = []

                # 添加新的元数据记录
                metadata_record = {
                    "timestamp": datetime.now().isoformat(),
                    "user_input": user_input[:100],  # 只保存前100字符
                    "agent_response_length": len(agent_response),
                    "metadata": metadata
                }
                metadata_list.append(metadata_record)

                # 保存元数据（保留最近100条）
                if len(metadata_list) > 100:
                    metadata_list = metadata_list[-100:]

                self.redis_client.setex(metadata_key, 86400 * 30, json.dumps(metadata_list))  # 30天

                logger.info(f"保存对话元数据: account={account_id}, session={session_id}")

            # 更新用户画像的使用统计
            await self._update_user_profile_stats(account_id)

        except Exception as e:
            logger.error(f"保存对话失败: {e}")

    async def update_conversation_summary(self, account_id: str, force: bool = False):
        """
        更新对话摘要
        
        Args:
            account_id: 用户ID
            force: 是否强制更新
        """
        try:
            # 获取对话计数
            count_key = f"conv_count:{account_id}"
            conversation_count = self.redis_client.incr(count_key)

            # 检查是否需要更新摘要（每N次对话或强制更新）
            if force or conversation_count % self.context_config["summary_update_threshold"] == 0:
                summary = await self.generate_summary(account_id)
                logger.info(f"更新对话摘要: account={account_id}, count={conversation_count}")

                # 重置计数
                if conversation_count >= 100:
                    self.redis_client.set(count_key, 0)

                return summary

            # 返回缓存的摘要
            summary_key = self._get_summary_key(account_id)
            return self.redis_client.get(summary_key) or ""

        except Exception as e:
            logger.error(f"更新对话摘要失败: {e}")
            return ""
