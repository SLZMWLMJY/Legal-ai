import asyncio, logging
from datetime import datetime
from typing import List, Optional, Callable, AsyncIterator

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

from core.llm import get_default_llm
from models.json_response import JsonData
from services import legal_service
from services.legal_service import LegalService
from tools.legal_tool import get_tools


# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
now = datetime.now()
timestamp = datetime.now().isoformat()


def create_legal_agent(tools: List[Tool]):
    """创建法律智能体"""

    system_prompt = """ 你是一个专业的法律智能助手，名为“LegalAI。
    专业领域：合同法，劳动法，知识产权法为主
    
    ## 1.服务对象：为企业和公民提供基础的法律咨询
   
    ## 2.相关法条引用：
        《xxx》第x条：...
        《xxx》第x条：...

    ## 3. 责任界限
    ### 必须明确区分：
    - ✅ **法律事实**：法律法规的明确规定
    - 💡 **法律分析**：基于法条的推理和解释
    - ⚠️ **潜在风险**：可能的法律后果和不确定性
    - 📝 **一般建议**：程序性指引和常见做法
    
    ### 严格禁止：
    - ❌ 代替执业律师提供法律意见
    - ❌ 预测法院判决结果
    - ❌ 提供超出知识范围的专业意见
    - ❌ 鼓励或暗示采取任何违法行动
    
    ## 4. 回答框架
    请按以下结构组织回答：
    
    ### 一、核心法律依据
    [引用2条及以上相关法条]
    
    ### 二、法律要点分析
    1. 权利界定：[明确相关权利义务]
    2. 法律要件：[分析构成要件或适用条件]
    3. 程序要求：[如需，说明法律程序]
    
    ### 三、风险提示
    - 主要风险：[列举主要法律风险]
    - 证据建议：[提示关键证据材料]
    - 时效注意：[如有，说明诉讼时效等]
    
    ### 四、参考案例（如有）
    案例名称：[相关典型案例]
    裁判要点：[核心裁判观点]
    *注：案例仅供参考，不构成判例约束*
    
    ### 五、行动建议
    1. 建议步骤：[一般性操作建议]
    2. 专业求助：[提示需要律师介入的情形]
    3. 机构指引：[相关行政机关或仲裁机构]
    
    ## 5. 限制声明
    **重要免责提示**：
    1. 我是AI助手，回答基于公开法律信息
    2. 具体情况需结合证据和事实综合判断
    3. 复杂法律问题必须咨询执业律师
    4. 法律法规可能变更，请以最新官方发布为准
    
    ## 6. 特殊情况处理
    当遇到以下情况时，请明确回答：
    - "根据我的知识范围，我无法提供具体建议"
    - "此问题涉及专业领域，建议咨询专业律师"
    - "相关法律尚不明确，存在解释空间"
    - "该情况可能涉及刑事责任，请立即寻求法律帮助"
    
    ## 7. 对话引导
    在回答结束时，可适当询问：
    - "请问是否需要对某一点进一步说明？"
    - "如果需要，我可以提供相关法律文书的一般格式"
    - "请提供更多事实细节以便更准确分析"
    
    
    ## 特别说明：图像处理
    当用户上传图像时，系统会提供图像文件路径（格式如 uploads/f322258afa8b415980ae15a98927563d.jpg）。
    你需要使用 image_analysis 工具来分析图像内容。
    
    ### 正确使用方式：
    1. 当看到类似 [图像文件: uploads/xxx.jpg] 或 (用户上传了图像：uploads/xxx.jpg) 的提示时
    2. 调用 image_analysis 工具，参数格式为：{{"image_url": "文件路径", "analysis_type": "general"}}
    3. 注意：使用双花括号 {{}} 来表示字典，而不是单花括号
    
    **最终提示**：我的所有回答仅供参考，不构成正式法律意见。任何重要法律决策前，请务必咨询执业律师。
    """

    # 消息模板组装
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "以下是之前的对话摘要：{summary}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 获取大模型实例
    llm = get_default_llm()

    # 创建智能体
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 创建代理
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor


async def chat_with_legal_agent(agent_executor: AgentExecutor,
                                legal_service: LegalService,
                                account_id: str,
                                input_text: str,
                                stream_callback: Optional[Callable] = None
                                ) -> AsyncIterator[str]:
    """增强版智能体对话函数"""

    try:
        # 1. 获取历史上下文（支持多种策略）
        context = await legal_service.get_conversation_context(
            account_id,
            strategy="hybrid",  # 混合策略：最近几条完整+更早摘要
            max_tokens=1000
        )

        # 2. 准备智能体输入
        agent_input = {
            "input": input_text,
            "chat_history": context.get("history"),
            "summary": context.get("summary"),
            "user_profile": await legal_service.get_user_profile(account_id)
        }

        # 3. 执行智能体流式响应
        full_response = []
        tool_calls_log = []

        async for chunk in agent_executor.astream(agent_input):
            # 处理输出内容
            if "output" in chunk:
                token = chunk["output"]
                full_response.append(token)

                # 流式返回
                yield token

                # 回调通知（用于前端进度显示等）
                if stream_callback:
                    await stream_callback("token", token)

                await asyncio.sleep(0.01)

            # 记录中间步骤（如果配置了return_intermediate_steps=True）
            elif "intermediate_steps" in chunk:
                tool_calls_log.append(chunk["intermediate_steps"])

                if stream_callback:
                    await stream_callback("thinking", chunk["intermediate_steps"])

            # 处理错误或特殊状态
            elif "error" in chunk:
                logger.warning(f"Agent execution warning: {chunk['error']}")

        # 4. 后处理：保存完整的对话记录
        if full_response:
            complete_response = "".join(full_response)

            # 异步保存（不阻塞响应）
            asyncio.create_task(
                legal_service.save_conversation(
                    account_id=account_id,
                    user_input=input_text,
                    agent_response=complete_response,
                    metadata={
                        "tool_calls": tool_calls_log,
                        "context_used": context,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )

            # 异步更新对话摘要
            asyncio.create_task(
                legal_service.update_conversation_summary(account_id)
            )

    except asyncio.TimeoutError:
        logger.error(f"用户{account_id}对话超时")
        yield "抱歉，思考时间过长，请简化您的问题重试。"

    except Exception as e:
        logger.error(f"用户{account_id}对话失败：{e}", exc_info=True)

        # 根据错误类型返回不同提示
        error_msg = "对话失败，请稍后再试"
        if "rate limit" in str(e).lower():
            error_msg = "服务繁忙，请稍后重试"
        elif "context length" in str(e).lower():
            error_msg = "对话历史过长，已开启新会话"
            asyncio.create_task(legal_service.clear_chat_history(account_id))

        yield error_msg


async def generate_stream_response(legal_service: LegalService,
                                   account_id: str,
                                   input_text: str) -> AsyncIterator:
    """生成流式响应"""
    agent = create_legal_agent(get_tools())
    current_chunk = ""
    async for token in chat_with_legal_agent(agent, legal_service, account_id, input_text):
        current_chunk += token
        # 当遇到标点符号或者长度达到一定时，就发送chunk一次
        if token in ["。", "？", "！", "；", "，"] or len(current_chunk) >= 10:
            response = JsonData.stream_data(current_chunk)
            yield f"data: {response.model_dump_json()}\n\n"
            current_chunk = ""
            await asyncio.sleep(0.01)

    # 发送剩余的chunk
    if current_chunk:
        response = JsonData.stream_data(current_chunk)
        yield f"data: {response.model_dump_json()}\n\n"

    # 发送结束标记
    yield "data: [DONE]\n\n"
