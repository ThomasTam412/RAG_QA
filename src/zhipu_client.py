import os
import json
import logging
import requests
from typing import Dict, List, Optional
from src.config import Config

logger = logging.getLogger(__name__)


class ZhipuAIClient:
    """智谱AI API客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZHIPU_API_KEY
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.max_retries = 3
        self.timeout = 30

        if not self.api_key:
            raise ValueError("智谱API密钥未设置，请检查 .env 文件中的 ZHIPU_API_KEY")

    def generate_response(self,
                          messages: List[Dict[str, str]],
                          model: str = "glm-4",
                          max_tokens: int = 2000,
                          temperature: float = 0.7,
                          top_p: float = 0.7) -> str:
        """
        生成对话回复

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            model: 使用的模型名称
            max_tokens: 最大生成长度
            temperature: 温度参数
            top_p: 核采样参数

        Returns:
            str: 模型生成的回复内容
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"调用智谱API (尝试 {attempt + 1}/{self.max_retries})")
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                # 提取回复内容
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.debug("API调用成功")
                    return content.strip()
                else:
                    raise ValueError("API响应格式异常")

            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求失败 (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"API请求失败，已重试 {self.max_retries} 次: {e}")

            except (KeyError, ValueError) as e:
                logger.error(f"API响应解析失败: {e}")
                if "error" in result:
                    error_msg = result["error"].get("message", "未知错误")
                    raise Exception(f"API返回错误: {error_msg}")
                raise Exception(f"API响应解析失败: {e}")

        raise Exception("API调用失败，已达到最大重试次数")

    def chat(self,
             user_message: str,
             system_prompt: Optional[str] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        简化的聊天接口

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            conversation_history: 对话历史

        Returns:
            str: 模型回复
        """
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加对话历史
        if conversation_history:
            messages.extend(conversation_history)

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        return self.generate_response(messages)

    def rag_chat(self,
                 question: str,
                 context: str,
                 system_prompt: Optional[str] = None) -> str:
        """
        RAG专用的聊天接口，将检索到的上下文与问题结合

        Args:
            question: 用户问题
            context: 检索到的相关文档内容
            system_prompt: 自定义系统提示词

        Returns:
            str: 基于上下文的回答
        """

        # 默认的RAG系统提示词
        default_system_prompt = """你是一个智能助手，基于用户提供的参考信息来回答问题。
请遵循以下规则：
1. 仔细阅读参考信息，确保回答准确
2. 如果参考信息中包含答案，请基于参考信息回答
3. 如果参考信息中没有答案，请明确说明并基于你的知识回答
4. 回答要简洁明了，重点突出
5. 如果参考信息不相关或不足以回答问题，请诚实说明"""

        system_prompt = system_prompt or default_system_prompt

        # 构建用户消息，包含检索到的上下文
        user_message = f"""请基于以下参考信息回答问题：

参考信息：
{context}

问题：
{question}

请基于参考信息回答问题，如果参考信息不够充分，请说明。"""

        return self.chat(user_message, system_prompt)