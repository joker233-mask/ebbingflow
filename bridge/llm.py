"""
EbbingFlow 统一大模型驱动桥接层 (LLM Bridge)
支持 OpenAI, SiliconFlow, Volcengine, DeepSeek 等各种兼容 OpenAI API 格式的服务。
"""
import os
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any, Union
from openai import AsyncOpenAI
from config import LLMConfig

logger = logging.getLogger(__name__)

class LLMBridge:
    """提供统一的 LLM 访问接口，内置 Token 自动监测与异常屏蔽"""
    
    def __init__(self, config: LLMConfig, category: str = "chat"):
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,localhost:11434"
        self.config = config
        print(f"[LLM Bridge] Initialized for {category} | Model: {config.model} | Base: {config.base_url}")
        self.category = category
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )

    def _apply_provider_options(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        options = dict(kwargs)
        extra_body = dict(options.pop("extra_body", {}) or {})

        # Explicit request-level thinking options take precedence over the
        # provider default so OpenAI-compatible callers can enable reasoning.
        thinking = extra_body.get("thinking")
        if not isinstance(thinking, dict):
            thinking = dict(thinking or {}) if thinking else {}
        provided_thinking = options.pop("thinking", None)
        if isinstance(provided_thinking, dict):
            thinking.update(provided_thinking)
        elif provided_thinking is not None:
            thinking["type"] = str(provided_thinking)

        if "type" not in thinking:
            thinking_mode = getattr(self.config, "thinking_mode", "")
            if thinking_mode in {"enabled", "disabled"}:
                thinking["type"] = thinking_mode

        if thinking:
            extra_body["thinking"] = thinking

        reasoning_effort = options.pop("reasoning_effort", None)
        if reasoning_effort is None:
            reasoning_effort = extra_body.get("reasoning_effort")
        if reasoning_effort is not None:
            extra_body["reasoning_effort"] = reasoning_effort

        if extra_body:
            options["extra_body"] = extra_body
        return options

    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[str]:
        try:
            temp = temperature if temperature is not None else self.config.temperature
            kwargs = self._apply_provider_options(kwargs)
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                response_format=response_format,
                **kwargs
            )
            if hasattr(response, "usage") and response.usage:
                from core.monitoring import token_monitor
                token_monitor.record_llm_usage(
                    category=self.category,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[LLM Bridge] ChatCompletion Error: {e}")
            return None

    async def chat_stream(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        try:
            temp = temperature if temperature is not None else self.config.temperature
            kwargs = self._apply_provider_options(kwargs)
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            async for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    from core.monitoring import token_monitor
                    token_monitor.record_llm_usage(
                        category=self.category,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens
                    )
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            logger.error(f"[LLM Bridge] Stream Error: {e}")
            yield f"\n[AI 响应异常: {e}]"

    async def chat_stream_events(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream typed events so callers can forward reasoning_content separately."""
        try:
            temp = temperature if temperature is not None else self.config.temperature
            kwargs = self._apply_provider_options(kwargs)
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs
            )
            async for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    from core.monitoring import token_monitor
                    token_monitor.record_llm_usage(
                        category=self.category,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens
                    )
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    content = getattr(delta, "content", None)
                    if reasoning:
                        yield {"type": "reasoning", "text": reasoning}
                    if content:
                        yield {"type": "content", "text": content}
        except Exception as e:
            logger.error(f"[LLM Bridge] Stream Error: {e}")
            yield {"type": "content", "text": f"\n[AI 响应异常: {e}]"}
