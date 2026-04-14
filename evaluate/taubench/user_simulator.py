from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from prog_env.utils.chatvllm import ChatVLLM
from prog_env.utils.llm_server import AsyncLLMServer


USER_SYSTEM_PROMPT = """You are a user interacting with an agent.

Instruction: {instruction}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


class AsyncTextResponder:
    def __init__(
        self,
        backend: str,
        model_name: str,
        chat_engine: Optional[ChatVLLM] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.chat_engine = chat_engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_cost = 0.0
        self.client = None
        if backend == "openai":
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, messages: list[dict]) -> str:
        if self.backend == "vllm":
            assert self.chat_engine is not None
            assert isinstance(self.chat_engine.engine, AsyncLLMServer)
            return await self.chat_engine.engine.chat_one_async(
                messages,
                None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        if self.backend == "openai":
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content or ""
        raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def get_total_cost(self) -> float:
        return self.total_cost


class TauBenchUserSimulator:
    def __init__(self, responder: AsyncTextResponder) -> None:
        self.responder = responder
        self.messages: list[dict] = []

    async def reset(self, instruction: str) -> str:
        self.messages = [
            {"role": "system", "content": USER_SYSTEM_PROMPT.format(instruction=instruction)},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        response = await self.responder.generate(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        response = await self.responder.generate(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def get_total_cost(self) -> float:
        return self.responder.get_total_cost()
