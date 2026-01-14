"""Base agent class for all story factory agents."""

import ollama
from typing import Optional
from config import OLLAMA_BASE_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_TOKENS, CONTEXT_SIZE


class BaseAgent:
    """Base class for all agents in the story factory."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model: str = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model or DEFAULT_MODEL  # Use default if None
        self.temperature = temperature
        self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response from the agent."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"CURRENT STORY CONTEXT:\n{context}"
            })

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
                "num_predict": MAX_TOKENS,
                "num_ctx": CONTEXT_SIZE,  # Critical: set context window size
            },
        )

        return response["message"]["content"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"
