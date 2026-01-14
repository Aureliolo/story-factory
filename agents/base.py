"""Base agent class for all story factory agents."""

import ollama
from typing import Optional
from settings import Settings, get_model_info


class BaseAgent:
    """Base class for all agents in the story factory."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        agent_role: str = None,  # For auto model selection
        model: str = None,
        temperature: float = None,
        settings: Settings = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.agent_role = agent_role or role.lower().replace(" ", "_")

        # Load settings
        self.settings = settings or Settings.load()

        # Get model and temperature from settings if not specified
        if model:
            self.model = model
        else:
            self.model = self.settings.get_model_for_agent(self.agent_role)

        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = self.settings.get_temperature_for_agent(self.agent_role)

        self.client = ollama.Client(host=self.settings.ollama_url)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate a response from the agent."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"CURRENT STORY CONTEXT:\n{context}"
            })

        messages.append({"role": "user", "content": prompt})

        use_model = model or self.model

        response = self.client.chat(
            model=use_model,
            messages=messages,
            options={
                "temperature": temperature or self.temperature,
                "num_predict": self.settings.max_tokens,
                "num_ctx": self.settings.context_size,
            },
        )

        return response["message"]["content"]

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return get_model_info(self.model)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
