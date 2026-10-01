from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .ollama import ChatResponse, OllamaError, categorize_error


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any], timeout: float = 600):
        self.config = config
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        output_tokens: int,
    ) -> ChatResponse:
        env_name = str(self.config.get("api_key_env") or "")
        api_key = os.environ.get(env_name) if env_name else None
        if not api_key:
            raise OllamaError(f"Falta la variable de entorno {env_name or 'api_key_env'} para el modelo remoto.", "credentials")
        base_url = str(self.config.get("base_url") or "").rstrip("/")
        if not base_url.startswith("https://"):
            raise OllamaError("El endpoint remoto debe utilizar HTTPS.", "privacy")
        endpoint = base_url + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.get("model"),
            "messages": self._convert_messages(messages),
            "temperature": options.get("temperature", 0.3),
            "top_p": options.get("top_p", 0.9),
            "max_tokens": output_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if options.get("reasoning_effort"):
            body["reasoning_effort"] = options["reasoning_effort"]
        started = time.perf_counter()
        try:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            usage = data.get("usage") or {}
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            detail = exc.response.text[:1000] if isinstance(exc, httpx.HTTPStatusError) else str(exc)
            raise OllamaError(f"Error del modelo remoto: {detail}", categorize_error(detail)) from exc
        return ChatResponse(
            message=message,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_seconds=max(0.001, time.perf_counter() - started),
            generation_seconds=max(0.001, time.perf_counter() - started),
        )

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            item = dict(message)
            if "tool_name" in item:
                item["name"] = item.pop("tool_name")
            images = item.pop("images", None)
            if images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": str(item.get("content") or "")}]
                parts.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images)
                item["content"] = parts
            converted.append(item)
        return converted
