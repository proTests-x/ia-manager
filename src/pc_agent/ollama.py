from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class OllamaError(RuntimeError):
    def __init__(self, message: str, category: str = "backend"):
        super().__init__(message)
        self.category = category


@dataclass(slots=True)
class ChatResponse:
    message: dict[str, Any]
    prompt_tokens: int
    output_tokens: int
    elapsed_seconds: float
    generation_seconds: float
    total_duration_ns: int = 0
    load_duration_ns: int = 0


def categorize_error(text: str) -> str:
    lower = text.lower()
    if any(value in lower for value in ("out of memory", "cuda out", "oom", "memory allocation")):
        return "oom"
    if any(value in lower for value in ("context length", "context window", "too many tokens", "num_ctx")):
        return "context"
    if "does not support tools" in lower or "tool" in lower and "not support" in lower:
        return "tools"
    if any(value in lower for value in ("does not support images", "vision", "modality")):
        return "modality"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    return "backend"


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def health(self) -> dict[str, Any]:
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(
                f"No se pudo conectar a Ollama en {self.base_url}. Comprueba que Ollama esté abierto.",
                "connection",
            ) from exc

    def show(self, model: str) -> dict[str, Any]:
        try:
            response = self.client.post(f"{self.base_url}/api/show", json={"model": model, "verbose": True}, timeout=30)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            detail = exc.response.text[:500] if isinstance(exc, httpx.HTTPStatusError) else str(exc)
            raise OllamaError(f"No se pudo inspeccionar {model}: {detail}", categorize_error(detail)) from exc

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        context_size: int,
        output_tokens: int,
    ) -> ChatResponse:
        clean_options = {
            "temperature": options.get("temperature", 0.3),
            "top_p": options.get("top_p", 0.9),
            "top_k": options.get("top_k", 40),
            "num_ctx": context_size,
            "num_predict": output_tokens,
        }
        for key in ("num_gpu", "num_thread", "num_batch"):
            if options.get(key) is not None:
                clean_options[key] = options[key]
        clean_options.update(options.get("backend_options") or {})
        effort = options.get("reasoning_effort")
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": bool(effort and str(effort).lower() not in {"none", "off", "false", "0"}),
            "options": clean_options,
        }
        if tools:
            payload["tools"] = tools
        started = time.perf_counter()
        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama tardó demasiado en responder.", "timeout") from exc
        except (httpx.HTTPError, ValueError) as exc:
            detail = ""
            if isinstance(exc, httpx.HTTPStatusError):
                detail = exc.response.text[:1000]
            else:
                detail = str(exc)
            raise OllamaError(f"Error consultando Ollama. {detail}", categorize_error(detail)) from exc
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama devolvió una respuesta sin 'message'.", "invalid_response")
        return ChatResponse(
            message=message,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            elapsed_seconds=max(0.001, time.perf_counter() - started),
            generation_seconds=max(0.001, int(data.get("eval_duration") or 0) / 1_000_000_000),
            total_duration_ns=int(data.get("total_duration") or 0),
            load_duration_ns=int(data.get("load_duration") or 0),
        )

    def unload(self, model: str) -> None:
        try:
            self.client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
                timeout=30,
            )
        except httpx.HTTPError:
            pass

    def running_models(self) -> dict[str, Any]:
        try:
            response = self.client.get(f"{self.base_url}/api/ps", timeout=5)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            return {"models": []}

    def close(self) -> None:
        self.client.close()
