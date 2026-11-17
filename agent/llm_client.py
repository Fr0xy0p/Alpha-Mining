"""
agent/llm_client.py — Unified LLM API client for factor analysis agent.

Supports Google Gemini Flash, Groq, DeepSeek, and any OpenAI-compatible chat completion APIs.
Credentials loaded from GEMINI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY,
OPENAI_API_KEY, or .env file.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests


# Defaults
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-3-flash-preview"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096


def _read_env_file() -> dict[str, str]:
    """Read key-value pairs from .env file at the project root."""
    _AGENT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _AGENT_DIR.parent
    dotenv_path = _PROJECT_ROOT / ".env"
    env_vars: dict[str, str] = {}
    if dotenv_path.exists():
        with open(dotenv_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
    return env_vars


def load_api_key(env_var: Optional[str] = None) -> str:
    """Load API key from .env or environment variables.

    If env_var is specified, attempts to load that specific key.
    Otherwise checks in order based on LLM_PROVIDER or candidate priority:
      1. GEMINI_API_KEY / GOOGLE_API_KEY
      2. GROQ_API_KEY
      3. DEEPSEEK_API_KEY
      4. OPENAI_API_KEY
      5. LLM_API_KEY
    """
    env_file_vars = _read_env_file()

    if env_var:
        if env_var in env_file_vars:
            return env_file_vars[env_var]
        val = os.environ.get(env_var)
        if val:
            return val
        raise RuntimeError(
            f"{env_var} not found. Set it in .env or as an environment variable."
        )

    # If provider or model is explicitly Gemini, prioritize Gemini keys
    provider = (
        os.environ.get("LLM_PROVIDER")
        or env_file_vars.get("LLM_PROVIDER")
        or ""
    ).lower()
    model = (
        os.environ.get("LLM_MODEL")
        or env_file_vars.get("LLM_MODEL")
        or ""
    ).lower()

    if provider == "gemini" or "gemini" in model:
        for candidate in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if candidate in env_file_vars and env_file_vars[candidate]:
                return env_file_vars[candidate]
            val = os.environ.get(candidate)
            if val:
                return val
    elif provider == "groq" or "groq" in model:
        if "GROQ_API_KEY" in env_file_vars and env_file_vars["GROQ_API_KEY"]:
            return env_file_vars["GROQ_API_KEY"]
        val = os.environ.get("GROQ_API_KEY")
        if val:
            return val
    elif provider == "deepseek" or "deepseek" in model:
        if "DEEPSEEK_API_KEY" in env_file_vars and env_file_vars["DEEPSEEK_API_KEY"]:
            return env_file_vars["DEEPSEEK_API_KEY"]
        val = os.environ.get("DEEPSEEK_API_KEY")
        if val:
            return val

    # Candidate order when no specific provider/model is pinned
    candidates = (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "LLM_API_KEY",
    )
    for candidate in candidates:
        if candidate in env_file_vars and env_file_vars[candidate]:
            return env_file_vars[candidate]
        val = os.environ.get(candidate)
        if val:
            return val

    raise RuntimeError(
        "No LLM API key found. Please set GEMINI_API_KEY, GROQ_API_KEY, or DEEPSEEK_API_KEY in .env or as an environment variable."
    )


def _extract_json(text: str) -> dict:
    """Robustly extract and parse the first JSON object from *text*.

    Handles plain JSON, markdown-fenced JSON blocks (`````json … `````),
    and text with leading / trailing noise.
    """
    text = text.strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end -= 1
        text = "\n".join(lines[start:end]).strip()

    # Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost { … } block.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:600]}")


class LLMClient:
    """Lightweight client for OpenAI-compatible chat APIs (Gemini Flash, Groq, DeepSeek, OpenAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        env_file_vars = _read_env_file()

        # 1. Resolve API key
        self.api_key = api_key or load_api_key()

        # 2. Detect provider
        explicit_model = (
            model
            or os.environ.get("LLM_MODEL")
            or env_file_vars.get("LLM_MODEL")
            or ""
        )
        explicit_base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or env_file_vars.get("LLM_BASE_URL")
            or ""
        )
        provider = (
            os.environ.get("LLM_PROVIDER")
            or env_file_vars.get("LLM_PROVIDER")
            or ""
        ).lower()

        if self.api_key.startswith("AIza") or self.api_key.startswith("AQ."):
            is_gemini = True
            is_groq = False
        elif self.api_key.startswith("gsk_"):
            is_gemini = False
            is_groq = True
        else:
            is_gemini = (
                provider == "gemini"
                or (model and "gemini" in model.lower())
                or (not model and "gemini" in explicit_model.lower() and not self.api_key.startswith("sk-"))
                or "generativelanguage.googleapis.com" in explicit_base_url
                or (self.api_key == env_file_vars.get("GEMINI_API_KEY"))
                or (self.api_key == env_file_vars.get("GOOGLE_API_KEY"))
                or (self.api_key == os.environ.get("GEMINI_API_KEY"))
                or (self.api_key == os.environ.get("GOOGLE_API_KEY"))
            )
            is_groq = not is_gemini and (
                provider == "groq"
                or "groq.com" in explicit_base_url
                or (model and ("llama" in model.lower() or "gpt-oss" in model.lower()))
                or (not model and ("llama" in explicit_model.lower() or "gpt-oss" in explicit_model.lower()))
                or (self.api_key == env_file_vars.get("GROQ_API_KEY"))
                or (self.api_key == os.environ.get("GROQ_API_KEY"))
            )

        # 3. Resolve base_url
        if not base_url:
            base_url = (
                os.environ.get("LLM_BASE_URL")
                or env_file_vars.get("LLM_BASE_URL")
            )
            if not base_url:
                if is_gemini:
                    base_url = (
                        os.environ.get("GEMINI_BASE_URL")
                        or env_file_vars.get("GEMINI_BASE_URL")
                        or GEMINI_BASE_URL
                    )
                elif is_groq:
                    base_url = (
                        os.environ.get("GROQ_BASE_URL")
                        or env_file_vars.get("GROQ_BASE_URL")
                        or GROQ_BASE_URL
                    )
                else:
                    base_url = (
                        os.environ.get("DEEPSEEK_BASE_URL")
                        or env_file_vars.get("DEEPSEEK_BASE_URL")
                        or DEEPSEEK_BASE_URL
                    )
        self.base_url = base_url.rstrip("/")

        # 4. Resolve model
        env_model = os.environ.get("LLM_MODEL") or env_file_vars.get("LLM_MODEL")
        if is_gemini:
            if not model or model in ("deepseek-chat", "gemini-2.5-flash"):
                model = (
                    (env_model if env_model and "gemini" in env_model.lower() else None)
                    or os.environ.get("GEMINI_MODEL")
                    or env_file_vars.get("GEMINI_MODEL")
                    or GEMINI_DEFAULT_MODEL
                )
        elif is_groq:
            if not model or model == "deepseek-chat":
                model = (
                    (env_model if env_model and any(k in env_model.lower() for k in ("llama", "gpt-oss", "groq", "qwen")) else None)
                    or os.environ.get("GROQ_MODEL")
                    or env_file_vars.get("GROQ_MODEL")
                    or GROQ_DEFAULT_MODEL
                )
        elif not model:
            if env_model and "deepseek" in env_model.lower():
                model = env_model
            else:
                model = (
                    GEMINI_DEFAULT_MODEL
                    if "googleapis.com" in self.base_url
                    else (
                        GROQ_DEFAULT_MODEL
                        if "groq.com" in self.base_url
                        else DEEPSEEK_DEFAULT_MODEL
                    )
                )
        self.model = model
        self.is_gemini = is_gemini
        self.is_groq = is_groq
        self.max_retries = max_retries
        self.timeout = timeout

    # ── Public API ────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Send a chat completion request and return the response text."""
        payload = self._build_payload(messages, temperature, max_tokens)
        data = self._request(payload)
        return data["choices"][0]["message"]["content"]

    def chat_structured(
        self,
        messages: list[dict],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        """Send a chat request and parse the response as a JSON object."""
        content = self.chat(messages, temperature=temperature)
        return _extract_json(content)

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_payload(self, messages, temperature, max_tokens) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.is_gemini:
            headers["x-goog-api-key"] = self.api_key

        return {
            "headers": headers,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _request(self, payload: dict) -> dict:
        headers = payload.pop("headers")
        body = {k: v for k, v in payload.items() if k != "headers"}

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()

            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                resp_text = exc.response.text[:300] if exc.response is not None else ""
                last_error = f"HTTP {status}: {resp_text or exc}"

                # Rate limiting (HTTP 429)
                if status == 429:
                    retry_after = None
                    if exc.response is not None and "retry-after" in exc.response.headers:
                        try:
                            retry_after = float(exc.response.headers["retry-after"])
                        except (ValueError, TypeError):
                            pass
                    if retry_after is not None:
                        time.sleep(retry_after + 0.5)
                    else:
                        self._backoff(attempt)
                    continue

                if 500 <= status < 600 and attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"Network error: {exc}"
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise

            except (KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Unexpected API response: {exc}")

        raise RuntimeError(
            f"LLM request failed after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2 ** attempt, 30))
