"""
test_llm_client.py — Unit tests for LLMClient provider detection,
payload construction, and key resolution (Gemini, Groq, DeepSeek).
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from agent.llm_client import (
    LLMClient,
    load_api_key,
    GEMINI_BASE_URL,
    GEMINI_DEFAULT_MODEL,
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
)


class TestLLMClientGemini(unittest.TestCase):
    """Test Gemini Flash auto-detection and configuration."""

    def test_gemini_key_auto_detection(self):
        """Keys starting with AIza or AQ. should automatically configure Gemini endpoint & model."""
        client = LLMClient(api_key="AIzaSyMockGeminiKey1234567890")
        self.assertTrue(client.is_gemini)
        self.assertFalse(client.is_groq)
        self.assertEqual(client.base_url, GEMINI_BASE_URL)
        self.assertEqual(client.model, GEMINI_DEFAULT_MODEL)

        client_aq = LLMClient(api_key="AQ.MockGeminiKey1234567890")
        self.assertTrue(client_aq.is_gemini)
        self.assertEqual(client_aq.base_url, GEMINI_BASE_URL)
        self.assertEqual(client_aq.model, GEMINI_DEFAULT_MODEL)

    def test_gemini_custom_model(self):
        """User can specify custom Gemini models like gemini-2.0-flash."""
        client = LLMClient(api_key="AIzaSyMockKey", model="gemini-2.0-flash")
        self.assertTrue(client.is_gemini)
        self.assertEqual(client.base_url, GEMINI_BASE_URL)
        self.assertEqual(client.model, "gemini-2.0-flash")

    def test_gemini_default_overrides_deepseek_default(self):
        """CLI default model 'deepseek-chat' should be replaced by Gemini default for Gemini keys."""
        client = LLMClient(api_key="AIzaSyMockKey", model="deepseek-chat")
        self.assertTrue(client.is_gemini)
        self.assertEqual(client.model, GEMINI_DEFAULT_MODEL)

    def test_gemini_payload_headers(self):
        """Gemini payloads must include Authorization and x-goog-api-key headers."""
        client = LLMClient(api_key="AIzaSyMockKey")
        payload = client._build_payload([{"role": "user", "content": "hi"}], 0.3, 1024)
        self.assertEqual(payload["headers"]["Authorization"], "Bearer AIzaSyMockKey")
        self.assertEqual(payload["headers"]["x-goog-api-key"], "AIzaSyMockKey")
        self.assertEqual(payload["model"], GEMINI_DEFAULT_MODEL)

    @patch("agent.llm_client._read_env_file")
    def test_load_api_key_prioritizes_gemini(self, mock_env):
        mock_env.return_value = {
            "GEMINI_API_KEY": "AIzaSyFromEnv",
            "GROQ_API_KEY": "gsk_from_env",
        }
        key = load_api_key()
        self.assertEqual(key, "AIzaSyFromEnv")

    @patch("agent.llm_client._read_env_file")
    def test_load_api_key_google_api_key(self, mock_env):
        mock_env.return_value = {
            "GOOGLE_API_KEY": "AIzaSyGoogleKey",
        }
        key = load_api_key()
        self.assertEqual(key, "AIzaSyGoogleKey")


class TestLLMClientGroqAndDeepSeek(unittest.TestCase):
    """Test Groq and DeepSeek backward compatibility."""

    def test_groq_key_auto_detection(self):
        """Keys starting with gsk_ should automatically configure Groq endpoint & model."""
        client = LLMClient(api_key="gsk_MockGroqKey1234567890")
        self.assertFalse(client.is_gemini)
        self.assertTrue(client.is_groq)
        self.assertEqual(client.base_url, GROQ_BASE_URL)
        self.assertEqual(client.model, GROQ_DEFAULT_MODEL)

    def test_deepseek_auto_detection(self):
        """Arbitrary keys without Gemini or Groq markers default to DeepSeek."""
        client = LLMClient(api_key="sk-deepseek-test-key")
        self.assertFalse(client.is_gemini)
        self.assertFalse(client.is_groq)
        self.assertEqual(client.base_url, DEEPSEEK_BASE_URL)
        self.assertEqual(client.model, DEEPSEEK_DEFAULT_MODEL)

    @patch("agent.llm_client.requests.post")
    def test_chat_execution(self, mock_post):
        """Verify chat completion call format."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "alpha result"}}]
        }
        mock_post.return_value = mock_resp

        client = LLMClient(api_key="AIzaSyMockKey")
        result = client.chat([{"role": "user", "content": "test"}])
        self.assertEqual(result, "alpha result")
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        self.assertEqual(call_url, f"{GEMINI_BASE_URL}/chat/completions")


if __name__ == "__main__":
    unittest.main()
