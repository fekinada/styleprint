import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from styleprint.llm import LLM, LLMConfig, LLMError


@pytest.fixture
def server():
    """Fake OpenAI-style API: records requests, rejects a custom temperature like reasoning models do."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, payload):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_GET(self):
            seen.append(("GET", self.path, dict(self.headers), None))
            self._send(200, {"data": [{"id": "local-model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(("POST", self.path, dict(self.headers), body))
            if self.server.reject_temperature and "temperature" in body:
                return self._send(400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.8",
                                                  "type": "invalid_request_error", "param": "temperature"}})
            content = '{"items": []}' if "response_format" in body else "Bonjour."
            self._send(200, {"choices": [{"message": {"content": content}}]})

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    httpd.reject_temperature = False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, f"http://127.0.0.1:{httpd.server_port}", seen
    httpd.shutdown()


def test_openai_mode(server, monkeypatch):
    httpd, url, seen = server
    httpd.reject_temperature = True
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = LLM(LLMConfig(url=url + "/v1", kind="openai", model="gpt-x"))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "Bonjour."
    first, retry = seen[0][3], seen[1][3]
    assert seen[0][1] == "/v1/chat/completions"  # trailing /v1 in the url is tolerated
    assert seen[0][2]["Authorization"] == "Bearer sk-test"
    assert "chat_template_kwargs" not in first and "max_tokens" not in first
    assert first["max_completion_tokens"] == 4096
    assert "temperature" not in retry  # rejected once, dropped for good
    assert llm.json("p", {"type": "object"}) == {"items": []}
    assert "temperature" not in seen[-1][3]


def test_openai_needs_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STYLEPRINT_LLM_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        LLM(LLMConfig(url="https://api.openai.com"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(LLMError, match="set `model`"):
        _ = LLM(LLMConfig(url="https://api.openai.com")).model


def test_local_mode_never_sends_openai_key(server, monkeypatch):
    _, url, seen = server
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.delenv("STYLEPRINT_LLM_API_KEY", raising=False)
    llm = LLM(LLMConfig(url=url, thinking=True))
    assert llm.model == "local-model"
    llm.chat([{"role": "user", "content": "hi"}])
    assert all("Authorization" not in h for _, _, h, _ in seen)
    assert seen[-1][3]["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen[-1][3]["max_tokens"] == 4096


def test_kind_detection():
    assert LLMConfig(url="https://api.openai.com").is_openai
    assert not LLMConfig(url="http://192.168.0.10:8080").is_openai
    assert LLMConfig(url="http://proxy:9000", kind="openai").is_openai
