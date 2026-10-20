from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import tempfile
import unittest
from unittest.mock import patch

from agentbalance.config import Model, load_config
from agentbalance.planner import make_plan
from agentbalance.providers import ProviderError, complete, parse_response, preflight, http_error_hint
from agentbalance.runner import execute

PROMPT = [{"role": "system", "content": "Solve"}, {"role": "user", "content": "2+2"}]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        if self.path == "/error":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"secret-do-not-log"}')
            return
        if self.path == "/gateway-blocked":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'error code: 1010')
            return
        if self.path == "/redirect":
            self.send_response(307)
            self.send_header("Location", "/chat/completions")
            self.end_headers()
            return
        if self.path == "/responses":
            data = {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "4"}]}], "usage": {"input_tokens": 20, "output_tokens": 30}}
        else:
            data = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 30,
                              "completion_tokens_details": {"reasoning_tokens": 25}}}
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.requests = []
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.model = Model("test", "azure", "deployment", .9, 1, 3, .1,
                           endpoint=self.base + "/chat/completions", api_key_env="TEST_API_KEY",
                           auth_header="api-key", token_limit_field="max_completion_tokens")
        self.environment = patch.dict(os.environ, {"TEST_API_KEY": "test-key"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_azure_http_contract_and_reasoning_tokens_not_double_counted(self):
        result = complete(self.model, PROMPT, 64, 2)
        path, headers, body = self.server.requests[0]
        self.assertEqual(headers["Api-Key"], "test-key")
        self.assertEqual(body["model"], "deployment")
        self.assertEqual(body["max_completion_tokens"], 64)
        self.assertEqual(body["messages"], PROMPT)
        self.assertEqual(result.output_tokens, 30)
        self.assertEqual(result.text, "4")

    def test_azure_responses_http_contract(self):
        model = replace(self.model, endpoint=self.base + "/responses", protocol="responses")
        result = complete(model, PROMPT, 64, 2)
        _, headers, body = self.server.requests[0]
        self.assertEqual(headers["Api-Key"], "test-key")
        self.assertFalse(body["store"])
        self.assertEqual(body["max_output_tokens"], 64)
        self.assertEqual(body["input"], PROMPT)
        self.assertEqual(result.text, "4")
        self.assertEqual(result.input_tokens, 20)

    def test_groq_chat_http_contract(self):
        model = replace(self.model, provider="groq", auth_header="Authorization",
                        api_key_env="GROQ_API_KEY", endpoint=self.base + "/openai/v1/chat/completions")
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-fixture-key"}):
            result = complete(model, PROMPT, 64, 2)
        path, headers, body = self.server.requests[0]
        self.assertEqual(path, "/openai/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer groq-fixture-key")
        self.assertEqual(headers["User-Agent"], "AgentBalance-PoC/0.1.0")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(body["max_completion_tokens"], 64)
        self.assertEqual(body["messages"], PROMPT)
        self.assertNotIn("input", body)
        self.assertNotIn("store", body)
        self.assertFalse(body["stream"])
        self.assertEqual(result.text, "4")
        self.assertEqual(result.usage_source, "provider")

    def test_groq_template_runs_full_graph_with_env_credentials(self):
        source = Path(__file__).resolve().parents[1] / "configs/groq.example.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        for model in data["models"]:
            self.assertEqual(model["endpoint"], "https://api.groq.com/openai/v1/chat/completions")
            model.update(endpoint=self.base + "/openai/v1/chat/completions",
                         input_usd_per_million=0, output_usd_per_million=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groq.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            config = load_config(path)
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-fixture-key", "GROQ_CHEAP_MODEL": "cheap-model",
                                     "GROQ_STRONG_MODEL": "strong-model"}):
            report = execute(config, make_plan(config, "task"), "task")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["answer"], "4")
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(report["model_profiles"][-1]["model"], "strong-model")
        self.assertNotIn("groq-fixture-key", json.dumps(report))

    def test_ollama_needs_no_key(self):
        model = replace(self.model, provider="ollama", api_key_env="", token_limit_field="max_tokens")
        self.assertEqual(complete(model, PROMPT, 64, 2).text, "4")
        self.assertNotIn("Api-Key", self.server.requests[0][1])
        self.assertEqual(self.server.requests[0][2]["max_tokens"], 64)

    def test_missing_usage_is_not_zero_cost(self):
        response = {"choices": [{"message": {"content": "hello"}}]}
        result = parse_response(response, self.model, PROMPT, 64)
        self.assertEqual(result.usage_source, "estimated_reservation")
        self.assertGreater(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 64)

    def test_missing_credentials_abort_before_any_calls(self):
        config = load_config(Path(__file__).resolve().parents[1] / "agentbalance/demo.json")
        model = replace(self.model, id="strong", api_key_env="MISSING_POC_TEST_KEY")
        config = replace(config, models=(config.models[0], model))
        with patch.dict(os.environ, {}, clear=True), patch("agentbalance.providers.complete") as call:
            with self.assertRaises(ProviderError):
                execute(config, make_plan(config, "task"), "task")
            call.assert_not_called()

    def test_errors_are_sanitized_and_not_retried(self):
        with self.assertRaisesRegex(ProviderError, "HTTP 429") as error:
            complete(replace(self.model, endpoint=self.base + "/error"), PROMPT, 64, 2)
        self.assertNotIn("secret-do-not-log", str(error.exception))
        self.assertEqual(len(self.server.requests), 1)

    def test_gateway_error_is_distinguished_from_model_permissions(self):
        with self.assertRaisesRegex(ProviderError, "client signature.*1010"):
            complete(replace(self.model, endpoint=self.base + "/gateway-blocked"), PROMPT, 64, 2)
        self.assertEqual(len(self.server.requests), 1)

    def test_permission_errors_report_scope_without_raw_message(self):
        for code, scope in (("model_permission_blocked_org", "organization"),
                            ("model_permission_blocked_project", "project")):
            body = json.dumps({"error": {"code": code, "message": "secret-do-not-log"}}).encode()
            hint = http_error_hint(403, body)
            self.assertIn(scope + " permissions", hint)
            self.assertNotIn("secret-do-not-log", hint)

    def test_unknown_error_bodies_are_not_exposed(self):
        for body in (b'<html>secret-do-not-log</html>', b'[]', b'null',
                     b'{"error":{"code":"secret-do-not-log"}}', b'\xff'):
            with self.subTest(body=body):
                self.assertEqual(http_error_hint(403, body),
                                 "Access forbidden; check provider permissions and network access restrictions.")

    def test_status_specific_hints(self):
        self.assertIn("Authentication failed", http_error_hint(401, b''))
        self.assertIn("rate limit", http_error_hint(429, b''))

    def test_redirect_does_not_forward_credentials(self):
        with self.assertRaisesRegex(ProviderError, "HTTP 307"):
            complete(replace(self.model, endpoint=self.base + "/redirect"), PROMPT, 64, 2)
        self.assertEqual(len(self.server.requests), 1)

    def test_remote_plain_http_rejected(self):
        with self.assertRaises(ProviderError):
            preflight([replace(self.model, endpoint="http://example.com/chat/completions")])

    def test_env_expansion_for_azure_deployment_and_endpoint(self):
        with patch.dict(os.environ, {"TEST_DEPLOYMENT": "actual-name", "TEST_ENDPOINT": self.base + "/chat/completions"}):
            complete(replace(self.model, model="${TEST_DEPLOYMENT}", endpoint="${TEST_ENDPOINT}"), PROMPT, 64, 2)
        self.assertEqual(self.server.requests[0][2]["model"], "actual-name")

    def test_full_graph_over_http_records_actual_model_without_key(self):
        config = load_config(Path(__file__).resolve().parents[1] / "agentbalance/demo.json")
        cheap = replace(self.model, id="cheap", quality=.35, input_usd_per_million=0, output_usd_per_million=0)
        strong = replace(self.model, id="strong", model="${TEST_DEPLOYMENT}")
        config = replace(config, models=(cheap, strong))
        with patch.dict(os.environ, {"TEST_DEPLOYMENT": "resolved-strong"}):
            result = execute(config, make_plan(config, "task"), "task")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["mode"], "live")
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(result["model_profiles"][1]["model"], "resolved-strong")
        self.assertNotIn("test-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
